import os
import pandas as pd
import re
import torch
import torchaudio
import torch.jit as jit

from glob import glob
from concurrent.futures import as_completed, ThreadPoolExecutor
from multiprocessing import Queue
from pandas import DataFrame
from torch import device, Tensor
from torch.nn import BatchNorm2d, Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential
from torchaudio.transforms import MelSpectrogram, Resample
from typing import Any, Callable, NamedTuple, List, Tuple


SAMPLE_RATE = 16_000
WINDOW = int(SAMPLE_RATE * 5.)
HOP = WINDOW // 2

WEIGHTS = "./weights"
BYOLA_MODEL = os.path.join(WEIGHTS, "byola.pt")
PANNS_MODEL = os.path.join(WEIGHTS, "panns.ts")


class PrecomputedNorm(Module):
    """Precomputed Norm for Mel Spectrogram Module

    Arguments:
        stats (Tuple[float, float]): precomputed melspec mean and std
        axis (Tuple[int, int]): axis on which to compute mean and std (default: (1, 2))
    """

    def __init__(self, stats: Tuple[float, float], axis: Tuple[int, int] = (1, 2)) -> None:
        super().__init__()
        self.mean, self.std = stats
        self.axis = axis

    def forward(self, mel_spectrogram: Tensor) -> Tensor:
        """Forward Pass

        Arguments:
            mel_spectrogram (Tensor): mel spectrogram to be normalized (B, C, M, T)
            
        Returns:
            mel_spectrogram (Tensor): normalized mel spectrogram (B, C, M, T)
        """
        mean = mel_spectrogram.mean(dim=self.axis, keepdims=True)
        std = mel_spectrogram.std(dim=self.axis, keepdims=True)
        return (mel_spectrogram - mean) / std


class BYOLA(Module):
    """BYOL-A Module

    Arguments:
        n_mels (int): number of mel (default: 64)
        d (int): hidden dimension (default: 512)
    """

    def __init__(self, n_mels: int = 64, d: int = 512) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.d = d

        self.features = Sequential(
            Conv2d( 1, 64, 3, stride=1, padding=1), BatchNorm2d(64), ReLU(), MaxPool2d(2, stride=2),
            Conv2d(64, 64, 3, stride=1, padding=1), BatchNorm2d(64), ReLU(), MaxPool2d(2, stride=2),
            Conv2d(64, 64, 3, stride=1, padding=1), BatchNorm2d(64), ReLU(), MaxPool2d(2, stride=2),
        )

        self.fc = Sequential(
            Linear(64 * (self.n_mels // (2 ** 3)), self.d), ReLU(), Dropout(p=.3),
            Linear(self.d, self.d), ReLU(),
        )

    def forward(self, mel_spectrogram: Tensor) -> Tensor:
        """Forward Pass
        
        Arguments:
            mel_spectrogram (Tensor): normalized log mel spectrogram (B, C, M, T)

        Returns:
            features (Tensor): BYOL-A features (B, D)
        """
        x = self.features(mel_spectrogram)
        x = x.permute(0, 3, 2, 1)
        
        B, T, D, C = x.shape
        x = x.reshape((B, T, D * C))
        x = self.fc(x)

        x1, _ = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)

        return x1 + x2

    def load_weights(self, path: str, d: device) -> "BYOLA":
        """Utility to load a weight file to a device
        
        Arguments:
            path (str): path to the weights
            d (device): device (cpu, cuda, ...)

        Returns:
            byola (BYOLA): model with loaded weights
        """

        state_dict = torch.load(path, map_location=d)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        weights = {}
        for k in state_dict:
            m = re.search(r"(^fc\.|\.fc\.|^features\.|\.features\.)", k)
            if m is None: continue
            new_k = k[m.start():]
            new_k = new_k[1:] if new_k[0] == "." else new_k
            weights[new_k] = state_dict[k]
        
        self.load_state_dict(weights)
        self.eval()

        return self


class BYOLAMelSpectrogramConfig(NamedTuple):
    """BYOLA Mel Spectrogram Config"""
    sample_rate: int = 16_000
    n_fft: int = 1_024
    win_length: int = 1_024
    hop_length: int = 160
    n_mels: int = 64
    f_min: int = 60
    f_max: int = 7_800


class BYOLAConfig(NamedTuple):
    """BYOLA Config"""
    mel_spectrogram: BYOLAMelSpectrogramConfig = BYOLAMelSpectrogramConfig()
    precomputed_norm: Tuple[float, float] = (-5.4919195, 5.0389895)
    feature_d: int = 2_048


def get_files(root: str, csv: str) -> List[str]:
    """Get Files from CSV

    Arguments:
        root (str): mp3 root folder
        csv (str): csv with mp3 identifiers

    Returns:
        files (List[str]): mp3 files from root selected within indentifiers
    """
    df = pd.read_csv(csv)
    path = lambda idf: os.path.join(root, f"{idf}/**/*.mp3")
    return [f for idf in df.identifier for f in glob(path(idf), recursive=True)]


@torch.no_grad()
def features_work(
    device_id: int,
    byola_config: BYOLAConfig,
    byola_path: str,
    panns_path: str,
    files: Queue,
    batch_size: int,
    callbacks: List[Callable[..., Any]],
) -> DataFrame:
    """Features Work

    Arguments:
        device_id (int): cuda device id
        byola_config (BYOLAConfig): byola config
        byola_path (str): BYOLA model weights path
        panns_path (str): PANNS model weights path
        files (Queue): file queue
        batch_size (int): batch size
        callbacks (List[Callable[..., Any]]): callbacks launched when worker done
    """
    d = f"cuda:{device_id}"

    normalizer = PrecomputedNorm(byola_config.precomputed_norm).to(d)
    melspec = MelSpectrogram(
        sample_rate=byola_config.mel_spectrogram.sample_rate,
        n_fft=byola_config.mel_spectrogram.n_fft,
        win_length=byola_config.mel_spectrogram.win_length,
        hop_length=byola_config.mel_spectrogram.hop_length,
        n_mels=byola_config.mel_spectrogram.n_mels,
        f_min=byola_config.mel_spectrogram.f_min,
        f_max=byola_config.mel_spectrogram.f_max,
    ).to(d)
    
    eps = torch.finfo(torch.float16).eps
    byola = BYOLA(d=byola_config.feature_d)
    byola.load_weights(byola_path, device("cuda"))
    byola = byola.to(d)
    byola.logml = lambda wav: torch.log(melspec(wav) + eps)
    byola.featu = lambda wav: byola(normalizer(byola.logml(wav)))

    panns = jit.load(panns_path).to(d)
    panns.score = lambda wav: panns(wav)[0]
    panns.proba = lambda wav: torch.softmax(panns.score(wav), dim=1)
    panns.label = lambda wav: torch.argmax(panns.proba(wav), dim=1)

    df = None
    while files.qsize() > 0:
        file = files.get()

        wav, sr = torchaudio.load(file)
        wav = Resample(sr, SAMPLE_RATE)(wav)
        wav = wav.mean(0).unsqueeze(0)

        windows = Queue()
        for s in range(0, wav.size(1), HOP):
            e = min(s + WINDOW, wav.size(1) - 1)
            window = torch.zeros((wav.size(0), WINDOW))
            window[:, :e - s] = wav[:, s:e]
            windows.put(window)

        byola_data = []
        panns_data = []
        while windows.qsize() > 0:
            batch = []
            while len(batch) < batch_size and windows.qsize() > 0:
                batch.append(windows.get())

            batch = torch.stack(batch, dim=0).to(d)
            with torch.inference_mode():
                byola_data.append(byola.featu(batch).cpu())
                panns_data.append(panns.label(batch.squeeze(1)).cpu())

        byola_data = torch.cat(byola_data).numpy()
        panns_data = torch.cat(panns_data).numpy()
        byola_columns = [f"byola_feature_{f}" for f in range(byola_data.shape[1])]

        file_df = DataFrame(data=byola_data, columns=byola_columns)
        file_df["panns_label"] = panns_data
        file_df["file_path"] = [file] * len(file_df)
        file_df["file_sample_start"] = list(range(0, wav.size(1), HOP))
        file_df["file_sample_duration"] = [WINDOW] * len(file_df)

        if df is None: df = file_df
        else: df = pd.concat([df, file_df], ignore_index=True)
        for callback in callbacks: callback()

    return df



def launch_features_workers(
    root: str,
    csv: str,
    batch_size: int,
    jobs: int,
    callbacks: List[Callable[..., Any]],
) -> DataFrame:
    """Launch Features Workers

    Arguments:
        root (str): mp3 root folder
        csv (str): csv with mp3 identifiers
        batch_size (int): batch size for BYOLA
        jobs (int): number cuda device / processes to use
        callbacks (List[Callable[..., Any]]): callbacks launched when worker done
    """
    file_queue = Queue()
    files = get_files(root, csv)
    for file in files:
        file_queue.put(file)
    
    model_args = BYOLAConfig(), BYOLA_MODEL, PANNS_MODEL
    work_args = *model_args, file_queue, batch_size, callbacks
    
    df = None
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(features_work, j, *work_args) for j in range(jobs)]
        for future in as_completed(futures):
            files_df = future.result()
            if df is None: df = files_df
            else: df = pd.concat([df, files_df], ignore_index=True)

    return df