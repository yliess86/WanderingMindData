import os
import pandas as pd
import torch
import torchaudio
import torch.jit as jit

from glob import glob
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import Queue
from pandas import DataFrame
from torch import device, Tensor
from torch.cuda.amp import autocast
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
        self.d = self.d

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
        x = x.premute(0, 3, 2, 1)
        
        B, T, D, C = x.shape
        x = x.reshape((B, T, D * C))
        x = self.fc(x)

        x1, _ = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)

        return x1 + x2


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
    normalizer = PrecomputedNorm(byola_config.precomputed_norm)
    to_mel_spectrogram = MelSpectrogram(**byola_config.mel_spectrogram)
    
    eps = torch.finfo(torch.float16)
    byola = BYOLA(d=byola_config.feature_d)
    byola.load_weights(byola_path, device("cuda"))
    byola = jit.script(byola).to(f"cuda:{device_id}")
    byola.infer = lambda _, wav: byola(normalizer((to_mel_spectrogram(wav) + eps).log()))

    panns = jit.load(panns_path).to(f"cuda:{device_id}")
    panns.infer = lambda _, wav: torch.argmax(torch.softmax(panns(wav)[0], dim=1), dim=1)

    df = None
    while not files.empty():
        file = files.get()

        wav, sr = torchaudio.load(file)
        wav = Resample(sr, SAMPLE_RATE)
        wav = wav.mean(0).unsqueeze(0)

        windows = Queue()
        for s in range(0, wav.size(1), HOP):
            e = min(s + WINDOW, wav.size(1) - 1)
            window = torch.zeros((wav.size(0), WINDOW))
            window[:, :e - s] = wav[:, s:e]
            windows.put(window)

        byola_data = []
        panns_data = []
        while not windows.empty():
            batch = []
            while len(batch) < batch_size and not windows.empty():
                batch.append(windows.get())

            batch = torch.stack(batch, dim=0).to(f"cuda:{device_id}")
            with autocast():
                byola_data.append(byola.infer(batch).cpu())
                panns_data.append(panns.infer(batch).cpu())

        byola_data = torch.cat(byola_data).numpy()
        panns_data = torch.cat(panns_data).numpy()
        byola_columns = [f"byola_feature_{f}" for f in range(byola_data.shape[1])]

        file_df = DataFrame(data=byola_data, columns=byola_columns)
        file_df["panns_label"] = panns_data
        file_df["file_path"] = [file] * len(file_df)
        file_df["file_sample_start"] = list(range(0, wav.size(1), HOP))
        file_df["file_sample_duration"] = [WINDOW] * len(file_df)

        df = pd.concat([df, file_df], ignore_index=True) if df else file_df
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
    for file in get_files(root, csv):
        file_queue.put(file)
    
    model_args = BYOLAConfig(), BYOLA_MODEL, PANNS_MODEL
    work_args = *model_args, file_queue, batch_size, callbacks
    
    df = None
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(features_work, j, work_args) for j in range(jobs)]
        for future in as_completed(futures):
            files_df = future.result()
            df = files_df if df else pd.concat([df, files_df], ignore_index=True)

    return df