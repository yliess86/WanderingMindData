# Wandering Mind: Data

This repository contains the data processing pipeline used in the [Wandering Mind] project.

*Table of Content*
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Description](#description)
- [References](#references)

<span id="installation"></span>
## Installation

This implementation has been tested on `Ubuntu 20.04` with `Python 3.8`, and `torch 1.9`.
Install required package first `pip3 install -r requirements.txt`.
You may use `pyenv` or `conda` to avoid confilcts with your environement.

Download the [models weights](https://s3.dvic.devinci.fr/public/wmdata_weights.tar.gz) and place the `weights` folder into the project folder.

> Cuda enabled GPU is needed.

<span id="quickstart"></span>
## Quickstart

*Jupyter Notebook*
You first need to activate Jupyter Notebook Widgets.
Then, start the [wm.ipynb](wm.ipynb) notebook `jupyter notebook wm.ipynb`.
Run all the Cells and you will be granted with GUIs allowing you to configure, and launch the data pipeline.

*Manual*
```python
from wm.core import launch_features_workers, reduce

df = reduce(
    launch_features_workers(
        "audio_trim",          # Audio Folder
        "audio.csv",           # CSV w/ Audio Identifiers
        batch_size=32,         # Batch Size
        jobs=1,                # Number of GPU to use
    ),
    "pca.pkl",                 # Path to Save|Load the PCA weights
    "umap.pkl",                # Path to Save|Load the UMAP weights
)

print(df.head())               # Print the DataFrame Head (~5 Rows)
df.to_pickle("data.pkl")       # Save the DataFrame to Pickle
```

<span id="description"></span>
## Description

The data pipeline is used to extract audio features from a set of audio files.
The audio files need to be cropped to `2 min`.
They are then split into `5 sec` chunks.
Those chunks are used to extract `2048` [BYOL-A] features.
Those features are then reduced to `2` dimensions using a mix of [PCA] and [UMAP].
Every chunk is labeled using [PANNs].
The final output is a `pandas` `DataFrame` containing the reference to every audio file, their corresponding chunks, BYOL-A features (`2048`), PCA features (`15`), UMAP features (`2`), and PAANs label.

### Data

This section describes the data preparation needed to make this framework usable with your data.

Audio files need to be placed into a main folder `audios` (name is arbitrary).
Each of those audio files also needs to be placed into a indetification folder `subdataset/id_01/01.mp3`.
In this example the full path of the audio file is then `audios/subdataset/id_01/01.mp3` with `subdataset/id_01` being its identifer.

```txt
.
├── audios
    ├── aporee
        ├── 01
            └── 01.mp3
        ├── 02
            └── 02.wav
        └── ...
    └── china
        ├── 01
            └── 01.m4e
        ├── 02
            └── 02.flac
        └── ...
```

A second requirements is the identifier file `audios.csv` (name is arbitrary).
This file will be used to select the audios that will be fitted and/or transformed.

```csv
whatever1,identifier,whatever2
w1_a01,aporee/01,w2_a01
w1_a02,aporee/02,w2_a02
...
w1_c01,china/01,w2_c01
w1_c02,china/02,w2_c02
...
```

<span id="references"></span>
## References

- [Wandering Mind]
- [BYOL-A]
- [PCA]
- [UMAP]
- [PANNs]


[Wandering Mind]: https://ger.sh/The-Wandering-Mind
[BYOL-A]: https://github.com/nttcslab/byol-a
[PCA]: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
[UMAP]: https://github.com/lmcinnes/umap
[PANNs]: https://github.com/qiuqiangkong/audioset_tagging_cnn