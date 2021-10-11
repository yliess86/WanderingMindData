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

<span id="description"></span>
## Description

The data pipeline is used to extract audio features from a set of audio files.
The audio files need to be cropped to `2 min`.
They are then split into `5 sec` chunks.
Those chunks are used to extract `2048` [BYOL-A] features.
Those features are then reduced to `2` dimensions using a mix of [PCA] and [UMAP].
Every chunk is labeled using [PANNs].
The final output is a `pandas` `DataFrame` containing the reference to every audio file, their corresponding chunks, BYOL-A features (`2048`), PCA features (`15`), UMAP features (`2`), and PAANs label. 

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