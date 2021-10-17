import os
import pickle

from pandas import DataFrame
from umap import ParametricUMAP
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


COMPONENTS_PCA = 15
COMPONENTS_UMAP = 2


def reduce_pca(df: DataFrame, path: str) -> DataFrame:
    """Reduce DataFrame BYOL-A Features using PCA

    Fit and Transforms BYOL-A Features to PCA if path does not exists.
    Else loads PCA file and Transforms BYOL-A Features to PCA.
    
    Arguments:
        df (DataFrame): DataFrame containing BYOL-A features
        path (str): Path to save or Load PCA weights

    Returns:
        df (DataFrame): DataFrame with new PCA features 
    """
    features_byola = [c for c in df.columns if "byola_feature" in c]
    features_pca = [f"pca_feature_{f}" for f in range(COMPONENTS_PCA)]

    scaler = StandardScaler()
    pca = PCA(n_components=COMPONENTS_PCA)

    if os.path.isfile(path):
        with open(path, "rb") as f:
            pca = pickle.load(path)

        pipeline = make_pipeline(scaler, pca)
        df[features_pca] = pipeline.transform(df[features_byola])
    
    else:
        pipeline = make_pipeline(scaler, pca)
        df[features_pca] = pipeline.fit_transform(df[features_byola])

        with open(path, "wb") as f:
            pickle.dump(pca, f)

    return df


def reduce_umap(df: DataFrame, path: str) -> DataFrame:
    """Reduce DataFrame BYOL-A Features using UMAP

    Fit and Transforms PCA Features to UMAP if path does not exists.
    Else loads UMAP file and Transforms PCA Features to UMAP.
    
    Arguments:
        df (DataFrame): DataFrame containing BYOL-A and PCA features
        path (str): Path to save or Load UMAP weights

    Returns:
        df (DataFrame): DataFrame with new UMAP features 
    """
    features_pca = [f"pca_feature_{f}" for f in range(COMPONENTS_PCA)]
    features_umap = [f"umap_feature_{f}" for f in range(COMPONENTS_UMAP)]
    
    scaler = MinMaxScaler()
    umap = ParametricUMAP(n_components=COMPONENTS_UMAP)

    if os.path.isfile(path):
        with open(path, "rb") as f:
            umap = pickle.load(path)

        pipeline = make_pipeline(scaler, umap)
        df[features_umap] = pipeline.transform(df[features_pca])

    else:
        pipeline = make_pipeline(scaler, umap)
        df[features_umap] = pipeline.fit_transform(df[features_pca])

        with open(path, "wb") as f:
            pickle.dump(umap, f)

    return df


def reduce(df: DataFrame, pca_path: str, umap_path: str) -> DataFrame:
    """Reduce DataFrame BYOL-A Features using PCA and UMAP

    Fit and Transforms BYOL-A Features to PCA if pca_path does not exists.
    Else loads PCA file and Transforms BYOL-A Features to PCA.

    Fit and Transforms PCA Features to UMAP if umap_path does not exists.
    Else loads UMAP file and Transforms PCA Features to UMAP.

    Arguments:
        df (DataFrame): DataFrame containing BYOL-A features
        pca_path (str): Path to save or Load PCA weights
        umap_path (str): Path to save or Load UMAP weights

    Returns:
        df (DataFrame): DataFrame with new PCA and UMAP features 
    """
    return reduce_umap(reduce_pca(df, pca_path), umap_path)