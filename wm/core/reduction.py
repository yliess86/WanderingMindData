from pandas import DataFrame
from umap import ParametricUMAP
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COMPONENTS_PCA = 15
COMPONENTS_UMAP = 2


def reduce_pca(df: DataFrame) -> DataFrame:
    """Reduce DataFrame BYOL-A Features using PCA
    
    Arguments:
        df (DataFrame): DataFrame containing BYOL-A features

    Returns:
        df (DataFrame): DataFrame with new PCA features 
    """
    features_byola = [c for c in df.columns if "byola_feature" in c]
    features_pca = [f"pca_feature_{f}" for f in range(COMPONENTS_PCA)]
    
    pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=COMPONENTS_PCA))
    df[features_pca] = pca_pipeline.fit_transform(df[features_byola])

    return df


def reduce_umap(df: DataFrame) -> DataFrame:
    """Reduce DataFrame BYOL-A Features using UMAP
    
    Arguments:
        df (DataFrame): DataFrame containing BYOL-A and PCA features

    Returns:
        df (DataFrame): DataFrame with new UMAP features 
    """
    features_pca = [f"pca_feature_{f}" for f in range(COMPONENTS_PCA)]
    features_umap = [f"umap_feature_{f}" for f in range(COMPONENTS_UMAP)]
    
    umap_pipeline = make_pipeline(StandardScaler(), ParametricUMAP(n_components=COMPONENTS_UMAP))
    df[features_umap] = umap_pipeline.fit_transform(df[features_pca])

    return df


def reduce(df: DataFrame) -> DataFrame:
    """Reduce DataFrame BYOL-A Features using PCA and UMAP
    
    Arguments:
        df (DataFrame): DataFrame containing BYOL-A features

    Returns:
        df (DataFrame): DataFrame with new PCA and UMAP features 
    """
    return reduce_umap(reduce_pca(df))