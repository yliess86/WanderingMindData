import matplotlib
import matplotlib.pyplot as plt
import os

from IPython.display import display
from ipywidgets.widgets import Button, IntProgress, IntSlider, Text, VBox
from pandas import DataFrame, read_csv, read_pickle
from tqdm.auto import tqdm
from wm.core.features import get_files, launch_features_workers
from wm.core.reduction import reduce
from wm.gui.standard import StandardTabsWidget


class Config(StandardTabsWidget):
    """Configuration Widget"""

    def __init__(self) -> None:
        super().__init__()

    def setup_widgets(self) -> None:
        """Setup Configuration Widgets"""
        self.register_widget("pca", Text(placeholder="Enter pca file", value="pca.pkl", description="PCA"))
        self.register_widget("umap", Text(placeholder="Enter umap file", value="umap.pkl", description="UMAP"))
        
        self.register_widget("root", Text(placeholder="Enter audio folder", value="audios", description="Audios"))
        self.register_widget("csv", Text(placeholder="Enter identifier csv file", value="audios.csv", description="Identifiers"))
        self.register_widget("data", Text(placeholder="Enter data pickle file", value="data.pkl", description="Data"))

        self.register_widget("batch_size", IntSlider(value=64, min=2, max=1024, step=2, description="Batch Size"))
        self.register_widget("jobs", IntSlider(value=1, min=1, max=12, step=1, description="Jobs"))

    def setup_tabs(self) -> None:
        """Setup Configuration Tabs"""
        self.register_tab("weights", 1, 2, ["pca", "umap"])
        self.register_tab("I/O", 1, 3, ["root", "csv", "data"])
        self.register_tab("hyperparameters", 1, 2, ["batch_size", "jobs"])


class App(StandardTabsWidget):
    """Application Widget"""
    
    def __init__(self) -> None:
        super().__init__()
        self.config = Config()
        self.df: DataFrame = None
        self.classes = read_csv("classes.csv")
        
        path = self.config.data()
        if os.path.isfile(path):
            self.df = read_pickle(path)
            print(f"[DataFrame] Loaded from {path}")
            print(self.df.head())

    def setup_widgets(self) -> None:
        """Setup Application Widgets"""
        self.register_widget("features", Button(description="Features", icon="braille"))
        self.register_widget("reduction", Button(description="Reduction", icon="filter"))
        self.register_widget("pbar", IntProgress(min=0, max=1, disabled=True, visible=False, description="Features"))
        
        self.w_features.on_click(lambda _: self.on_features())
        self.w_reduction.on_click(lambda _: self.on_reduction())

    def setup_tabs(self) -> None:
        """Setup Application Tabs"""
        self.register_tab("actions", 1, 2, ["features", "reduction"])

    def update_callback(self) -> None:
        """Update ProgressBar"""
        self.w_pbar.value += 1

    def on_features(self) -> None:
        """Button on Features (Performs BYOLA and PANNS Feature Extraction)"""
        root = self.config.root()
        csv = self.config.csv()
        batch_size = self.config.batch_size()
        jobs = self.config.jobs()

        assert len(csv) and len(root), "[ERROR] No `root` or `csv` procided"

        files = get_files(root, csv)
        self.w_pbar.max = len(files)
        self.w_pbar.disabled = False
        self.w_pbar.visible = True

        callbacks = [lambda *args, **kwargs: self.update_callback()]
        self.df = launch_features_workers(root, csv, batch_size, jobs, callbacks)
        print(self.df.head())

        path = self.config.data()
        self.df.to_pickle(path)
        print(f"[DataFrame] Data Saved as {path}")

        self.w_pbar.disabled = True
        self.w_pbar.visible = False

    def on_reduction(self) -> None:
        """Button on Reduction (Performs PCA and UMAP Feature Reduction)"""
        assert self.df is not None and len(self.df), "[ERROR] No DataFrame available"
        
        pca_path = self.config.pca()
        umap_path = self.config.umap()

        reduce(self.df, pca_path, umap_path)
        print(self.df.head())

        path = self.config.data()
        self.df.to_pickle(path)
        print(f"[DataFrame] Data Saved as {path}")

        print(f"[DataFrame] Plot")
        self.plot()

    def plot(self) -> None:
        plt.style.use("dark_background")
        plt.figure()
        
        pbar = tqdm(self.df.panns_label.unique(), desc="Scatter")
        for l in pbar:
            ldf = self.df[self.df.panns_label == l]
            name = self.classes[self.classes.index == l].display_name.values[0]
            plt.scatter(x=ldf.umap_feature_0, y=ldf.umap_feature_1, label=name)
            pbar.set_postfix(label=name)

        plt.legend(fancybox=True, shadow=False)
        plt.show()

    def display(self) -> None:
        """Application Display Ovewrite"""
        display(VBox([self.config.app, self.app, self.w_pbar]))