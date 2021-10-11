from IPython.display import display
from ipywidgets.widgets import Button, IntProgress, IntSlider, Text, VBox
from pandas import DataFrame
from wm.core.features import get_files, launch_features_workers
from wm.core.reduction import reduce
from wm.gui.standard import StandardTabsWidget


class Config(StandardTabsWidget):
    """Configuration Widget"""

    def __init__(self) -> None:
        super().__init__()

    def setup_widgets(self) -> None:
        """Setup Configuration Widgets"""
        self.register_widget("root", Text(placeholder="Enter mp3 folder", value="mp3_trim", description="MP3"))
        self.register_widget("csv", Text(placeholder="Enter csv file", value="aporee.csv", description="CSV"))
        self.register_widget("pickle", Text(placeholder="Enter pickle file", value="data.pickle", description="Pickle"))
        self.register_widget("batch_size", IntSlider(value=64, min=2, max=1024, step=2, description="Batch Size"))
        self.register_widget("jobs", IntSlider(value=1, min=1, max=12, step=1, description="Jobs"))

    def setup_tabs(self) -> None:
        """Setup Configuration Tabs"""
        self.register_tab("files", 1, 3, ["root", "csv", "pickle"])
        self.register_tab("hyperparameters", 1, 2, ["batch_size", "jobs"])


class App(StandardTabsWidget):
    """Application Widget"""
    
    def __init__(self) -> None:
        super().__init__()
        self.config = Config()
        
        self.df: DataFrame = None

    def setup_widgets(self) -> None:
        """Setup Application Widgets"""
        self.register_widget("features", Button(description="Features", icon="braille"))
        self.register_widget("reduction", Button(description="Reduction", icon="filter", disabled=True))
        self.register_widget("save", Button(description="Save", icon="floppy-o", disabled=True))
        self.register_widget("pbar", IntProgress(min=0, max=1, disabled=True, visible=False, description="Features"))
        
        self.w_features.on_click(lambda _: self.on_features())
        self.w_reduction.on_click(lambda _: self.on_reduction())
        self.w_save.on_click(lambda _: self.on_save())

    def setup_tabs(self) -> None:
        """Setup Application Tabs"""
        self.register_tab("actions", 1, 3, ["features", "reduction", "save"])

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

        self.w_features.disabled = True
        self.w_reduction.disabled = False
        self.w_save.disabled = False

        self.w_pbar.disabled = True
        self.w_pbar.visible = False

    def on_reduction(self) -> None:
        """Button on Reduction (Performs PCA and UMAP Feature Reduction)"""
        assert self.df is not None and len(self.df), "[ERROR] No DataFrame available"
        
        reduce(self.df)
        print(self.df.head())

        self.w_features.disabled = True
        self.w_reduction.disabled = True
        self.w_save.disabled = False

    def on_save(self) -> None:
        """Button on Save (Saves DataFrame to `pickle` file)"""
        assert self.df is not None and len(self.df), "[ERROR] No DataFrame available"

        path = self.config.pickle()
        self.df.to_pickle(path)

    def display(self) -> None:
        """Application Display Ovewrite"""
        display(VBox([self.config.app, self.app, self.w_pbar]))