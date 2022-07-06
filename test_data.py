import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDuplicates
from deepchecks.tabular.checks import ConflictingLabels

# Load dataset
df = pd.read_csv("data/raw/Modern_Art.csv")
dataset = Dataset(
    df, cat_features=["image_path"], label="label", features=["image_path"]
)

# References:
# https://docs.deepchecks.com/stable/checks_gallery/tabular/data_integrity/plot_data_duplicates.html
# https://docs.deepchecks.com/stable/checks_gallery/tabular/data_integrity/plot_conflicting_labels.html

# Run the checks
duplicates = DataDuplicates().run(dataset).display
conflicting_labels = ConflictingLabels().run(dataset).display

assert len(duplicates) == 0
assert len(conflicting_labels) == 0
