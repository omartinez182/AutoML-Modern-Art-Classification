import glob
import logging
import os
import shutil
import yaml
import pandas as pd
from collections import namedtuple
from ludwig.api import LudwigModel
from ludwig.constants import TRAINER
from ludwig.utils.data_utils import load_json
from ludwig.visualize import learning_curves, confusion_matrix
from sklearn.model_selection import train_test_split

# Clean out old results
shutil.rmtree("./results", ignore_errors=True)
shutil.rmtree("./visualizations", ignore_errors=True)

file_list = glob.glob("./data/*.json")
file_list += glob.glob("./data/*.hdf5")
for f in file_list:
    try:
        os.remove(f)
    except FileNotFoundError:
        pass

# Read in base configd
with open("./config.yaml") as f:
    base_model = yaml.safe_load(f.read())

# Specify named tuple to keep track of training results
TrainingResult = namedtuple("TrainingResult", ["name", "train_stats"])

# Specify alternative architectures to test
FullyConnectedLayers = namedtuple("FullyConnectedLayers", ["name", "fc_layers"])

list_of_fc_layers = [
    FullyConnectedLayers(name="Option1", fc_layers=[{"output_size": 64}]),
    FullyConnectedLayers(name="Option2", fc_layers=[{"output_size": 128}, {"output_size": 64}]),
    FullyConnectedLayers(name="Option3", fc_layers=[{"output_size": 128}]),
]

list_of_train_stats = []

# Load dataset
df = pd.read_csv("data/raw/Modern_Art.csv")
X, y = df.image_path, df.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
training_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

# Change directory to save logs inside the data folder
os.chdir('data/')

# Train models
for model_option in list_of_fc_layers:
    print(">>>> training: ", model_option.name)

    # Set up Python dictionary to hold model training parameters
    config = base_model.copy()
    config["input_features"][0]["fc_layers"] = model_option.fc_layers
    config[TRAINER]["epochs"] = 5

    # Define Ludwig model object that drive model training
    model = LudwigModel(config, logging_level=logging.INFO)

    # Initiate model training
    train_stats, _, _ = model.train(
        training_set=training_set,
        test_set=test_set,
        experiment_name="multiple_experiment",
        model_name=model_option.name,
    )

    # Save training stats for later use
    list_of_train_stats.append(TrainingResult(name=model_option.name, train_stats=train_stats))

    print(">>>>>>> completed: ", model_option.name, "\n")

os.chdir('../')

# Generating learning curves from training
option_names = [trs.name for trs in list_of_train_stats]
train_stats = [trs.train_stats for trs in list_of_train_stats]
learning_curves(
    train_stats, "Survived", model_names=option_names, output_directory="./visualizations", file_format="png"
)

# Confustion matrix by model
train_metadata_json = load_json('./data/results/multiple_experiment_Option1/model/training_set_metadata.json')
models_list = ['Option1', 'Option2', 'Option3']