# AutoML for Modern Art Classification

This AutoML solution leverages [Ludwig](https://ludwig-ai.github.io/ludwig-docs/0.5/index.html),
to train, evaluate, and deploy a model that classifies artwork based on image embeddings.

The goal is to do binary classification to predict whether a painting belongs to the "Oil" category or not. In this process, we test different architectures and take advantage of Ludwig's AutoML features to select the best one based on performance (accuracy).


## Getting Started

1) Create a virtual environment.

```
cd AutoML-Modern-Art-Classification/
virtualenv venv
```

2) Activate the virtual environment.
  
- Mac:
```
source venv/bin/activate
```

- Windows :

```
.\venv\Scripts\activate
```

3) Then run: **(recommended)**

```
make all
```

- or install the requirements using pip.

```
pip install -r requirements.txt
```


## Usage

To train the model and test different architectures you can run:

```
python model_training.py
```

The results of the experiment are saved under ```data``` -> ```results```.

Additionally, plots for the learning curves (loss and accuracy) can be found under the ```visualizations``` folder.

## Dataset

To get the required data, I used the information provided publicly in [The Museum of Modern Art (MoMA) Collection repository](https://github.com/MuseumofModernArt/collection/tree/v1.6); then I scraped publicly available images for each artwork and also the location within the museum.


## Disclaimer

This work is for educational purposes only and the MoMA has not endorsed this work. The source can be found in the citation below.


## Citations
```
@dataset{The Museum of Modern Art (MoMA),
  title={The Museum of Modern Art (MoMA) Collection},
  repository={https://github.com/MuseumofModernArt/collection/tree/v1.62},
}

@misc{Molino2019,
  author = {Piero Molino and Yaroslav Dudin and Sai Sumanth Miryala},
  title = {Ludwig: a type-based declarative deep learning toolbox},
  year = {2019},
  eprint = {arXiv:1909.07930},
}
```