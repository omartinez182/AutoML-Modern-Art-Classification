# AutoML for Modern Art Classification

This AutoML solution leverages [Ludwig](https://ludwig-ai.github.io/ludwig-docs/0.5/index.html),
to train, evaluate, and deploy a model that classifies artwork based on image embeddings.


## Getting Started

1) Create a virtual environment.

```
cd AutoML-Modern-Art-Classification/
virtualenv venv
```

2) Activate the virtual environment.

```
source venv/bin/activate
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

To train the model you can run:

```
ludwig train --config config.yaml --dataset data/raw/Modern_Art.csv
```

To run hyperparameter optimization you can run:

```
ludwig hyperopt --config config.yaml --dataset data/raw/Modern_Art.csv
```


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