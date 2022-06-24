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