[![Build test with Github Actions](https://github.com/omartinez182/AutoML-Modern-Art-Classification/actions/workflows/main.yml/badge.svg)](https://github.com/omartinez182/AutoML-Modern-Art-Classification/actions/workflows/main.yml)


# AutoML for Modern Art Classification

This AutoML solution leverages [Ludwig](https://ludwig-ai.github.io/ludwig-docs/0.5/index.html),
to train, evaluate, and deploy a model that classifies artwork based on image embeddings.

The goal is to do binary classification to predict whether a painting belongs to the "Oil" category or not. In this process, we test different architectures and take advantage of Ludwig's AutoML features to select the best one based on performance (accuracy).

![AutoML_Art_Classification](https://user-images.githubusercontent.com/63601717/177655332-b7de4bb8-6889-4586-b07c-8246c99815c1.jpeg)


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


## Results
![Accuracy Learning Curve](https://raw.githubusercontent.com/omartinez182/AutoML-Modern-Art-Classification/main/visualizations/learning_curves_label_accuracy.png)

The results of the experiment are saved under ```data``` -> ```results```.

Additionally, plots for the learning curves (loss and accuracy) can be found under the ```visualizations``` folder.


## Dataset

To get the required data I used the dataset placed under the public domain from [The Museum of Modern Art (MoMA) Collection repository](https://github.com/MuseumofModernArt/collection/tree/v1.6).


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
