# Plant Disease Status Classification by CNN

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short project exploring the use of TensorFlow/Keras in generating a Convolutional Neural Network (CNN) that can classify leaf images according to their disease status.

## Data

Original data is taken from [Kaggle](https://www.kaggle.com/datasets/amandam1/healthy-vs-diseased-leaf-image-dataset/data)

## Project Organization

```
├── LICENSE                <- License for use.
├── Makefile               <- Makefile with convenience commands.
├── README.md              <- The top-level README.
├── data
│   ├── raw                <- The original raw data from Kaggle.
│   └── interim            <- The reduced images.
│
├── notebooks              <- Jupyter notebooks.
│
├── pyproject.toml         <- Project configuration file with package metadata for
│                             plant_classification and configuration for tools like black.
│
├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures            <- Generated graphics and figures to be used in reporting.
│
├── environment.yml        <- The requirements file for reproducing the analysis environment, e.g.
│                             generated with `conda env create -f environment.yml` and activated by
│                             `conda activate plant_classification`.
│
├── setup.cfg              <- Configuration file for flake8.
│
└── plant_classification   <- Source code for use in this project.
    │
    ├── __init__.py              <- Makes plant_classification a Python module.
    │
    ├── config.py                <- Store useful variables and configuration.
    │
    ├── dataset.py               <- Reduce data.
    │
    ├── train.py                 <- Train the CNN model.
    │
    ├── evaluate.py              <- Evaluate the model accuracy/loss and produce plots.
    │
    ├── predict.py               <- Use the model to make predictions of unseen images.
    │
    ├── data                     <- Useful functions to generate reduced data.
    │   └── reduce_data.py     
    │
    ├── model                    <- Useful functions to generate and run model.
    │   ├── compile_data.py
    │   └── create_model.py
    │
    ├── evaluate                 <- Useful functions to evaluate model and produce plots.
    │   ├── make_statistics.py
    │   └── make_figures.py
    │
    └── test                    
        ├── test_reduce_data.py  <- Testing script for functions in data/reduce_data.py
        ├── test_compile_data.py <- Testing script for functions in model/compile_data.py
        ├── test_create_model.py <- Testing script for functions in model/create_model.py
        └── test_vars.py         <- Script to generate test variables
```
