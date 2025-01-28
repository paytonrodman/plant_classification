# test_make_dataset.py

import pytest

import sys
sys.path.append("../..") # Adds higher directory to python modules path.

from plant_classification.model import create_model
from plant_classification.test import test_vars
from plant_classification.config import INTERIM_DATA_DIR
import pandas as pd
import keras

def test_create_simple_model():
    """
    Function to test create_simple_model from plant_classification.model.create_model
    """
    image_shape, num_classes, _ = test_vars.get_test_vars()
    test_model = create_model.create_simple_model(image_shape, num_classes)

    # test that output is of type keras.Sequential
    assert isinstance(test_model, keras.src.models.sequential.Sequential)
