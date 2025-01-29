# test_make_dataset.py

import pytest

import sys
sys.path.append("../..") # Adds higher directory to python modules path.

from plant_classification.model import create_model
from plant_classification.test import test_vars
from plant_classification.config import INTERIM_DATA_DIR
import pandas as pd
import keras

def test_create_model():
    """
    Function to test create_model from plant_classification.model.create_model
    """
    image_shape, num_classes, _ = test_vars.get_test_vars()

    # test that function returns an error when invalid model_type specified
    with pytest.raises(ValueError) as excinfo:
        test_model = create_model('not_in_list', image_shape, num_classes, print_summary=0)
    assert str(excinfo.value) == "model_type must be 'simple' or 'pretrained'."

    # test that output is of type keras.Sequential
    test_model = create_model('simple', image_shape, num_classes, print_summary=0)
    assert isinstance(test_model, keras.src.models.sequential.Sequential)



def test_create_simple_model():
    """
    Function to test create_simple_model from plant_classification.model.create_model
    """
    image_shape, num_classes, _ = test_vars.get_test_vars()
    test_model = create_model.create_simple_model(image_shape, num_classes)

    # test that output is of type keras.Sequential
    assert isinstance(test_model, keras.src.models.sequential.Sequential)



def test_init_pretrained_model():
    """
    Function to test init_pretrained_model from plant_classification.model.create_model
    """

    image_shape, num_classes, _ = test_vars.get_test_vars()
    test_model = create_model.init_pretrained_model(image_shape, num_classes)

    # test that output is of type keras.Sequential
    assert isinstance(test_model, keras.src.models.sequential.Sequential)
