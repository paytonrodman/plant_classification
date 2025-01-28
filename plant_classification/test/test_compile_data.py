# test_make_dataset.py

import pytest

import sys
sys.path.append("../..") # Adds higher directory to python modules path.

from plant_classification.model import compile_data
from plant_classification.test import test_vars
from plant_classification.config import INTERIM_DATA_DIR
import pandas as pd

def test_build_dataframe():
    """
    Function to test build_dataframe from plant_classification.model.compile_data
    """

    test_df = compile_data.build_dataframe(INTERIM_DATA_DIR / "train")

    # test that output is of type pd.DataFrame
    assert isinstance(test_df, pd.DataFrame)
    # test that dataframe contains keys 'file_path' and 'label'
    assert set(test_df.keys())==set(['file_path','label'])
    # test that labels are 'diseased' or 'healthy'
    assert set(test_df['label'].value_counts().keys())==set(['diseased','healthy'])


def test_create_generator():
    """
    Function to test create_generator from plant_classification.model.compile_data
    """
    import keras
    image_shape, num_classes, batch_size = test_vars.get_test_vars()

    test_gen = compile_data.create_generator(INTERIM_DATA_DIR / "train", image_shape, num_classes, batch_size)

    # test that output has correct type, DataFrameIterator
    assert isinstance(test_gen, keras.src.legacy.preprocessing.image.DataFrameIterator)

    # test that function raises the correct error when bad types are passed
    with pytest.raises(TypeError) as e_info:
        test_gen = compile_data.create_generator(INTERIM_DATA_DIR / "train", 'A', num_classes, batch_size)
    with pytest.raises(TypeError) as e_info:
        test_gen = compile_data.create_generator(INTERIM_DATA_DIR / "train", image_shape, None, batch_size)
    with pytest.raises(TypeError) as e_info:
        test_gen = compile_data.create_generator(INTERIM_DATA_DIR / "train", image_shape, num_classes, 11.27)
