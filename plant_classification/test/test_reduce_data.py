# test_make_dataset.py

import pytest

import sys
sys.path.append("../..") # Adds higher directory to python modules path.

from plant_classification.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from plant_classification.data import reduce_data

def test_check_file_exists():
    """
    Function to test check_file_exists from plant_classification.reduce_data
    """

    # test that function returns non-empty list when overwrite=1
    assert len(reduce_data.check_file_exists(RAW_DATA_DIR, PROCESSED_DATA_DIR, overwrite_flag=1))>0


def test_reduce_image_size():
    """
    Function to test check_file_exists from plant_classification.reduce_data
    """

    # test that function returns None if input has length 0
    assert reduce_data.reduce_image_size([], RAW_DATA_DIR, 256) == None
    # test that function returns None if target_dim is non-integer-castable
    for test_target_dim in [False, 'hello', {'dict_key': [1]}, ['test'], [1], None]:
        with pytest.raises(TypeError) as excinfo:
            reduce_data.reduce_image_size(['test'], RAW_DATA_DIR, test_target_dim)
        assert str(excinfo.value) == "target_dim must be integer-castable."
