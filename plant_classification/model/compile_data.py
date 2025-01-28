from pathlib import Path
import glob

import pandas as pd

def build_dataframe(input_path):
    """
    A function to produce a dictionary of file path / label pairs.

    Args:
        Path input_path: Path to the data to be collated

    Returns:
        pd.DataFrame data_df: DataFrame containing the columns "file_path" and "label"
    """
    data_dict = {'file_path': [], 'label': []}

    for folder in list(glob.glob(str(input_path)+'/*')):
        if folder.startswith('.'): # ignore hidden files
            continue
        disease_label = folder.split(' ')[-2]

        for file in list(glob.glob(str(folder)+'/*')):
            data_dict['file_path'].append(file)
            data_dict['label'].append(disease_label)

    data_df = pd.DataFrame(data_dict)
    return data_df


def create_generator(input_path, image_shape, num_classes, batch_size):
    """
    A function to generate additional data.

    Args:
        pd.DataFrame data_df: DataFrame containing data path and label
        array image_shape: Array containing image shape
        int num_classes: Integer of number of classes to classify
        int batch_size: Integer batch size

    Returns:
        Data generator for data_df
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if not isinstance(image_shape, tuple):
        raise TypeError("image_shape must be of type tuple")
    if not isinstance(num_classes, int):
        raise TypeError("num_classes must be of type int")
    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be of type int")

    # build dataframe for given data source
    data_df = build_dataframe(input_path)

    data_type = data_df['file_path'].iloc[0].split('/')[-3]
    if data_type == 'train':
        # expand our training dataset with rotation, zoom, and flip
        datagen = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=20,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        vertical_flip=True
                        )
    else:
        # rescale the test and validation data colour values
        datagen = ImageDataGenerator(rescale=1./255)


    generator = datagen.flow_from_dataframe(
                        dataframe=data_df,
                        directory=None,
                        x_col='file_path',
                        y_col='label',
                        target_size=image_shape[:2],
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=True
                        )

    return generator
