import numpy as np
import matplotlib.pyplot as plt

import keras

def plot_pred_individual(model, image_path, ax):
    """
    A function to generate a tile plot of all 'predict' images and their
    predicted class under the supplied model

    Args:
        keras.model model: CNN model to use for prediction
        image_path: Location of images to predict
        axesObject ax: Axis within tile to plot to

    Returns:
        None
    """
    print(image_path)
    img = keras.utils.load_img(str(image_path), target_size=(256,256))
    ax.imshow(img)
    ax.grid(False)

    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(keras.ops.sigmoid(predictions[0][0]))
    ax.annotate(f"{100 * (1 - score):.2f}\% healthy, {100 * score:.2f}\% diseased", (0,0), (0, -20))

def make_image(model, files_to_predict, output_path):
    """
    A function to generate a tile plot containing 2x4 subplots

    Args:
        keras.model model: CNN model to use for prediction
        list files_to_predict: List of image files to predict, including path
        Path output_path: Location to save plot

    Returns:
        None
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    ax_list = list(axes.flat)[::-1]
    num_subplots = len(ax_list)
    for i in range(num_subplots):
        ax = ax_list.pop()
        plot_pred_individual(model, files_to_predict[i], ax)

    plt.savefig(output_path / 'predictions.png', dpi=400, bbox_inches='tight')
