from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from plant_classification.config import INTERIM_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR
from plant_classification.model import compile_data
from plant_classification.evaluate import make_figures, make_statistics

import keras
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = typer.Typer()


@app.command()
def main(
    model_type: str = 'pretrained',
    model_path: Path = MODELS_DIR,
    figure_path: Path = FIGURES_DIR,
):

    # reload model
    model = keras.models.load_model(model_path / model_type / 'model.keras')
    history = pd.read_csv(model_path / model_type / 'modelhistory.csv')
    history.rename(columns={"Unnamed: 0": "epoch"}, inplace=True)

    image_shape = (256, 256, 3) # size of images
    num_classes = 2 # disease status
    batch_size = 10 # must be divisor of no. test images (110)

    test_gen  = compile_data.create_generator(INTERIM_DATA_DIR / "test", image_shape, num_classes, batch_size)
    valid_gen = compile_data.create_generator(INTERIM_DATA_DIR / "valid", image_shape, num_classes, batch_size)

    logger.info("Evaluating model...")
    score = make_statistics.get_basic_stats(model, valid_gen, batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1], '\n')

    report = make_statistics.get_report(model, test_gen)
    print(report, '\n')

    logger.info("Creating plots...")
    save_path = figure_path / model_type
    os.makedirs(save_path, exist_ok=True)
    make_figures.plot_time(history, save_path)
    make_figures.plot_confusion(model, test_gen, save_path)



if __name__ == "__main__":
    app()
