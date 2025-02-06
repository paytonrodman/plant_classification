from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from plant_classification.config import RAW_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR, FIGURES_DIR
from plant_classification.predict import make_pred

import keras
import glob
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


app = typer.Typer()


@app.command()
def main(
    model_type: str = 'simple',
    input_path: Path = INTERIM_DATA_DIR,
    model_path: Path = MODELS_DIR,
    figure_path: Path = FIGURES_DIR,
):
    # reload model
    model = keras.models.load_model(model_path / model_type / 'model.keras')

    # make list of prediction images
    files_to_predict = list(glob.glob(str(RAW_DATA_DIR / 'predict')+'/*'))

    save_path = figure_path / model_type
    os.makedirs(save_path, exist_ok=True)

    # create plot
    make_pred.make_image(model, files_to_predict, save_path)



if __name__ == "__main__":
    app()
