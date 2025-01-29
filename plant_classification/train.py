from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import scipy as sc
import pandas as pd
import os

from plant_classification.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from plant_classification.model import compile_data, create_model

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR,
    model_path: Path = MODELS_DIR,
):

    image_shape = (256, 256, 3) # size of images
    num_classes = 2 # disease status
    batch_size = 10 # must be divisor of no. test images (110)
    epochs = 15
    model_type = 'simple'

    logger.info("Generating data...")
    train_gen = compile_data.create_generator(INTERIM_DATA_DIR / "train", image_shape, num_classes, batch_size)
    valid_gen = compile_data.create_generator(INTERIM_DATA_DIR / "valid", image_shape, num_classes, batch_size)
    logger.success("Data generation complete.")

    logger.info(f"Training {model_type} model for {epochs} epochs...")
    # create model
    model = create_model.create_model(model_type, image_shape, num_classes, print_summary=0)

    # compile model
    create_model.compile_model(model)

    # train model
    history = model.fit(train_gen,
                        epochs=epochs,
                        validation_data=valid_gen)

    score = model.evaluate(valid_gen, batch_size=batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save model and model history
    os.makedirs(model_path / model_type, exist_ok=True)
    model.save(model_path / model_type / 'model.keras')
    hist_df = pd.DataFrame(history.history)
    with open(model_path / model_type / 'modelhistory.csv', mode='w') as f:
        hist_df.to_csv(f)
    logger.success("Network training complete.")



if __name__ == "__main__":
    app()
