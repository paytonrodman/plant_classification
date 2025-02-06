from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import scipy as sc
import pandas as pd
import os
import keras

from plant_classification.config import INTERIM_DATA_DIR, MODELS_DIR
from plant_classification.model import compile_data, create_model

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = typer.Typer()


@app.command()
def main(
    model_type: str = 'simple',
    input_path: Path = INTERIM_DATA_DIR,
    model_path: Path = MODELS_DIR,
):

    image_shape = (256, 256, 3) # size of images
    num_classes = 2 # disease status
    batch_size = 10 # must be divisor of no. test images (110)
    if model_type == 'simple':
        epochs = 15
    else:
        epochs = 10

    logger.info("Generating data...")
    train_data = compile_data.create_generator(INTERIM_DATA_DIR / 'train', image_shape, num_classes, batch_size)
    valid_data = compile_data.create_generator(INTERIM_DATA_DIR / 'valid', image_shape, num_classes, batch_size)
    logger.success("Data generation complete.")

    logger.info(f"Training {model_type} model for {epochs} epochs...")
    # create model
    model = create_model.create_model(model_type, image_shape, num_classes, print_summary=1)

    # compile model
    if model_type == 'simple':
        lr = 0.001
    else:
        lr = 0.001
    create_model.compile_model(model,
                               opt=keras.optimizers.Adam(learning_rate=lr),
                               loss_func=keras.losses.CategoricalCrossentropy(),
                               metrics=['accuracy'])

    # train model
    history = model.fit(train_data,
                        epochs=epochs,
                        validation_data=valid_data)


    score = model.evaluate(valid_data, batch_size=batch_size)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Save model and model history
    save_path = model_path / model_type
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path / 'model.keras')
    hist_df = pd.DataFrame(history.history)
    with open(save_path / 'modelhistory.csv', mode='w') as f:
        hist_df.to_csv(f)
    logger.success("Network training complete.")



if __name__ == "__main__":
    app()
