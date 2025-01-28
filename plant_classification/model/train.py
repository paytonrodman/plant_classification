from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import scipy as sc
import pandas as pd

from plant_classification.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from plant_classification.model import compile_data, create_model

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR,
    model_path: Path = MODELS_DIR / "model.keras",
    history_path: Path = MODELS_DIR / "modelhistory.csv",
):

    image_shape = (256, 256, 3) # size of images
    num_classes = 2 # disease status
    batch_size = 10 # must be divisor of no. test images (110)
    epochs = 15

    logger.info("Generating data...")
    train_gen = compile_data.create_generator(INTERIM_DATA_DIR / "train", image_shape, num_classes, batch_size)
    test_gen  = compile_data.create_generator(INTERIM_DATA_DIR / "test", image_shape, num_classes, batch_size)
    valid_gen = compile_data.create_generator(INTERIM_DATA_DIR / "valid", image_shape, num_classes, batch_size)
    logger.success("Data generation complete.")

    logger.info(f"Training network for {epochs} epochs...")

    # create model
    model = create_model.create_simple_model(image_shape, num_classes)

    # compile model
    create_model.compile_model(model)

    # train model
    history = model.fit(train_gen,
                        epochs=epochs,
                        validation_data=valid_gen)

    score = model.evaluate(valid_gen, batch_size=batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save(model_path)
    hist_df = pd.DataFrame(history.history)
    with open(history_path, mode='w') as f:
        hist_df.to_csv(f)
    logger.success("Network training complete.")



if __name__ == "__main__":
    app()
