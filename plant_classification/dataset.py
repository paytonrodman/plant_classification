from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import glob

from plant_classification.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
    # ----------------------------------------------
):

    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
