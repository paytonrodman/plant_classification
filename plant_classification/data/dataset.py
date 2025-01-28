from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import glob

from plant_classification.config import RAW_DATA_DIR, INTERIM_DATA_DIR
from plant_classification.data import reduce_data

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR,
):
    logger.info("Reducing data...")
    file_list = reduce_data.check_file_exists(input_path, output_path, overwrite_flag=0)
    reduce_data.reduce_image_size(file_list, output_path, target_dim=256)
    logger.success("Reducing data complete.")


if __name__ == "__main__":
    app()
