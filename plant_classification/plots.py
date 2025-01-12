from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from student_anxiety.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR,
    output_path: Path = FIGURES_DIR,
    # -----------------------------------------
):


    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
