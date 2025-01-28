from pathlib import Path
from plant_classification.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

from tqdm import tqdm # process bar
from loguru import logger

def check_file_exists(input_path, output_path, overwrite_flag=0):
    """
    A function to determine if processed files already exist, or whether
    they should be overwritten.

    Args:
        Path RAW_DATA_DIR: Path object to raw data directory.
        Path PROCESSED_DATA_DIR: Path object to processed data directory.
        bool overwrite_flag: A flag indicating data should be overwritten.

    Returns:
        list filelist: A list of files to be processed
    """
    import glob

    raw_files = []
    for file in list(glob.glob(str(input_path)+'/*/*/*')):
        if file.startswith('.'): # ignore hidden files
            continue
        if file.endswith('.JPG'):
            raw_fn = file.split('/')[-1]
            raw_files.append(file)

    if overwrite_flag==1:
        logger.info(f'Overwrite selected. Reducing all images.')
        return raw_files
    else:
        logger.info(f'Only reducing images not already present.')
        proc_files = []
        for file in list(glob.glob(str(output_path)+'/*/*/*')):
            if file.startswith('.'): # ignore hidden files
                continue
            if file.endswith('.JPG'):
                proc_fn = file.split('/')[-1]
                proc_files.append(proc_fn)

        return [file for file in raw_files if file.split('/')[-1] not in proc_files] # files in RAW that are not in PROCESSED


def reduce_image_size(file_list, output_path, target_dim):
    """
    A function to reduce the size of image files

    Args:
        list file_list: A list of files (including path) to be reduced.
        int target_dim: The target dimensions of the new images, in pixels

    Returns:
        None
    """
    import os
    from PIL import Image
    import numpy as np

    if len(file_list)==0:
        logger.info(f'No files to reduce.')
        return None

    if isinstance(target_dim, (int, float, complex)) and not isinstance(target_dim, bool):
        target_dim = int(target_dim)
    else:
        raise TypeError("target_dim must be integer-castable.")

    for imfile in tqdm(file_list):
        image = Image.open(imfile)

        # rescale image so the shortest side is 256pix, maintaining aspect ratio
        conv_fact = np.min(np.shape(image)[:2])/target_dim
        scale_dim = [np.shape(image)[0]/conv_fact, np.shape(image)[1]/conv_fact]
        image.thumbnail((np.max(scale_dim),np.max(scale_dim)), Image.LANCZOS)

        # make image square by cropping longer side (width) to central 256
        w_start = (np.shape(image)[1]-target_dim)/2
        w_end = np.shape(image)[1]-w_start
        (left, upper, right, lower) = (w_start, 0, w_end, target_dim)
        image = image.crop((left, upper, right, lower))

        # convert to RGB in case it's RGBA
        image = image.convert("RGB")

        # create any necessary subdirectories and save data
        raw_id = imfile.split('/').index('raw')
        rel_path = '/'.join(imfile.split('/')[raw_id+1:-1])
        save_path = output_path / rel_path
        os.makedirs(save_path, exist_ok=True)
        fname = imfile.split('/')[-1]
        image.save(save_path / fname, image.format)
