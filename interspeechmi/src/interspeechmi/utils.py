import os
import logging
import gzip
import shutil
from re import finditer
from tqdm import *
import requests
from interspeechmi.constants import PYTHON_SCRIPTS_DIR, LOGS_DIR
from interspeechmi.standalone_utils import full_path

logger = logging.getLogger(__name__)


def get_log_path_for_python_script(script_path, python_scripts_dir=PYTHON_SCRIPTS_DIR, log_dir=LOGS_DIR):
    script_path_relative_to_python_scripts_dir = os.path.relpath(full_path(script_path), python_scripts_dir)
    log_path = os.path.join(log_dir, "py_scripts_logs", f"{script_path_relative_to_python_scripts_dir}.log")
    log_path_parent_dir = os.path.dirname(log_path)
    if not os.path.isdir(log_path_parent_dir):
        os.makedirs(log_path_parent_dir)
    return log_path


def camel_case_split(identifier: str):
    """
    CamelCase split, e.g. camel_case_split("CamelCase") = ["camel", "case"]
    
    Credit: https://stackoverflow.com/a/29920015
    """
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def download_from_url(
    url: str,
    save_path: str,
    overwrite: bool=False,
):
    """
    Adadpted from https://stackoverflow.com/a/56796119
    """
    # url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    # name = "video"
    if os.path.isfile(save_path) and not overwrite:
        logger.info(f"No overwrite. {save_path} already exists")
        return
    
    logger.info(f"Downloading {url} to {save_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            pbar = tqdm(total=int(r.headers['Content-Length']))
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))    


def gzip_decompress(
    gzip_file_path: str,
    gzip_decompression_output_path: str=None,
    overwrite: bool=False
):
    """
    Adapted from https://stackoverflow.com/a/44712152
    """
    if gzip_file_path.endswith(".gz"):
        gzip_decompression_output_path = gzip_file_path[:-len(".gz")]

    if os.path.isfile(gzip_decompression_output_path) and not overwrite:
        logger.info(f"No overwrite. {gzip_decompression_output_path} already exists")
        return

    logger.info(f"GZIP-decompressing {gzip_file_path} into {gzip_decompression_output_path}")

    with gzip.open(gzip_file_path, 'rb') as f_in:
        with open(gzip_decompression_output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)    
