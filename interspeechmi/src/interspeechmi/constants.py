import os
import logging
from interspeechmi.standalone_utils import full_path
from pathlib import Path

REPO_DIR = Path(os.path.dirname(full_path(__file__))).parent.parent.parent
WANDB_LOGS_DIR_PARENT_DIR = str(REPO_DIR)
INTERSPEECHMI_CODE_DIR = os.path.join(REPO_DIR, "interspeechmi")
TMP_FILES_DIR = os.path.join(REPO_DIR, "tmp")
SRC_DIR = os.path.join(INTERSPEECHMI_CODE_DIR, "src")
PYTHON_SCRIPTS_DIR = os.path.join(INTERSPEECHMI_CODE_DIR, "py_scripts")
LOGS_DIR = os.path.join(INTERSPEECHMI_CODE_DIR, "logs")
TB_LOGS_DIR = os.path.join(INTERSPEECHMI_CODE_DIR, "tb_logs")
DATA_DIR = os.path.join(INTERSPEECHMI_CODE_DIR, "data")
MODELS_DIR = os.path.join(INTERSPEECHMI_CODE_DIR, "models")
CACHE_DIR = os.path.join(INTERSPEECHMI_CODE_DIR, "cache")
VISUALS_DIR = os.path.join(INTERSPEECHMI_CODE_DIR, "visuals")

for dir in [
    TMP_FILES_DIR, PYTHON_SCRIPTS_DIR, LOGS_DIR, 
    TB_LOGS_DIR, DATA_DIR, MODELS_DIR, 
    CACHE_DIR, VISUALS_DIR
]:
    if not os.path.isdir(dir):
        os.mkdir(dir)

DEFAULT_LOG_FORMAT = "    [%(asctime)s - %(filename)30s:%(lineno)4s - %(levelname)5s - %(funcName)30s()]\n%(message)s\n"
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FILEMODE = 'w'

