import argparse
import logging
import transformers
from interspeechmi.constants import (
    DEFAULT_LOG_FORMAT, 
    DEFAULT_LOG_FILEMODE,
)
from interspeechmi.data_handling.preprocess_annomi import (
    preprocess_anno_mi, 
)
from interspeechmi.standalone_utils import full_path
from interspeechmi.utils import get_log_path_for_python_script

LOG_TO_CONSOLE = True
logger = logging.getLogger(__name__)
log_path = get_log_path_for_python_script(full_path(__file__))


def main():

    parser=argparse.ArgumentParser()
    parser.add_argument(
        "-o", '--overwrite', 
        help='Overwrite all data and results', 
        action="store_true"
    )
    args=parser.parse_args()

    try:
        if LOG_TO_CONSOLE:
            logging.basicConfig(format=DEFAULT_LOG_FORMAT,
                                level=logging.DEBUG)  
        else:
            logging.basicConfig(filename=log_path,
                                filemode=DEFAULT_LOG_FILEMODE,
                                format=DEFAULT_LOG_FORMAT,
                                level=logging.DEBUG)

        transformers.logging.enable_propagation()
        transformers.logging.disable_default_handler()   

        preprocess_anno_mi(overwrite=args.overwrite)
                                
    except Exception as e:
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()        
