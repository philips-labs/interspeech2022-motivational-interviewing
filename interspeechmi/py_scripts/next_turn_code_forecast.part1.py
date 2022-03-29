from interspeechmi.nlp.unitask_code_prediction.constants import (
    BERT_BASE_CHEKCPOINT, 
    ROBERTA_BASE_CHECKPOINT, 
    ROBERTA_LARGE_CHECKPOINT
)
from interspeechmi.nlp.unitask_code_prediction.model_manager import (
    UniTaskCodePredictionManager
)
import pandas as pd
import argparse
import logging
import transformers
from interspeechmi.constants import (
    DEFAULT_LOG_FORMAT, 
    DEFAULT_LOG_FILEMODE,
)
from interspeechmi.standalone_utils import full_path
from interspeechmi.utils import get_log_path_for_python_script

LOG_TO_CONSOLE = True
logger = logging.getLogger(__name__)
log_path = get_log_path_for_python_script(full_path(__file__))


def main():

    parser=argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--overwrite",
        help="Overwrite all data and results", 
        action="store_true"
    )
    parser.add_argument(
        "-d", "--debug",
        help="Use debug mode", 
        action="store_true"
    )
    parser.add_argument(
        "-f", "--file",
        help="Log to file", 
        action="store_true"
    )
    args=parser.parse_args()

    try:
        if args.file:
            logging.basicConfig(filename=log_path,
                                filemode=DEFAULT_LOG_FILEMODE,
                                format=DEFAULT_LOG_FORMAT,
                                level=logging.DEBUG)
        else:
            logging.basicConfig(format=DEFAULT_LOG_FORMAT,
                                level=logging.DEBUG)  

        transformers.logging.enable_propagation()
        transformers.logging.disable_default_handler()   

        next_turn_code_forecast_base_config = {
            "dialogue_mi_quality_filter": "high",
            "prediction_mode": "next_turn_code_forecast",
            "code_prediction_interlocutor": "therapist",
            "prepend_codes_in_context": "no",
            "prepend_code_in_response": False,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.0,
            "max_seq_len": 512,
            "num_train_epochs": 10,
            "do_domain_adaptation": False,
            "use_adapters": False,
            "use_augmentation_during_training": False,
            "num_augs_to_use_per_train_example": None,
            "dev_data_to_fold_size_ratio": 0.5
        }

        next_turn_code_forecast_configs = []
        for window in [1, 3, 5, 7, 9, 1000]:
            config = {
                "conv_history_window": window,
                "base_model": ROBERTA_BASE_CHECKPOINT,
                "train_batch_size": 8,
                "grad_acc_steps": 1,
                **next_turn_code_forecast_base_config
            }
            next_turn_code_forecast_configs.append(config)
            config = {
                "conv_history_window": window,
                "base_model": ROBERTA_LARGE_CHECKPOINT,
                "train_batch_size": 4,
                "grad_acc_steps": 2,
                **next_turn_code_forecast_base_config
            }
            next_turn_code_forecast_configs.append(config)            

        configs_to_run = next_turn_code_forecast_configs
        for config in configs_to_run:
            uni_task_code_prediction_manager = \
                UniTaskCodePredictionManager(
                    mi_quality=config["dialogue_mi_quality_filter"],
                    base_model=config["base_model"],
                    conv_history_window=config["conv_history_window"],
                    code_prediction_interlocutor=config["code_prediction_interlocutor"],
                    prepend_codes_in_context=config["prepend_codes_in_context"],
                    prepend_code_in_response=config["prepend_code_in_response"],
                    hyperparameters={
                        "per_device_train_batch_size": config["train_batch_size"],
                        "per_device_eval_batch_size": config["train_batch_size"],
                        "max_seq_len": config["max_seq_len"],
                        "gradient_accumulation_steps": config["grad_acc_steps"],
                        "num_train_epochs": config["num_train_epochs"],
                        "learning_rate": config["learning_rate"],
                        "warmup_ratio": config["warmup_ratio"],
                    },
                    do_domain_adaptation=config["do_domain_adaptation"],
                    use_augmentation_during_training=config["use_augmentation_during_training"],
                    num_augs_to_use_per_train_example=config["num_augs_to_use_per_train_example"],
                    use_adapters=config["use_adapters"],
                    debug=args.debug,
                    prediction_mode=config["prediction_mode"],
                    use_only_one_split_for_debug=False,
                    # use_only_one_split_for_debug=True,
                    dev_data_to_fold_size_ratio=config["dev_data_to_fold_size_ratio"]
                )
            cv_results = uni_task_code_prediction_manager.cross_validation_pipeline(
                debug=args.debug,
                also_get_label_prediction_order=True,
                overwrite=False                    
            )


    except Exception as e:
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()        

