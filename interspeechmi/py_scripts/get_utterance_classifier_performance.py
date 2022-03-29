import os
import pandas as pd
import argparse
import logging
from interspeechmi.constants import (
    DEFAULT_LOG_FORMAT, 
    DEFAULT_LOG_FILEMODE,
    MODELS_DIR,
)
from interspeechmi.standalone_utils import full_path
from interspeechmi.utils import get_log_path_for_python_script
from sklearn.metrics import f1_score

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

        
        therapist_utt_classifier_results_path = os.path.join(
            MODELS_DIR,
            "unitask_code_prediction",
            "current_turn_code_prediction",
            "anno_mi.high_quality.responses.no_code_in_response.clean_separation_of_dialogues.response_from_therapist_only.5_fold_splits",
            "roberta-base.joint_model.max_seq_len_512.seed_42.bs_8.lr_2e-05.warmup_ratio_0.0.weight_decay_0.01.grad_acc_1.epochs_10",
            "cross_validation_test_results.csv"
        )
        therapist_utt_classifier_results = pd.read_csv(therapist_utt_classifier_results_path)
        therapist_utt_classification_f1 = f1_score(
            y_true=therapist_utt_classifier_results["ground_truth_code"], 
            y_pred=therapist_utt_classifier_results["predicted_code"], 
            average="macro"
        )

        client_utt_classifier_results_path = os.path.join(
            MODELS_DIR,
            "unitask_code_prediction",
            "current_turn_code_prediction",
            "anno_mi.high_quality.responses.no_code_in_response.clean_separation_of_dialogues.response_from_client_only.5_fold_splits",
            "roberta-base.joint_model.max_seq_len_512.seed_42.bs_8.lr_2e-05.warmup_ratio_0.0.weight_decay_0.01.grad_acc_1.epochs_10",
            "cross_validation_test_results.csv"
        )
        client_utt_classifier_results = pd.read_csv(client_utt_classifier_results_path)
        client_utt_classification_f1 = f1_score(
            y_true=client_utt_classifier_results["ground_truth_code"], 
            y_pred=client_utt_classifier_results["predicted_code"], 
            average="macro"
        )

        logger.debug(
            f"""
            Therapist utternace classification F1: {therapist_utt_classification_f1}
            Client utternace classification F1: {client_utt_classification_f1}
            """
        )

    except Exception as e:
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()        


