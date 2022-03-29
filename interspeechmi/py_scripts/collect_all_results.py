import os
import json
from interspeechmi.nlp.unitask_code_prediction.constants import (
    ROBERTA_BASE_CHECKPOINT, 
    ROBERTA_LARGE_CHECKPOINT
)
from interspeechmi.nlp.unitask_code_prediction.model_manager import (
    UniTaskCodePredictionManager
)
import pandas as pd
import argparse
import logging
from interspeechmi.constants import (
    DEFAULT_LOG_FORMAT, 
    DEFAULT_LOG_FILEMODE,
    VISUALS_DIR,
)
from interspeechmi.standalone_utils import (
    full_path, 
    json_load, 
    json_pretty_write
)
from interspeechmi.utils import get_log_path_for_python_script

LOG_TO_CONSOLE = True
logger = logging.getLogger(__name__)
log_path = get_log_path_for_python_script(full_path(__file__))


def get_results_for_high_quality_trained_models(
    base_model: str=None,
    conv_history_window: int=None,
    prepend_codes_in_context: str=None,
    per_device_train_batch_size: int=None,
    gradient_accumulation_steps: int=None,
    use_augmentation_during_training: bool=False,
    num_augs_to_use_per_train_example: int=None,    
):
    uni_task_code_prediction_manager = \
        UniTaskCodePredictionManager(
            mi_quality="high",
            base_model=base_model,
            conv_history_window=conv_history_window,
            code_prediction_interlocutor="therapist",
            prepend_codes_in_context=prepend_codes_in_context,
            prepend_code_in_response=False,
            hyperparameters={
                "per_device_train_batch_size": per_device_train_batch_size,
                "per_device_eval_batch_size": per_device_train_batch_size,
                "max_seq_len": 512,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_train_epochs": 10,
                "learning_rate": 2e-5,
                "warmup_ratio": 0.0,
            },
            do_domain_adaptation=False,
            use_adapters=False,
            use_augmentation_during_training=use_augmentation_during_training,
            num_augs_to_use_per_train_example=num_augs_to_use_per_train_example,
            debug=False,
            prediction_mode="next_turn_code_forecast",
            use_only_one_split_for_debug=False,
            dev_data_to_fold_size_ratio=0.5
        )
    cv_results = uni_task_code_prediction_manager.cross_validation_pipeline(
        debug=False,
        overwrite=False ,
        only_collect_results=True                  
    )    
    per_split_performances = uni_task_code_prediction_manager.get_test_set_performance_per_split()    
    return cv_results, per_split_performances


def get_results_for_mixed_quality_trained_models(
    conv_history_window: int=None,
    mi_quality_balancing_method: str=None
):
    uni_task_code_prediction_manager = \
        UniTaskCodePredictionManager(
            mi_quality=None,
            mi_quality_balancing_method=mi_quality_balancing_method,
            base_model=ROBERTA_BASE_CHECKPOINT,
            conv_history_window=conv_history_window,
            code_prediction_interlocutor="therapist",
            prepend_codes_in_context="no",
            prepend_code_in_response=False,
            hyperparameters={
                "per_device_train_batch_size": 8,
                "per_device_eval_batch_size": 8,
                "max_seq_len": 512,
                "gradient_accumulation_steps": 1,
                "num_train_epochs": 10,
                "learning_rate": 2e-5,
                "warmup_ratio": 0.0,
            },
            do_domain_adaptation=False,
            use_adapters=False,
            use_augmentation_during_training=False,
            num_augs_to_use_per_train_example=None,
            debug=False,
            prediction_mode="next_turn_code_forecast",
            use_only_one_split_for_debug=False,
            dev_data_to_fold_size_ratio=0.5
        )
    cv_results = uni_task_code_prediction_manager.cross_validation_pipeline(
        debug=False,
        overwrite=False ,
        only_collect_results=True,
        get_performance_on_high_quality_mi_examples_only=True
    )    
    per_split_performances = uni_task_code_prediction_manager.get_test_set_performance_per_split(
        get_performance_on_high_quality_mi_examples_only=True
    )    
    return cv_results, per_split_performances


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

        overall_results_path = os.path.join(
            VISUALS_DIR, "overall_results.json"
        )
        overall_results_df_path = os.path.join(
            VISUALS_DIR, "overall_results.csv"
        )        
        if args.overwrite or not os.path.isfile(overall_results_path):
            overall_results_dict = dict()
        else:
            overall_results_dict = json_load(overall_results_path)

        per_split_results_path = os.path.join(
            VISUALS_DIR, "per_split_results.json"
        )
        per_split_results_df_path = os.path.join(
            VISUALS_DIR, "per_split_results.csv"
        )  
        if args.overwrite or not os.path.isfile(per_split_results_path):
            per_split_results_dict = dict()
        else:
            per_split_results_dict = json_load(per_split_results_path)

        prepend_codes_in_context = "no"
        base_model = ROBERTA_LARGE_CHECKPOINT
        for conv_history_window in [1, 3, 5, 7, 9, 1000]:
            result_key = json.dumps({
                "base_model": base_model,
                "setup": prepend_codes_in_context,
                "window": conv_history_window,
            })
            if (
                result_key in overall_results_dict and 
                result_key in per_split_results_dict and 
                not args.overwrite
            ):
                continue
            cv_results, per_split_performances = get_results_for_high_quality_trained_models(
                base_model=base_model,
                conv_history_window=conv_history_window,
                prepend_codes_in_context=prepend_codes_in_context,
                use_augmentation_during_training=False,
                num_augs_to_use_per_train_example=None,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2
            )
            overall_results_dict[result_key] = {
                "f1": cv_results["overall_f1"],
                "mcc": cv_results["overall_mcc"]
            }
            per_split_results_dict[result_key] = {
                "f1": per_split_performances["f1"],
                "mcc": per_split_performances["mcc"]
            }               

        for prepend_codes_in_context in [
            "no", 
            "therapist_oracle", 
            "client_oracle", 
            "therapist_and_client_oracle",
            "therapist_predicted",
            "client_predicted",
            "therapist_and_client_predicted",
            "therapist_random",
            "client_random",
            "therapist_and_client_random",
        ]:
            for conv_history_window in [1, 3, 5, 7, 9, 1000]:
                base_model = ROBERTA_BASE_CHECKPOINT
                result_key = json.dumps({
                    "base_model": base_model,
                    "setup": prepend_codes_in_context,
                    "window": conv_history_window,
                })
                if (
                    result_key in overall_results_dict and 
                    result_key in per_split_results_dict and 
                    not args.overwrite
                ):
                    continue

                if (
                    conv_history_window == 1 and
                    prepend_codes_in_context in [
                        "therapist_oracle", "therapist_predicted", "therapist_random"
                    ]
                ):
                    # Basically, if only therapist codes are used, with one-utterance history,
                    # it makes no difference what kind of therapist codes are used, because
                    # the only utterance in the history is from the client
                    baseline_result_key = json.dumps({
                        "base_model": base_model,
                        "setup": "no",
                        "window": 1,
                    })
                    overall_results_dict[result_key] = overall_results_dict[baseline_result_key]
                    per_split_results_dict[result_key] = per_split_results_dict[baseline_result_key]
                else:
                    cv_results, per_split_performances = get_results_for_high_quality_trained_models(
                        base_model=base_model,
                        conv_history_window=conv_history_window,
                        prepend_codes_in_context=prepend_codes_in_context,
                        use_augmentation_during_training=False,
                        num_augs_to_use_per_train_example=None,
                        per_device_train_batch_size=8,
                        gradient_accumulation_steps=1
                    )
                    overall_results_dict[result_key] = {
                        "f1": cv_results["overall_f1"],
                        "mcc": cv_results["overall_mcc"]
                    }
                    per_split_results_dict[result_key] = {
                        "f1": per_split_performances["f1"],
                        "mcc": per_split_performances["mcc"]
                    }                

        
        prepend_codes_in_context = "no"
        num_augs_to_use_per_train_example = 5
        base_model = ROBERTA_BASE_CHECKPOINT
        for conv_history_window in [1, 3, 5, 7, 9, 1000]:
            result_key = json.dumps({
                "base_model": base_model,
                "setup": f"{num_augs_to_use_per_train_example}_aug",
                "window": conv_history_window,
            })
            if (
                result_key in overall_results_dict and 
                result_key in per_split_results_dict and 
                not args.overwrite
            ):
                continue
            cv_results, per_split_performances = get_results_for_high_quality_trained_models(
                base_model=base_model,
                conv_history_window=conv_history_window,
                prepend_codes_in_context=prepend_codes_in_context,
                use_augmentation_during_training=True,
                num_augs_to_use_per_train_example=num_augs_to_use_per_train_example,
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1                
            )
            overall_results_dict[result_key] = {
                "f1": cv_results["overall_f1"],
                "mcc": cv_results["overall_mcc"]
            }
            per_split_results_dict[result_key] = {
                "f1": per_split_performances["f1"],
                "mcc": per_split_performances["mcc"]
            }     

        base_model = ROBERTA_BASE_CHECKPOINT
        for mi_quality_balancing_method in [None, "augmentation"]:
            for conv_history_window in [1, 3, 5, 7, 9, 1000]:
                result_key = json.dumps({
                    "base_model": base_model,
                    "setup": "high_and_low_{}".format(
                        "aug_balanced" \
                            if mi_quality_balancing_method == "augmentation" \
                                else "unbalanced"
                    ),
                    "window": conv_history_window,
                })
                if (
                    result_key in overall_results_dict and 
                    result_key in per_split_results_dict and 
                    not args.overwrite
                ):
                    continue
                cv_results, per_split_performances = get_results_for_mixed_quality_trained_models(
                    conv_history_window=conv_history_window,
                    mi_quality_balancing_method=mi_quality_balancing_method
                )
                overall_results_dict[result_key] = {
                    "f1": cv_results["overall_f1"],
                    "mcc": cv_results["overall_mcc"]
                }
                per_split_results_dict[result_key] = {
                    "f1": per_split_performances["f1"],
                    "mcc": per_split_performances["mcc"]
                }                   

        json_pretty_write(overall_results_dict, overall_results_path)
        json_pretty_write(per_split_results_dict, per_split_results_path)

        overall_results_df = {
            "base_model": [],
            "setup": [],
            "window": [],
            "f1": [],
            "mcc": []
        }
        for full_setup_str, scores in overall_results_dict.items():
            full_setup = json.loads(full_setup_str)
            for key, value in full_setup.items():
                overall_results_df[key].append(value)
            for score_type in ["f1", "mcc"]:
                overall_results_df[score_type].append(scores[score_type])
        overall_results_df = pd.DataFrame.from_dict(overall_results_df)
        overall_results_df.to_csv(overall_results_df_path, index=False)

        per_split_results_df = {
            "base_model": [],
            "setup": [],
            "window": [],
            "split": [],
            "f1": [],
            "mcc": []
        }
        for full_setup_str, scores in per_split_results_dict.items():
            full_setup = json.loads(full_setup_str)
            for split_id in range(5):
                for key, val in full_setup.items():
                    per_split_results_df[key].append(val)
                per_split_results_df["split"].append(split_id)
                for score_type in ["f1", "mcc"]:
                    per_split_results_df[score_type].append(
                        scores[score_type][f"split_{split_id}"]
                    )
        per_split_results_df = pd.DataFrame.from_dict(per_split_results_df)
        per_split_results_df.to_csv(per_split_results_df_path, index=False)

    except Exception as e:
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()        

