import logging
import os
import shutil
import json
from typing import Any, Dict, List
from datasets.load import load_metric
from interspeechmi.constants import MODELS_DIR
from datasets import Dataset, DatasetDict
from interspeechmi.data_handling.constants import (
    ANNO_MI_NORMALIZED_AUGMENTED_PATH, 
    ANNO_MI_NORMALIZED_PATH
)
from interspeechmi.nlp.constants import(
    HF_TOKENIZERS_DIR, 
    TRANSFORMERS_CACHE_DIR
)
from interspeechmi.nlp.unitask_code_prediction.constants import (
    BERT_BASE_CHEKCPOINT,
    DEBERTA_CHECKPOINT, 
    DEBERTA_MAX_SEQ_LEN,
    ROBERTA_BASE_CHECKPOINT,
    ROBERTA_LARGE_CHECKPOINT
)
from interspeechmi.nlp.unitask_code_prediction.utils import (
    tokenize_sequence_with_left_truncation,
    tokenize_sequence_with_left_truncation_and_mi_quality_info
)
from interspeechmi.nlp.utils import (
    MyTrainerCallback, 
    load_context_response_pair_datasets, 
    load_context_response_pair_datasets_with_augmented_train_data,
    load_mi_quality_balanced_context_response_pair_datasets, 
    pad_to_max_seq_len
)
import pandas as pd
import torch
import numpy as np
import re
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as mcc
from collections import Counter
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer, 
    TrainingArguments,
    PreTrainedModel,
    set_seed
)

logger = logging.getLogger(__name__)


class TokenizerManagerForUniTaskCodePredictionModels():

    def __init__(
        self,
        base_model: str,
        max_seq_len: int=512,
        label2id: Dict[str, int]=None,
    ) -> None:
        self.base_model = base_model
        self.max_seq_len = max_seq_len
        self.label2id = label2id

        logger.info(f"Loading the tokenizer of model {self.base_model}")
        tokenizer_dir = os.path.join(
            HF_TOKENIZERS_DIR,
            self.base_model.replace('/', '.')
        )
        if os.path.isdir(tokenizer_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model, 
                use_fast=True,
                cache_dir=TRANSFORMERS_CACHE_DIR
            )
            self.tokenizer.save_pretrained(tokenizer_dir)
            if "gpt" in self.base_model or "GPT" in self.base_model:
                self.tokenizer.pad_token = self.tokenizer.eos_token            


    def tokenize_response_batch_with_right_truncation(
        self, 
        examples: Dict[str, List[Any]]
    ):
        """
        Just a wrapper around self.tokenizer.__call__(), except
        that it also handles label2id
        """
        results = self.tokenizer(
            examples["response"],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="np",
            return_attention_mask=True,
            return_token_type_ids=True
        )
        results["label"] = [
            self.label2id[response_code] for response_code \
                in examples["response_code"]
        ]
        return results


    def tokenize_context_batch_with_left_truncation(
        self, 
        examples: Dict[str, List[Any]]
    ):
        """
        Tokenize `examples` with self.tokenizer
        """
        tokenized_examples = [
            tokenize_sequence_with_left_truncation(
                sequence=text, 
                textual_label=response_code, 
                tokenizer=self.tokenizer, 
                max_seq_len=self.max_seq_len,
                label2id=self.label2id
            ) for text, response_code in zip(
                examples["context"],
                examples["response_code"]
            )
        ]

        results = dict()
        for tokenization_result_id, padding_value in zip(
            ["input_ids", "token_type_ids", "attention_mask"],
            [self.tokenizer.pad_token_id, 0, 0]
        ):
            results[tokenization_result_id] = pad_to_max_seq_len(
                int_seqs = [
                    tokenized_example[tokenization_result_id] for \
                        tokenized_example in tokenized_examples
                ],
                padding_value=padding_value,
                max_seq_len=self.max_seq_len,
                padding_side="right",
                return_tensors="np"
            )
        
        for tokenization_result_id in [
            "unprocessed_input_ids", "label"
        ]:
            results[tokenization_result_id] = [
                example[tokenization_result_id] for \
                    example in tokenized_examples
            ]
        
        return results        


    def tokenize_context_batch_with_left_truncation_and_mi_quality_info(
        self, 
        examples: Dict[str, List[Any]]
    ):
        """
        Tokenize `examples` with self.tokenizer
        """
        tokenized_examples = [
            tokenize_sequence_with_left_truncation_and_mi_quality_info(
                sequence=text, 
                textual_label=response_code, 
                tokenizer=self.tokenizer, 
                max_seq_len=self.max_seq_len,
                label2id=self.label2id,
                mi_quality=mi_quality
            ) for text, response_code, mi_quality in zip(
                examples["context"],
                examples["response_code"],
                examples["mi_quality"],
            )
        ]

        results = dict()
        for tokenization_result_id, padding_value in zip(
            ["input_ids", "token_type_ids", "attention_mask"],
            [self.tokenizer.pad_token_id, 0, 0]
        ):
            results[tokenization_result_id] = pad_to_max_seq_len(
                int_seqs = [
                    tokenized_example[tokenization_result_id] for \
                        tokenized_example in tokenized_examples
                ],
                padding_value=padding_value,
                max_seq_len=self.max_seq_len,
                padding_side="right",
                return_tensors="np"
            )
        
        for tokenization_result_id in [
            "unprocessed_input_ids", "label"
        ]:
            results[tokenization_result_id] = [
                example[tokenization_result_id] for \
                    example in tokenized_examples
            ]
        
        return results   


class UniTaskCodePredictionManager():

    def __init__(
        self,
        mi_quality: str=None, # "high" or "low"
        mi_quality_balancing_method: str=None, # "augmentation" or None
        base_model: str=None,
        conv_history_window: int=3,
        num_fold: int=5,
        code_prediction_interlocutor: str=None, 
        prepend_codes_in_context: str=None, # "no", "therapist_oracle", "therapist_predicted"
        prepend_code_in_response: bool=False,
        hyperparameters: Dict[str, Any]=None,
        do_domain_adaptation: bool=False,
        use_augmentation_during_training: bool=False,
        num_augs_to_use_per_train_example: int=None,
        use_adapters: bool=False,
        device: str="cuda",
        # if True, we simply test the prediction of current-turn
        # response code, instead of trying to forecast next-turn
        # response code
        debug: bool=False,
        prediction_mode: str="next_turn_code_forecast",
        use_only_one_split_for_debug: bool=False,
        dev_data_to_fold_size_ratio: float=1.0
    ) -> None:
        """
        Initialize some parameters, create some folders,
        and tokenize the datasets we're gonna use.
        """
        self.conv_history_window = conv_history_window
        self.num_fold = num_fold
        self.debug = debug
        self.device = device
        self.use_adapters = use_adapters
        self.use_only_one_split_for_debug = use_only_one_split_for_debug
        self.mi_quality = mi_quality
        self.do_domain_adaptation = do_domain_adaptation
        self.base_model = base_model
        self.use_augmentation_during_training = use_augmentation_during_training
        self.num_augs_to_use_per_train_example = num_augs_to_use_per_train_example
        self.dev_data_to_fold_size_ratio = dev_data_to_fold_size_ratio
        self.mi_quality_balancing_method = mi_quality_balancing_method

        if (
            mi_quality in ["high", "low"] and 
            mi_quality_balancing_method is not None
        ):
            logger.warning(
                f"""
                We're only dealing with {mi_quality}-quality MI data,
                so your value of balance_mi_quality, i.e. {mi_quality_balancing_method},
                will be ignored
                """
            )

        assert prediction_mode in ["current_turn_code_prediction", "next_turn_code_forecast"]
        self.prediction_mode = prediction_mode

        self.hyperparameters = self.get_default_hyperparameters()
        if hyperparameters is not None:
            for hp_name, hp_value in hyperparameters.items():
                self.hyperparameters[hp_name] = hp_value
        self.max_seq_len = self.hyperparameters["max_seq_len"]
        set_seed(self.hyperparameters["random_seed"])

        assert isinstance(prepend_codes_in_context, str)
        if self.use_augmentation_during_training:
            assert isinstance(num_augs_to_use_per_train_example, int)
            self.split_dataset_dicts, self.context_response_pair_data_name = \
                load_context_response_pair_datasets_with_augmented_train_data(
                    anno_mi_data_path=ANNO_MI_NORMALIZED_PATH,
                    anno_mi_augmented_data_path=ANNO_MI_NORMALIZED_AUGMENTED_PATH,
                    mi_quality=self.mi_quality,
                    conv_history_window=conv_history_window,
                    num_fold=num_fold,
                    random_seed=self.hyperparameters["random_seed"],
                    return_data_name=True,
                    response_interlocutor=code_prediction_interlocutor,
                    prepend_codes_in_context=prepend_codes_in_context,
                    prepend_code_in_response=prepend_code_in_response,
                    num_augs_to_use_per_train_example=self.num_augs_to_use_per_train_example,            
                    debug=self.debug,
                    dev_data_to_fold_size_ratio=self.dev_data_to_fold_size_ratio,
                )
        else:
            if self.mi_quality is None and self.mi_quality_balancing_method is not None:
                self.split_dataset_dicts, self.context_response_pair_data_name = \
                    load_mi_quality_balanced_context_response_pair_datasets(
                        anno_mi_data_path=ANNO_MI_NORMALIZED_PATH,
                        mi_quality=self.mi_quality,
                        conv_history_window=conv_history_window,
                        num_fold=num_fold,
                        random_seed=self.hyperparameters["random_seed"],
                        return_data_name=True,
                        response_interlocutor=code_prediction_interlocutor,
                        prepend_codes_in_context=prepend_codes_in_context,
                        prepend_code_in_response=prepend_code_in_response,                
                        debug=self.debug,
                        dev_data_to_fold_size_ratio=self.dev_data_to_fold_size_ratio,
                        mi_quality_balancing_method=self.mi_quality_balancing_method
                    )
            else:
                self.split_dataset_dicts, self.context_response_pair_data_name = \
                    load_context_response_pair_datasets(
                        anno_mi_data_path=ANNO_MI_NORMALIZED_PATH,
                        mi_quality=self.mi_quality,
                        conv_history_window=conv_history_window,
                        num_fold=num_fold,
                        random_seed=self.hyperparameters["random_seed"],
                        return_data_name=True,
                        response_interlocutor=code_prediction_interlocutor,
                        prepend_codes_in_context=prepend_codes_in_context,
                        prepend_code_in_response=prepend_code_in_response,                
                        debug=self.debug,
                        dev_data_to_fold_size_ratio=self.dev_data_to_fold_size_ratio
                    )

        ########################################################################

        # If we're only predicting the code of the current turn (response),
        # Then it doesn't matter how we configure the context, so we just
        # remove all that info
        if self.prediction_mode == "current_turn_code_prediction":
            self.context_response_pair_data_name = \
                self.context_response_pair_data_name.replace(
                    "context_response_pairs", "responses"
                )
            self.context_response_pair_data_name = re.sub(
                ".[no_]*codes_in_context", "", 
                self.context_response_pair_data_name
            )
            self.context_response_pair_data_name = re.sub(
                ".context_window_[0-9]+", "", 
                self.context_response_pair_data_name
            )
        # If we're forecasting the node of the next turn,
        # Then it doesn't matter how we configure the response, so we just
        # remove all that info            
        else:
            self.context_response_pair_data_name = \
                self.context_response_pair_data_name.replace(
                    "context_response_pairs", "contexts"
                )
            self.context_response_pair_data_name = re.sub(
                ".[no_]*code_in_response", "", 
                self.context_response_pair_data_name
            )

        unitask_models_dir = os.path.join(
            MODELS_DIR, "unitask_code_prediction"
        )
        if not os.path.isdir(unitask_models_dir):
            os.mkdir(unitask_models_dir)
        
        unitask_models_of_prediction_mode_dir = os.path.join(
            unitask_models_dir, prediction_mode
        )
        if not os.path.isdir(unitask_models_of_prediction_mode_dir):
            os.mkdir(unitask_models_of_prediction_mode_dir)        

        self.dir_all_models = os.path.join(
            unitask_models_of_prediction_mode_dir, 
            self.context_response_pair_data_name
        )
        if debug:
            self.dir_all_models += ".debug"
        if not os.path.isdir(self.dir_all_models):
            os.mkdir(self.dir_all_models)

        model_name = (
            f"{base_model.split('/')[-1]}.joint_model" +
            f".max_seq_len_{self.max_seq_len}" +
            f".seed_{self.hyperparameters['random_seed']}" +
            f".bs_{self.hyperparameters['per_device_train_batch_size']}" +
            f".lr_{self.hyperparameters['learning_rate']}" +
            f".warmup_ratio_{self.hyperparameters['warmup_ratio']}" +
            f".weight_decay_{self.hyperparameters['weight_decay']}" +
            f".grad_acc_{self.hyperparameters['gradient_accumulation_steps']}" +
            f".epochs_{self.hyperparameters['num_train_epochs']}"
        )

        if self.use_adapters:
            model_name += ".adapters"
        if self.do_domain_adaptation:
            model_name += ".domain_adapted"

        if self.do_domain_adaptation:
            self.dir_domain_adapted_and_fine_tuned_models = os.path.join(
                self.dir_all_models, model_name
            )
            self.dir_domain_adapted_models = os.path.join(
                self.dir_domain_adapted_and_fine_tuned_models, "domain_adaptation"
            )
            self.dir_fine_tuned_models = os.path.join(
                self.dir_domain_adapted_and_fine_tuned_models, "fine_tuning"
            )
            for dir_path in [
                self.dir_domain_adapted_and_fine_tuned_models, 
                self.dir_domain_adapted_models, 
                self.dir_fine_tuned_models
            ]:
                if not os.path.isdir(dir_path):
                    os.mkdir(dir_path)                
        else:
            self.dir_fine_tuned_models = os.path.join(
                self.dir_all_models, model_name
            )
            if not os.path.isdir(self.dir_fine_tuned_models):
                os.mkdir(self.dir_fine_tuned_models)  

        # Lazy loading --- so that tokenization of the data splits is 
        # only done when necessary
        self.tokenizer_manager = None
        self.tokenizer = None
        self.domain_adaptation_split_dataset_dicts = None
        self.split_dataset_dicts_tokenized = None
        self.label2id = None
        self.id2label = None

        self.f1 = load_metric("f1")


    def tokenize_split_dataset_dicts(self):
        if self.split_dataset_dicts_tokenized is not None:
            return

        if self.tokenizer is None or self.tokenizer_manager is None:
            self.tokenizer_manager = TokenizerManagerForUniTaskCodePredictionModels(
                base_model=self.base_model,
                max_seq_len=self.max_seq_len
            )
            self.tokenizer = self.tokenizer_manager.tokenizer

        self.split_dataset_dicts_tokenized = dict()
        self.label2id = None
        self.id2label = None
        for split_id, _ in zip(
            self.split_dataset_dicts.keys(),
            range(1 if self.use_only_one_split_for_debug or self.debug else 100000)
        ):
            logger.info(f"Tokenizing split {split_id}")
            if self.label2id is None:
                label_list = self.split_dataset_dicts[split_id]["train"].unique("response_code")
                self.label2id = {l: i for i, l in enumerate(label_list)}
                self.id2label = {id: label for label, id in self.label2id.items()} 

            self.tokenizer_manager.label2id = self.label2id
            if self.prediction_mode == "current_turn_code_prediction":
                self.split_dataset_dicts_tokenized[split_id] = \
                    self.split_dataset_dicts[split_id].map(
                        self.tokenizer_manager.tokenize_response_batch_with_right_truncation,
                        batched=True,
                        batch_size=(5 if self.debug else 1000),
                        num_proc=(1 if self.debug else 4),
                    )  
            else:
                if self.mi_quality in ["high", "low"]:
                    self.split_dataset_dicts_tokenized[split_id] = \
                        self.split_dataset_dicts[split_id].map(
                            self.tokenizer_manager.tokenize_context_batch_with_left_truncation,
                            batched=True,
                            batch_size=(5 if self.debug else 1000),
                            num_proc=(1 if self.debug else 4),
                        )  
                else:
                    assert self.mi_quality is None
                    self.split_dataset_dicts_tokenized[split_id] = \
                        self.split_dataset_dicts[split_id].map(
                            self.tokenizer_manager.tokenize_context_batch_with_left_truncation_and_mi_quality_info,
                            batched=True,
                            batch_size=(5 if self.debug else 1000),
                            num_proc=(1 if self.debug else 4),
                            # num_proc=1,
                        )                      


    def create_domain_adaptation_split_dataset_dicts(
        self,
        debug: bool=False,
        overwrite: bool=False
    ):
        """
        Domain adaptation data preprocessing
        """
        return NotImplementedError(
            f"""
            Domain adaptation dataset dicts have NOT
            been implemented!
            """
        )


    def get_default_hyperparameters(self):
        if self.base_model in [
            ROBERTA_LARGE_CHECKPOINT, 
            ROBERTA_BASE_CHECKPOINT
        ]:
            return {
                "per_device_train_batch_size": 8,
                "per_device_eval_batch_size": 16,
                "max_seq_len": 512,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "num_train_epochs": 10,
                "random_seed": 42,
                "gradient_accumulation_steps": 1,
            }
        else:
            raise ValueError(
                f"""
                Cannot get default hyperparameters for model
                {self.base_model}
                """
            )
        

    def compute_macro_f1(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.f1.compute(
            predictions=predictions, 
            references=labels,
            average="macro"
        )


    def cross_validation_pipeline(
        self,
        debug: bool=False,
        overwrite: bool=False,
        also_get_label_prediction_order: bool=False,
        only_collect_results: bool=False,
        get_performance_on_high_quality_mi_examples_only: bool=False
    ):
        """
        The complete pipeline for cross-validating
        a model.

        First do training, then use the validation-time
        optimal models to get the results --- 
        effectively on the whole dataset, because we're
        using cross validation here
        """
        cross_validation_output_path = os.path.join(
            self.dir_fine_tuned_models,
            "cross_validation_test_results.csv"
        )
        cross_validation_output_with_label_prediction_order_path = \
            f"{cross_validation_output_path}.with_prediction_order.csv"   
        if only_collect_results:
            if not os.path.isfile(cross_validation_output_path):
                raise FileNotFoundError(
                    f"""
                    I couldn't find the file of cross validation results
                    {cross_validation_output_path}
                    which means you most likely haven't trained the models
                    """
                )
        else:
            if not os.path.isfile(cross_validation_output_path):
                self.cross_validation_train(debug=debug, overwrite=overwrite)
            if not (
                os.path.isfile(cross_validation_output_path) and 
                (
                    not also_get_label_prediction_order or 
                    (
                        also_get_label_prediction_order and
                        os.path.isfile(cross_validation_output_with_label_prediction_order_path)
                    )
                )
            ):            
                self.cross_validation_test(
                    output_path=cross_validation_output_path,
                    output_with_label_prediction_order_path=cross_validation_output_with_label_prediction_order_path,
                    overwrite=overwrite,
                    debug=debug,
                    also_get_label_prediction_order=also_get_label_prediction_order
                )
        return self.calc_cross_validation_code_prediction_performance(
            cross_validation_output_path,
            cross_validation_output_with_label_prediction_order_path,
            get_performance_on_high_quality_mi_examples_only=get_performance_on_high_quality_mi_examples_only
        )


    def get_test_set_performance_per_split(
        self,
        get_performance_on_high_quality_mi_examples_only: bool=False
    ):
        """
        The complete pipeline for cross-validating
        a model.

        First do training, then use the validation-time
        optimal models to get the results --- 
        effectively on the whole dataset, because we're
        using cross validation here
        """
        cross_validation_output_path = os.path.join(
            self.dir_fine_tuned_models,
            "cross_validation_test_results.csv"
        )
        if not os.path.isfile(cross_validation_output_path):
            raise FileNotFoundError(
                f"""
                I couldn't find the file of cross validation results
                {cross_validation_output_path}
                which means you most likely haven't trained the models
                """
            )
        per_split_performances = {
            "f1": dict(),
            "mcc": dict()
        }
        for split_id in range(self.num_fold):
            test_set_results_path = os.path.join(
                self.dir_fine_tuned_models, 
                f"split_{split_id}",
                "test_set.results.csv"
            )
            test_set_results = pd.read_csv(test_set_results_path)

            if get_performance_on_high_quality_mi_examples_only:
                test_set_df = self.split_dataset_dicts[split_id]["test"].to_pandas()
                high_quality_anno_mi_row_ids = set(test_set_df.loc[
                    test_set_df["mi_quality"] == "high"
                ]["anno_mi_row_id"])
                test_set_results = test_set_results.loc[
                    test_set_results["anno_mi_row_id"].isin(high_quality_anno_mi_row_ids)
                ]                
            
            performance_f1 = f1_score(
                y_true=test_set_results["ground_truth_code"],
                y_pred=test_set_results["predicted_code"],
                average="macro"        
            )
            performance_mcc = mcc(
                y_true=test_set_results["ground_truth_code"],
                y_pred=test_set_results["predicted_code"],
            )
            per_split_performances["f1"][f"split_{split_id}"] = performance_f1
            per_split_performances["mcc"][f"split_{split_id}"] = performance_mcc
        return per_split_performances     


    def cross_validation_train(
        self,
        debug: bool=False,
        overwrite: bool=False,
    ):
        """
        Train a model on the training set of each split
        """
        if self.split_dataset_dicts_tokenized is None:
            self.tokenize_split_dataset_dicts()

        domain_adapted_models_paths = dict()

        if self.do_domain_adaptation:
            if self.domain_adaptation_split_dataset_dicts is None:
                self.create_domain_adaptation_split_dataset_dicts()

            for split_id, domain_adaptation_split_data in self.domain_adaptation_split_dataset_dicts.items():
                output_dir = os.path.join(
                    self.dir_domain_adapted_models, 
                    f"split_{split_id}"
                )

                training_is_complete_flag_path = os.path.join(
                    output_dir, "training_is_complete.flag"
                )

                if os.path.isfile(training_is_complete_flag_path) and not overwrite:
                    logger.info(f"Model already trained at {output_dir}. Skipping training now.")

                else:
                    if os.path.isdir(output_dir):
                        shutil.rmtree(output_dir)
                    os.mkdir(output_dir)            

                    # Train
                    model = AutoModelForMaskedLM.from_pretrained(
                        self.base_model,
                        cache_dir=TRANSFORMERS_CACHE_DIR
                    )
                    training_args = TrainingArguments(
                        output_dir=output_dir,
                        overwrite_output_dir=overwrite,
                        do_train=True,
                        do_eval=True,
                        evaluation_strategy=("steps" if debug else "epoch"),
                        eval_steps=(50 if debug else None),
                        per_device_train_batch_size=self.hyperparameters["per_device_train_batch_size"],
                        per_device_eval_batch_size=self.hyperparameters["per_device_eval_batch_size"],            
                        learning_rate=self.hyperparameters["learning_rate"],
                        weight_decay=self.hyperparameters["weight_decay"],
                        num_train_epochs=self.hyperparameters["num_train_epochs"],
                        seed=self.hyperparameters["random_seed"],
                        gradient_accumulation_steps=self.hyperparameters["gradient_accumulation_steps"],
                        logging_strategy="steps",
                        logging_first_step=True,
                        logging_steps=(10 if debug else 100),
                        save_strategy=("steps" if debug else "epoch"),
                        save_steps=(50 if debug else None),
                        save_total_limit=1,
                        fp16=True,
                        load_best_model_at_end=True,
                        push_to_hub=False,
                        report_to="none",
                        max_steps=(100 if debug else -1),
                        warmup_ratio=self.hyperparameters["warmup_ratio"],
                    )
                    data_collator = DataCollatorForLanguageModeling(
                        tokenizer=self.tokenizer, 
                        mlm_probability=0.15
                    )
                    trainer_class = Trainer
                    trainer = trainer_class(
                        model=model,
                        args=training_args,
                        train_dataset=domain_adaptation_split_data["train"],
                        eval_dataset=domain_adaptation_split_data["dev"],
                        data_collator=data_collator,
                        callbacks=[
                            MyTrainerCallback(
                                remove_optimizer_of_best_checkpoint_on_train_end=True
                            ),
                            EarlyStoppingCallback(
                                early_stopping_patience=3,
                                early_stopping_threshold=0
                            )
                        ]
                    )
                    trainer.train()
            
                best_checkpoint_path = os.path.join(output_dir, "best_checkpoint")
                domain_adapted_models_paths[split_id] = best_checkpoint_path

        for split_id, split_data_tokenized in self.split_dataset_dicts_tokenized.items():
            output_dir = os.path.join(self.dir_fine_tuned_models, f"split_{split_id}")

            training_is_complete_flag_path = os.path.join(output_dir, "training_is_complete.flag")
            # regularized_model_name = model_name.replace('.', '-')

            if os.path.isfile(training_is_complete_flag_path) and not overwrite:
                logger.info(f"Model already trained at {output_dir}. Skipping training now.")

            else:
                if os.path.isdir(output_dir):
                    shutil.rmtree(output_dir)
                os.mkdir(output_dir)            

                # Train
                if self.do_domain_adaptation:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        domain_adapted_models_paths[split_id]
                    )
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.base_model,
                        num_labels=len(self.label2id),
                        cache_dir=TRANSFORMERS_CACHE_DIR,
                    )
                model.config.label2id = self.label2id
                model.config.id2label = self.id2label

                effective_batch_size = (
                    self.hyperparameters["per_device_train_batch_size"] *
                    self.hyperparameters["gradient_accumulation_steps"]
                )
                if self.debug:
                    evaluation_strategy = "steps"
                    eval_steps = 50
                    max_steps = 100
                elif self.use_augmentation_during_training:
                    unaugmented_train_data_size = (
                        len(split_data_tokenized["train"]) //
                        self.num_augs_to_use_per_train_example
                    )
                    evaluation_strategy = "steps"
                    # Basically, fixing max_steps to be the number of steps
                    # that would be taken when an unaugmented dataset was being used 
                    # for training.
                    # Similarly, fix eval_steps to be the number evaluation steps
                    # that would be the case if an unaugmented dataset was being used 
                    # for training.
                    eval_steps = unaugmented_train_data_size // effective_batch_size
                    max_steps = self.hyperparameters["num_train_epochs"] * eval_steps
                else:
                    evaluation_strategy = "epoch"
                    eval_steps = None      
                    max_steps = -1

                training_args = TrainingArguments(
                    output_dir=output_dir,
                    overwrite_output_dir=overwrite,
                    do_train=True,
                    do_eval=True,
                    evaluation_strategy=evaluation_strategy,
                    eval_steps=eval_steps,
                    per_device_train_batch_size=self.hyperparameters["per_device_train_batch_size"],
                    per_device_eval_batch_size=self.hyperparameters["per_device_eval_batch_size"],            
                    learning_rate=self.hyperparameters["learning_rate"],
                    weight_decay=self.hyperparameters["weight_decay"],
                    num_train_epochs=self.hyperparameters["num_train_epochs"],
                    seed=self.hyperparameters["random_seed"],
                    gradient_accumulation_steps=self.hyperparameters["gradient_accumulation_steps"],
                    logging_strategy="steps",
                    logging_first_step=True,
                    logging_steps=(10 if debug else 100),
                    save_strategy=evaluation_strategy,
                    save_steps=eval_steps,
                    save_total_limit=1,
                    fp16=True,
                    load_best_model_at_end=True,
                    push_to_hub=False,
                    report_to="none",
                    max_steps=max_steps,
                    warmup_ratio=self.hyperparameters["warmup_ratio"],
                    metric_for_best_model="f1",
                    greater_is_better=True,
                )

                # trainer_class = AdapterTrainer if self.use_adapters else Trainer
                trainer_class = Trainer
                trainer = trainer_class(
                    model=model,
                    args=training_args,
                    train_dataset=split_data_tokenized["train"],
                    eval_dataset=split_data_tokenized["dev"],
                    callbacks=[
                        MyTrainerCallback(
                            remove_optimizer_of_best_checkpoint_on_train_end=True
                        ),
                        EarlyStoppingCallback(
                            early_stopping_patience=3,
                            early_stopping_threshold=0
                        )                        
                    ],
                    compute_metrics=self.compute_macro_f1
                )             
                trainer.train()

                # model.save_adapter("adapter_anno_mi", "anno_mi")


    def cross_validation_test(
        self,
        output_path: str=None,
        output_with_label_prediction_order_path: str=None,
        overwrite: bool=False,
        debug: bool=False,
        also_get_label_prediction_order: bool=False
    ):
        """
        Test the model of every split on the 
        test set of every split
        """
        if output_path is None:
            output_path = os.path.join(
                self.dir_fine_tuned_models,
                "cross_validation_test_results.csv"
            )
        if output_with_label_prediction_order_path is None:
            output_with_label_prediction_order_path = \
                f"{output_path}.with_prediction_order.csv"
        
        if (
            os.path.isfile(output_path) and 
            (
                not also_get_label_prediction_order or 
                (
                    also_get_label_prediction_order and
                    os.path.isfile(output_with_label_prediction_order_path)
                )
            ) and
            not overwrite
        ):
            logger.info(
                f"""Cross Validation test is over. The results are stored
                at {output_path}
                """
            )
            all_test_results = pd.read_csv(output_path)
            if also_get_label_prediction_order:
                all_test_results_with_label_prediction_order = \
                    pd.read_csv(output_with_label_prediction_order_path)
            else:
                all_test_results_with_label_prediction_order = None
        else:
            if self.split_dataset_dicts_tokenized is None:
                self.tokenize_split_dataset_dicts()

            all_test_results = []
            all_test_results_with_label_prediction_order = []
            for split_id, split_data_tokenized in self.split_dataset_dicts_tokenized.items():
                output_dir = os.path.join(self.dir_fine_tuned_models, f"split_{split_id}")
                best_checkpoint_dir = os.path.join(output_dir, "best_checkpoint")

                predictions_output_path = os.path.join(
                    output_dir, 
                    "test_set.results.csv"
                )
                predictions_output_with_prediction_order_path = os.path.join(
                    output_dir, 
                    f"{predictions_output_path}.with_prediction_order.csv"
                )      

                if (
                    os.path.isfile(predictions_output_path) and 
                    (
                        not also_get_label_prediction_order or 
                        (
                            also_get_label_prediction_order and
                            os.path.isfile(predictions_output_with_prediction_order_path)
                        )
                    ) and                    
                    not overwrite
                ):
                    logger.info(
                        f"""
                        Therapist/Client code predictions on test set
                        already exist at {predictions_output_path}
                        We're skipping it now.
                        """
                    )
                else:            
                    logger.info(
                        f"""
                        Testing model {best_checkpoint_dir} on its test set
                        """
                    )
                    test_set = split_data_tokenized["test"]
                    model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_dir)
                    model.to(self.device)
                    self.test_code_prediction(
                        dataset=test_set,
                        output_path=predictions_output_path,
                        output_with_label_prediction_order_path=predictions_output_with_prediction_order_path,
                        model=model,
                        debug=debug
                    )
                test_results = pd.read_csv(predictions_output_path)
                all_test_results.append(test_results)
                if os.path.isfile(predictions_output_with_prediction_order_path):
                    test_results_with_label_prediction_order = \
                        pd.read_csv(predictions_output_with_prediction_order_path)
                    all_test_results_with_label_prediction_order.append(
                        test_results_with_label_prediction_order
                    )

            all_test_results = pd.concat(all_test_results)
            all_test_results = all_test_results.sort_values(
                by=["anno_mi_row_id"]
            ).reset_index(drop=True)
            all_test_results.to_csv(output_path, index=False)

            if also_get_label_prediction_order:
                all_test_results_with_label_prediction_order = pd.concat(
                    all_test_results_with_label_prediction_order
                )
                all_test_results_with_label_prediction_order = \
                    all_test_results_with_label_prediction_order.sort_values(
                        by=["anno_mi_row_id"]
                    ).reset_index(drop=True)
                all_test_results_with_label_prediction_order.to_csv(
                    output_with_label_prediction_order_path, index=False
                )


    def test_code_prediction(
        self,
        dataset: Dataset,
        output_path: str,
        model: PreTrainedModel,
        output_with_label_prediction_order_path: str=None,
        debug: bool=False,
    ):
        """
        Test `model` on `dataset` and write the results to
        `output_path`
        """
        if debug:
            dataset_ = dataset.select(range(20))
        else:
            dataset_ = dataset

        def predict_batch(examples: Dict[str, List[Any]]):
            token_type_ids = examples.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = torch.tensor(token_type_ids).to(self.device)
            prediction = model(
                input_ids=torch.tensor(examples["input_ids"]).to(self.device),
                attention_mask=torch.tensor(examples["attention_mask"]).to(self.device),
                token_type_ids=token_type_ids
            )               
            prediction_logits = prediction.logits
            prediction_labels = prediction_logits.argmax(dim=1).detach().cpu()
            prediction_labels_with_order = prediction_logits.argsort(
                dim=1, descending=True
            ).detach().cpu()

            # Make sure the predicted labels are 1D, i.e. one label per example
            assert len(list(prediction_labels.shape)) == 1
            prediction_labels = prediction_labels.tolist()
            prediction_labels = [
                model.config.id2label[label] for label in prediction_labels
            ]

            assert len(list(prediction_labels_with_order.shape)) == 2
            prediction_labels_with_order = prediction_labels_with_order.tolist()
            prediction_labels_with_order = [
                [model.config.id2label[label] for label in ordered_labels_for_one_example] \
                    for ordered_labels_for_one_example in prediction_labels_with_order
            ]            

            return {
                "response_predicted_code": prediction_labels,
                "response_predicted_codes_with_order": prediction_labels_with_order
            }
        
        dataset_with_predictions = dataset_.map(
            predict_batch,
            batched=True,
            batch_size=self.hyperparameters["per_device_eval_batch_size"]
        )
        results_to_persist = {
            "anno_mi_row_id": dataset_with_predictions["anno_mi_row_id"],
            "ground_truth_code": dataset_with_predictions["response_code"],
            "predicted_code": dataset_with_predictions["response_predicted_code"],
        }
        results_to_persist = pd.DataFrame.from_dict(results_to_persist)
        results_to_persist.to_csv(
            output_path, index=False
        )

        if output_with_label_prediction_order_path is not None:
            results_to_persist = {
                "anno_mi_row_id": dataset_with_predictions["anno_mi_row_id"],
                "ground_truth_code": dataset_with_predictions["response_code"],
                "predicted_code": dataset_with_predictions["response_predicted_code"],
                "predicted_codes_ordered": dataset_with_predictions["response_predicted_codes_with_order"],
            }
            results_to_persist = pd.DataFrame.from_dict(results_to_persist)
            results_to_persist["predicted_codes_ordered"] = results_to_persist["predicted_codes_ordered"].apply(
                lambda predicted_codes_ordered: json.dumps(predicted_codes_ordered)
            )
            results_to_persist.to_csv(
                output_with_label_prediction_order_path, index=False
            )            


    def calc_cross_validation_code_prediction_performance(
        self,
        cross_validation_test_results_path: str,
        cross_validation_test_results_with_label_prediction_order_path=None,
        top_k: int=2,
        get_performance_on_high_quality_mi_examples_only: bool=False
    ):
        if os.path.isfile(cross_validation_test_results_path):
            all_test_results = pd.read_csv(cross_validation_test_results_path)
        else:
            return None

        all_test_sets_df = pd.concat([
            dataset_dict["test"].to_pandas() for _, dataset_dict \
                in self.split_dataset_dicts.items()
        ])
        assert (
            sorted(all_test_sets_df["anno_mi_row_id"]) == 
            sorted(all_test_results["anno_mi_row_id"])
        )
        high_quality_anno_mi_row_ids = set(all_test_sets_df.loc[
            all_test_sets_df["mi_quality"] == "high"
        ]["anno_mi_row_id"])
        if get_performance_on_high_quality_mi_examples_only:
            all_test_results = all_test_results.loc[
                all_test_results["anno_mi_row_id"].isin(high_quality_anno_mi_row_ids)
            ]

        overall_f1 = f1_score(
            y_true=all_test_results["ground_truth_code"],
            y_pred=all_test_results["predicted_code"],
            average="macro"        
        )
        overall_mcc = mcc(
            y_true=all_test_results["ground_truth_code"],
            y_pred=all_test_results["predicted_code"],
        )        

        if os.path.isfile(cross_validation_test_results_with_label_prediction_order_path):
            all_test_results_with_label_prediction_order = \
                pd.read_csv(
                    cross_validation_test_results_with_label_prediction_order_path
                )
        else:
            all_test_results_with_label_prediction_order = None
        if get_performance_on_high_quality_mi_examples_only:
            all_test_results_with_label_prediction_order = \
                all_test_results_with_label_prediction_order.loc[
                    all_test_results_with_label_prediction_order["anno_mi_row_id"]\
                        .isin(high_quality_anno_mi_row_ids)
                ]            

        all_test_results_with_label_prediction_order["predicted_codes_ordered"] = \
            all_test_results_with_label_prediction_order["predicted_codes_ordered"].apply(
                lambda x: json.loads(x)
            )
        ground_truth_in_top_k_prediction = all_test_results_with_label_prediction_order.apply(
            lambda row: row["ground_truth_code"] in row["predicted_codes_ordered"][:top_k],
            axis=1
        )
        top_k_accuracy = (
            sum(ground_truth_in_top_k_prediction) /
            all_test_results_with_label_prediction_order.shape[0]
        )

        logger.info(
            f"""
            Cross Validation performance:
            Overall F1: {overall_f1}
            Overall MCC: {overall_mcc}
            Top-{top_k} accuracy: {top_k_accuracy}
            """
        )

        return {
            "overall_f1": overall_f1,
            "overall_mcc": overall_mcc,
            "top_k_accuracy": top_k_accuracy
        }
