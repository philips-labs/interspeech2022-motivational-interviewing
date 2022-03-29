import itertools
from math import ceil
import os
import random
import json
import re
import numpy as np
from interspeechmi.constants import MODELS_DIR
from interspeechmi.nlp.constants import (
    CLIENT_CODES,
    HF_DATASETS_CACHE_DIR,
    THERAPIST_CODES,
)

import logging
import pandas as pd
from typing import Any, Callable, Dict, List, NamedTuple
from interspeechmi.data_handling.constants import (
    ANNO_MI_NORMALIZED_AUGMENTED_PATH,
    ANNO_MI_NORMALIZED_PATH,
    ANNO_MI_NORMALIZED_PATH, 
    ANNO_MI_DATA_DIR
)
from datasets import load_dataset
from interspeechmi.standalone_utils import (
    json_pretty_str, 
    json_pretty_write
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import shutil
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    TrainerCallback,
    TrainingArguments
)
from transformers.trainer_callback import (
    TrainerControl, 
    TrainerState
)

logger = logging.getLogger(__name__)


class MyTrainerCallback(TrainerCallback):
    """
    Callback used to control the training process

    Here we basically do two things:
    
    1) make sure we use the local logger everytime HF wants to log something
    This is especially useful when you're trying to log to file -- smth that
    HF doesn't do for you automatically, for some reason

    2) make the trainer leave a "training is complete" flag
    in the output folder and document the logged evaluation history.
    """

    def __init__(
        self,
        remove_optimizer_of_best_checkpoint_on_train_end: bool=False
    ) -> None:
        super().__init__()
        self.remove_optimizer_of_best_checkpoint_on_train_end = remove_optimizer_of_best_checkpoint_on_train_end


    def on_log(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        logs=None, 
        **kwargs
    ):
        control.should_log = False
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs) # using your custom logger


    def on_train_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        """
        At the end of a training cycle, produce a "training is complete" flag,
        which is an empty file named "training_is_complete.flag",
        in the output folder

        Also, write the logged history to "log_history.json"
        in the output folder
        """
        # Get the log history
        output_dir = args.output_dir
        metric = args.metric_for_best_model
        performances = [
            log[f"eval_{metric}"] for log in state.log_history if (
                f"eval_{metric}" in log.keys()
            )
        ]
        if args.greater_is_better:
            best_performance = max(performances)
        else:
            best_performance = min(performances)

        # Keep only the best checkpoint and delete the others
        best_checkpoint = None
        for log in state.log_history:
            if f"eval_{metric}" in log.keys():
                checkpoint_performance = log[f"eval_{metric}"]
                step = log["step"]
                # Keep the best checkpoint
                # It is possible that multiple checkpoints have
                # the same best metric value on the dev set. 
                # In such cases, we just keep the earliest best checkpoint
                if (
                    best_checkpoint is None and
                    checkpoint_performance == best_performance
                ):
                    best_checkpoint = step
                # Delete other checkpoints
                else:
                    checkpoint_dir = os.path.join(
                        output_dir, f"checkpoint-{step}"
                    )
                    if os.path.isdir(checkpoint_dir):
                        shutil.rmtree(checkpoint_dir)

        # Rename the folder of the best checkpoint as "best_checkpoint"
        assert best_checkpoint is not None
        best_checkpoint_dir = os.path.join(
            output_dir, f"checkpoint-{best_checkpoint}"
        )
        assert os.path.isdir(best_checkpoint_dir)
        os.rename(
            best_checkpoint_dir,
            os.path.join(
                output_dir,
                "best_checkpoint"
            )
        )
        best_checkpoint_dir = os.path.join(output_dir, "best_checkpoint")

        # Remove the optimizer of the best checkpoint, because it's quite big
        # and not really useful when the training is complete
        if self.remove_optimizer_of_best_checkpoint_on_train_end:
            optimizer_path = os.path.join(best_checkpoint_dir, "optimizer.pt")
            os.remove(optimizer_path)

        # Persist the log history
        eval_log_history_path = os.path.join(output_dir, "log_history.json")
        json_pretty_write(state.log_history, eval_log_history_path)

        # Create "training is complete" flag
        training_is_complete_flag_path = os.path.join(args.output_dir, "training_is_complete.flag")
        with open(training_is_complete_flag_path, 'w') as training_is_complete_flag_writer:
            training_is_complete_flag_writer.write("\n")

        # # Just in case any checkpoint directory remains, delete it
        # checkpoint_pattern = r"^checkpoint-\d+$"
        # for dir_content_name in os.listdir(output_dir):
        #     dir_content_path = os.path.join(output_dir, dir_content_name)
        #     if (
        #         os.path.isdir(dir_content_path) and
        #         re.match(checkpoint_pattern, dir_content_name)
        #     ):
        #         shutil.rmtree(dir_content_path)



def build_annomi_context_and_response_pairs_df(
    anno_mi_data_path: str=ANNO_MI_NORMALIZED_PATH,
    conv_history_window: int=1,
    prepend_codes_in_context: str=None,
    prepend_code_in_response: bool=False,
    context_utterances_connector: str=" ",
    utterance_condition: Callable[[NamedTuple], bool]=None,
    mi_quality_filter: str=None,
    return_response_codes: bool=False,
    random_seed: int=42
):
    """
    Convert transcripts into a DataFrame of (conversation context, response) pairs

    For example, if a conversation from the transcripts looks like

    Client: "How are you?",
    Therapist: "Good. You?",
    Client: "All good",

    Then 2 pairs will be generated, assuming a window size of 2:
    1) Context: "<client> How are you?", Response: "<therapist> Good. You?"
    2) Context: "<client> How are you? <therapist> Good. You?", Response: "<client> All good"

    A DataFrame with 3 columns ["anno_mi_row_id", "context, "response"] will be returned,
    where each row stores a (context, response) pair and the "anno_mi_row_id" column value
    indicates the row ID of the last utterance.

    If `prepend_code` is `True`, we add the therapist behaviour / client talk type
    between the interlocutor identifier and actual utterance. The example above would become
    1) Context: "<client> <neutral> How are you?", Response: "<therapist> <question> Good. You?"
    2) Context: "<client> <neutral> How are you? <therapist> <question> Good. You?", Response: "<client> <neutral> All good"
    """
    assert prepend_codes_in_context in [
        "no", "therapist_oracle", "therapist_predicted",
        "client_oracle", "therapist_and_client_oracle",
        "client_predicted", "therapist_and_client_predicted",
        "therapist_random", "client_random", "therapist_and_client_random"
    ]
    logger.info("Creating <context, response, response_code> pairs ...")

    rand = random.Random(random_seed)

    if prepend_codes_in_context in [
        "therapist_predicted", "client_predicted",
        "therapist_and_client_predicted"
    ]:
        anno_mi_predicted_codes = get_predicted_codes_for_annomi(
            mi_quality_filter=mi_quality_filter,
            get_predicted_therapist_codes=(
                prepend_codes_in_context in [
                    "therapist_predicted", "therapist_and_client_predicted"
                ]
            ),
            get_predicted_client_codes=(
                prepend_codes_in_context in [
                    "client_predicted", "therapist_and_client_predicted"
                ]
            ),
        )
    else:
        anno_mi_predicted_codes = None

    context_strs = []
    response_strs = []
    response_anno_mi_row_ids = []
    response_codes = []
    dialogue_ids = []
    dialogue_mi_qualities = []

    def get_code_processed_utt(interlocutor, code, utt_text):
        if prepend_codes_in_context in [
            "therapist_oracle", "therapist_predicted",
            "therapist_and_client_oracle",
            "therapist_and_client_predicted",
            "therapist_random", "therapist_and_client_random"
        ]:
            prepend_therapist_codes_in_context = True
        elif prepend_codes_in_context in [
            "no", "client_oracle", "client_predicted",
            "client_random"
        ]:
            prepend_therapist_codes_in_context = False
        else:
            raise NotImplementedError(
                f"""
                Unimplemeted way of prepending codes in context: 
                {prepend_codes_in_context}
                """
            )            

        if prepend_codes_in_context in [
            "client_oracle", "client_predicted",
            "therapist_and_client_oracle",
            "therapist_and_client_predicted",
            "client_random", "therapist_and_client_random"
        ]:
            prepend_client_codes_in_context = True
        elif prepend_codes_in_context in [
            "no", "therapist_oracle", "therapist_predicted",
            "therapist_random"
        ]:
            prepend_client_codes_in_context = False
        else:
            raise NotImplementedError(
                f"""
                Unimplemeted way of prepending codes in context: 
                {prepend_codes_in_context}
                """
            ) 

        if interlocutor == "therapist":
            if prepend_therapist_codes_in_context:
                assert code in THERAPIST_CODES
                code_processed_utt = "<{}>~<{}>{}".format(
                    interlocutor, code, utt_text
                ).strip()
            else:
                code_processed_utt = "<{}>{}".format(
                    interlocutor, utt_text
                ).strip()     
        elif interlocutor == "client":
            if prepend_client_codes_in_context:
                assert code in CLIENT_CODES
                code_processed_utt = "<{}>~<{}>{}".format(
                    interlocutor, code, utt_text
                ).strip()
            else:
                code_processed_utt = "<{}>{}".format(
                    interlocutor, utt_text
                ).strip()  
        else:
            raise ValueError(f"Unknown interlocutor: {interlocutor}")

        return code_processed_utt

    # Use a dict to keep track of the sampled codes
    # So that the same utterance in overlapping conversation windows
    # will have the same sampled code, which is more consistent
    random_utt_codes = dict() # {anno_mi_row_id: sampled_code}

    for context_and_response in iter_annomi_utterance_with_context(
        num_preceding_utterances_to_return=conv_history_window,
        anno_mi_data_path=anno_mi_data_path,
        skip_no_context_utterances=True,
        utterance_condition=utterance_condition,
        mi_quality=mi_quality_filter
    ):
        assert isinstance(context_and_response, pd.DataFrame)

        oracle_context_and_response_codes = []
        random_context_and_response_codes = []

        therapist_behaviour_rephrasing_dict = {
            # "question": "asking",
            "question": "question",
            # "therapist_input": "informing",
            "therapist_input": "input",
            # "reflection": "listening",
            "reflection": "reflection",
            "other": "other"
        }
        if prepend_codes_in_context in [
            "therapist_predicted", "client_predicted",
            "therapist_and_client_predicted"
        ]:
            predicted_context_and_response_codes = []
            
        for row in context_and_response.itertuples():
            therapist_behaviour = getattr(row, "main_therapist_behaviour")
            client_talk_type = getattr(row, "client_talk_type")
            assert (
                (therapist_behaviour == "n/a" and client_talk_type != "n/a") or
                (client_talk_type == "n/a" and therapist_behaviour != "n/a")
            )
            if therapist_behaviour == "n/a":
                oracle_context_and_response_codes.append(client_talk_type)
                if row.Index not in random_utt_codes.keys():
                    random_utt_codes[row.Index] = rand.sample(CLIENT_CODES, 1)[0]
            else:
                oracle_context_and_response_codes.append(
                    therapist_behaviour_rephrasing_dict[therapist_behaviour]
                )
                if row.Index not in random_utt_codes.keys():
                    random_utt_codes[row.Index] = rand.sample(THERAPIST_CODES, 1)[0]                
            random_context_and_response_codes.append(random_utt_codes[row.Index])
            
            if prepend_codes_in_context in [
                "therapist_predicted", "client_predicted",
                "therapist_and_client_predicted"
            ]:
                if row.Index not in anno_mi_predicted_codes.keys():
                    if prepend_codes_in_context == "therapist_predicted":
                        if getattr(row, "interlocutor") == "therapist":
                            raise ValueError(
                                f"""
                                AnnoMI row ID {row.Index} not found 
                                in code predictions
                                """
                            )                            
                    elif prepend_codes_in_context == "client_predicted":
                        if getattr(row, "interlocutor") == "client":
                            raise ValueError(
                                f"""
                                AnnoMI row ID {row.Index} not found 
                                in code predictions
                                """
                            )                            
                    else:
                        raise ValueError(
                            f"""
                            AnnoMI row ID {row.Index} not found 
                            in code predictions
                            """
                        )
                    predicted_context_and_response_codes.append(None)
                else:
                    assert (
                        anno_mi_predicted_codes[row.Index]["ground_truth_code"] == 
                        oracle_context_and_response_codes[-1]
                    )
                    predicted_context_and_response_codes.append(
                        anno_mi_predicted_codes[row.Index]["predicted_code"]
                    )
        
        if prepend_codes_in_context in [
            "therapist_predicted", "client_predicted",
            "therapist_and_client_predicted"
        ]:
            assert (
                len(oracle_context_and_response_codes) == 
                len(predicted_context_and_response_codes)
            )
            codes_to_use_in_context = predicted_context_and_response_codes[:-1]
        elif prepend_codes_in_context in [
            "therapist_random", "client_random",
            "therapist_and_client_random"
        ]:
            assert (
                len(oracle_context_and_response_codes) == 
                len(random_context_and_response_codes)
            )        
            codes_to_use_in_context = random_context_and_response_codes[:-1]
        else:
            codes_to_use_in_context = oracle_context_and_response_codes[:-1]

        context_str = context_utterances_connector.join([
            get_code_processed_utt(interlocutor, code, utt_text) \
                for interlocutor, code, utt_text in zip(
                    context_and_response["interlocutor"].iloc[:-1], 
                    codes_to_use_in_context,
                    context_and_response["utterance_text"].iloc[:-1]
                )
        ])

        response_interlocutor = context_and_response["interlocutor"].iloc[-1]
        response_code = oracle_context_and_response_codes[-1]
        response_text = context_and_response["utterance_text"].iloc[-1]
        dialogue_mi_quality = context_and_response["mi_quality"].iloc[-1]

        response_str = f"<{response_interlocutor}>"
        # Basically, if the dataset contains both high- and low-quality-MI
        # conversations, we need to use a special label at the beginning
        # of the response string of a therapist response, because high-
        # and low-quality-MI therapists are different.
        # This is not needed for client responses, because the client in
        # a high-quality session should not be too different from that 
        # in a low-quality session
        if mi_quality_filter is None and response_interlocutor == "therapist":
            response_str += "~<{}>".format(
                "good" if dialogue_mi_quality == "high" else "bad"
            )
        if prepend_code_in_response:
            response_str += f"~<{response_code}>"
        response_str += response_text
        response_str = response_str.strip()

        response_anno_mi_row_id = context_and_response.index[-1]
        dialogue_id = context_and_response.iloc[-1]["transcript_id"]

        context_strs.append(context_str)
        response_strs.append(response_str)
        response_anno_mi_row_ids.append(response_anno_mi_row_id)
        response_codes.append(response_code)
        dialogue_ids.append(dialogue_id)
        dialogue_mi_qualities.append(dialogue_mi_quality)
        
    annomi_context_and_response_pairs_dict = {
        "anno_mi_row_id": response_anno_mi_row_ids,
        "dialogue_id": dialogue_ids,
        "context": context_strs,
        "response": response_strs,
        "mi_quality": dialogue_mi_qualities,
    }
    if return_response_codes:
        annomi_context_and_response_pairs_dict["response_code"] = response_codes
    
    return pd.DataFrame.from_dict(annomi_context_and_response_pairs_dict)


def build_augmented_annomi_context_and_response_pairs_df(
    anno_mi_augmented_data_path: str=ANNO_MI_NORMALIZED_AUGMENTED_PATH,
    conv_history_window: int=1,
    prepend_codes_in_context: str=None,
    prepend_code_in_response: bool=False,
    context_utterances_connector: str=" ",
    utterance_condition: Callable[[NamedTuple], bool]=None,
    mi_quality_filter: str=None,
    return_response_codes: bool=False,
    num_augs_to_use: int=5,
    random_seed: int=42,
):
    """
    Convert transcripts into a DataFrame of (conversation context, response) pairs

    For example, if a conversation from the transcripts looks like

    Client: "How are you?",
    Therapist: "Good. You?",
    Client: "All good",

    Then 2 pairs will be generated, assuming a window size of 2:
    1) Context: "<client> How are you?", Response: "<therapist> Good. You?"
    2) Context: "<client> How are you? <therapist> Good. You?", Response: "<client> All good"

    A DataFrame with 3 columns ["anno_mi_row_id", "context, "response"] will be returned,
    where each row stores a (context, response) pair and the "anno_mi_row_id" column value
    indicates the row ID of the last utterance.

    If `prepend_code` is `True`, we add the therapist behaviour / client talk type
    between the interlocutor identifier and actual utterance. The example above would become
    1) Context: "<client> <neutral> How are you?", Response: "<therapist> <question> Good. You?"
    2) Context: "<client> <neutral> How are you? <therapist> <question> Good. You?", Response: "<client> <neutral> All good"
    """
    logger.info("Creating augmented <context, response, response_code> pairs ...")

    assert prepend_codes_in_context == "no"

    context_strs = []
    response_strs = []
    response_anno_mi_row_ids = []
    response_codes = []
    dialogue_ids = []
    dialogue_mi_qualities = []

    rand = random.Random(random_seed)

    def get_code_processed_utt(interlocutor, code, utt_text):
        if prepend_codes_in_context == "no":
            prepend_therapist_codes_in_context = False
        else:
            raise NotImplementedError(
                f"""
                Unimplemeted way of prepending codes in context: 
                {prepend_codes_in_context}
                """
            )            

        if interlocutor == "therapist":
            assert code is None
            assert not prepend_therapist_codes_in_context
            code_processed_utt = "<{}>{}".format(
                interlocutor, utt_text
            ).strip()     
        else:
            assert interlocutor == "client"
            assert code is None
            code_processed_utt = "<{}>{}".format(
                interlocutor, utt_text
            ).strip()

        return code_processed_utt

    for context_and_response in iter_annomi_utterance_with_context(
        num_preceding_utterances_to_return=conv_history_window,
        anno_mi_data_path=anno_mi_augmented_data_path,
        skip_no_context_utterances=True,
        utterance_condition=utterance_condition,
        mi_quality=mi_quality_filter
    ):
        assert isinstance(context_and_response, pd.DataFrame)

        oracle_context_and_response_codes = []
        therapist_behaviour_rephrasing_dict = {
            # "question": "asking",
            "question": "question",
            # "therapist_input": "informing",
            "therapist_input": "input",
            # "reflection": "listening",
            "reflection": "reflection",
            "other": "other"
        }

        for row in context_and_response.itertuples():
            therapist_behaviour = getattr(row, "main_therapist_behaviour")
            client_talk_type = getattr(row, "client_talk_type")
            assert (
                (therapist_behaviour == "n/a" and client_talk_type != "n/a") or
                (client_talk_type == "n/a" and therapist_behaviour != "n/a")
            )
            if therapist_behaviour == "n/a":
                oracle_context_and_response_codes.append(client_talk_type)
            else:
                oracle_context_and_response_codes.append(
                    therapist_behaviour_rephrasing_dict[therapist_behaviour]
                )

        # codes_to_use_in_context = oracle_context_and_response_codes[:-1]
        sampled_context_strs = []
        max_sample_attempts = 500
        assert max_sample_attempts >= 5 * num_augs_to_use
        for _ in range(max_sample_attempts):

            if len(sampled_context_strs) >= num_augs_to_use:
                break

            context_sampled_utts = []
            for interlocutor, utt, utt_augmentations_serialized in zip(
                context_and_response["interlocutor"].iloc[:-1], 
                context_and_response["utterance_text"].iloc[:-1],
                context_and_response["utterance_augmentations"].iloc[:-1]
            ):
                utt_augmentations = json.loads(utt_augmentations_serialized)
                texts_to_sample_from = utt_augmentations + [utt]
                sampled_utt_text = rand.sample(texts_to_sample_from, 1)[0]

                sampled_utt_text_with_meta_info = get_code_processed_utt(
                    interlocutor=interlocutor,
                    code=None,
                    utt_text=sampled_utt_text
                )

                context_sampled_utts.append(sampled_utt_text_with_meta_info)
            context_str = context_utterances_connector.join(context_sampled_utts)
            if context_str not in sampled_context_strs:
                sampled_context_strs.append(context_str)

        if len(sampled_context_strs) < num_augs_to_use:
            logger.warning(
                f"""
                I wasn't able to get {num_augs_to_use} different
                random samples of augmented responses.

                Below was my best effort at it:
                {sampled_context_strs}

                I'm gonna make it to {num_augs_to_use} anyway by
                repeating some samples
                """
            )
            num_forced_resamples = num_augs_to_use - len(sampled_context_strs)
            if num_forced_resamples == 1:
                sampled_context_strs.append(sampled_context_strs[0])
            else:
                sampled_context_strs += sampled_context_strs[:num_forced_resamples]

        assert len(sampled_context_strs) == num_augs_to_use

        response_interlocutor = context_and_response["interlocutor"].iloc[-1]
        response_code = oracle_context_and_response_codes[-1]
        response_text = context_and_response["utterance_text"].iloc[-1]
        response_text_augmentations = json.loads(context_and_response["utterance_augmentations"].iloc[-1])
        dialogue_mi_quality = context_and_response["mi_quality"].iloc[-1]

        # Make sure that the ground-truth response is included before 
        # sampling from the augmentations --- because we will need
        # the ground-truth response in the downstream tasks
        sampled_response_strs = []
        sampled_utt_texts = rand.sample(
            response_text_augmentations, 
            num_augs_to_use - 1
        ) + [response_text]
        for sampled_utt_text in sampled_utt_texts:
            sampled_response_str = f"<{response_interlocutor}>"
            # Basically, if the dataset contains both high- and low-quality-MI
            # conversations, we need to use a special label at the beginning
            # of the response string of a therapist response, because high-
            # and low-quality-MI therapists are different.
            # This is not needed for client responses, because the client in
            # a high-quality session should not be too different from that 
            # in a low-quality session
            if mi_quality_filter is None and response_interlocutor == "therapist":
                sampled_response_str += "~<{}>".format(
                    "good" if dialogue_mi_quality == "high" else "bad"
                )
            if prepend_code_in_response:
                sampled_response_str += f"~<{response_code}>"
            sampled_response_str += sampled_utt_text
            sampled_response_str = sampled_response_str.strip()
            sampled_response_strs.append(sampled_response_str)        

        response_anno_mi_row_id = context_and_response.index[-1]
        dialogue_id = context_and_response.iloc[-1]["transcript_id"]

        context_strs += sampled_context_strs
        response_strs += sampled_response_strs
        response_anno_mi_row_ids += [response_anno_mi_row_id] * num_augs_to_use
        response_codes += [response_code] * num_augs_to_use
        dialogue_ids += [dialogue_id] * num_augs_to_use
        dialogue_mi_qualities += [dialogue_mi_quality] * num_augs_to_use
        
    annomi_context_and_response_pairs_dict = {
        "anno_mi_row_id": response_anno_mi_row_ids,
        "dialogue_id": dialogue_ids,
        "context": context_strs,
        "response": response_strs,
        "mi_quality": dialogue_mi_qualities,
    }
    if return_response_codes:
        annomi_context_and_response_pairs_dict["response_code"] = response_codes
    
    return pd.DataFrame.from_dict(annomi_context_and_response_pairs_dict)


def iter_annomi_utterance_with_context(
    anno_mi_data_path: str,
    num_preceding_utterances_to_return: int=0,
    utterance_condition: Callable[[NamedTuple], bool]=None,
    mi_quality: str=None,
    skip_no_context_utterances: bool=False,
):
    """
    Iterate over all the reflections, optionally with 
    `num_preceding_utterances_to_return` preceding utterances
    """
    anno_mi_data = pd.read_csv(anno_mi_data_path, keep_default_na=False)

    if mi_quality in ["high", "low"]:
        anno_mi_data = anno_mi_data.loc[
            anno_mi_data["mi_quality"] == mi_quality
        ].reset_index(drop=True)
    
    for row in tqdm(anno_mi_data.itertuples(), total=anno_mi_data.shape[0]):
        if utterance_condition is not None and not utterance_condition(row):
            continue

        if num_preceding_utterances_to_return > 0:
            # Get the reflection's row as well as the 
            # `num_preceding_utterances_to_return` rows before it
            # as the context of the reflection
            df_utterance_with_context = anno_mi_data.iloc[
                max(0, (row.Index - num_preceding_utterances_to_return)):
                (row.Index + 1)
            ]
            # Make sure the utterances in the context are from the same
            # conversation of the reflection in question
            df_utterance_with_context = df_utterance_with_context.loc[
                df_utterance_with_context["transcript_id"] == row.transcript_id
            ].copy(deep=True)
            if df_utterance_with_context.shape[0] < 2 and skip_no_context_utterances:
                continue
            yield df_utterance_with_context
        else:
            yield anno_mi_data.iloc[row.Index].to_frame().T


def get_predicted_codes_for_annomi(
    get_predicted_therapist_codes: bool=False,
    get_predicted_client_codes: bool=False,
    mi_quality_filter: str=None,
    all_models_dir: str=MODELS_DIR
):
    assert get_predicted_therapist_codes or get_predicted_client_codes
    if mi_quality_filter == "high":
        therapist_code_prediction_model_dir = os.path.join(
            all_models_dir,
            "unitask_code_prediction",
            "current_turn_code_prediction",
            "anno_mi.high_quality.responses.no_code_in_response.clean_separation_of_dialogues.response_from_therapist_only.5_fold_splits",
            "roberta-base.joint_model.max_seq_len_512.seed_42.bs_8.lr_2e-05.warmup_ratio_0.0.weight_decay_0.01.grad_acc_1.epochs_10"
        )
        client_code_prediction_model_dir = os.path.join(
            all_models_dir,
            "unitask_code_prediction",
            "current_turn_code_prediction",
            "anno_mi.high_quality.responses.no_code_in_response.clean_separation_of_dialogues.response_from_client_only.5_fold_splits",
            "roberta-base.joint_model.max_seq_len_512.seed_42.bs_8.lr_2e-05.warmup_ratio_0.0.weight_decay_0.01.grad_acc_1.epochs_10"
        )

        all_code_predictions_dict = dict()
        
        prediction_models_dirs = []
        if get_predicted_therapist_codes:
            prediction_models_dirs.append(therapist_code_prediction_model_dir)
        if get_predicted_client_codes:
            prediction_models_dirs.append(client_code_prediction_model_dir)
        
        for prediction_model_dir in prediction_models_dirs:
            code_prediction_results_path = os.path.join(
                prediction_model_dir, "cross_validation_test_results.csv"
            )
            if not os.path.isfile(code_prediction_results_path):
                raise FileNotFoundError(
                    f"""
                    Could not find the file with current-turn therapist code
                    prediction results, namely
                    {code_prediction_results_path}

                    It's probably because you haven't trained the model (cross validation),
                    so do that first and I'll be able to read the predictions
                    """
                )
            code_prediction_results = pd.read_csv(code_prediction_results_path)
            code_predictions_dict = dict()
            for row in code_prediction_results.itertuples():
                code_predictions_dict[getattr(row, "anno_mi_row_id")] = {
                    "ground_truth_code": getattr(row, "ground_truth_code"),
                    "predicted_code": getattr(row, "predicted_code"),
                }
            all_code_predictions_dict = {
                **all_code_predictions_dict,
                **code_predictions_dict
            }
        return all_code_predictions_dict
    else:
        raise NotImplementedError(
            f"""
            Haven't implemented code prediction model for 
            {mi_quality_filter}-quality therapist utterances
            """
        )


def build_context_response_pair_datasets_with_clean_separation_of_dialogues(
    anno_mi_data_path: str=ANNO_MI_NORMALIZED_PATH,
    mi_quality: str=None,
    conv_history_window: int=3,
    num_fold: int=10,
    # Basically, this means how much data from a fold we use
    # as the validation data of a particular split
    # Should be <= 1
    dev_data_to_fold_size_ratio: float=1.0,
    random_seed: int=42,
    response_interlocutor: str=None, # "therapist" or "client"
    prepend_codes_in_context: str=None,
    prepend_code_in_response: bool=False,
    overwrite: bool=False,
):
    rand = random.Random(random_seed)
    assert prepend_codes_in_context in [
        "no", "therapist_oracle", "therapist_predicted",
        "client_oracle", "therapist_and_client_oracle",
        "client_predicted", "therapist_and_client_predicted",
        "therapist_random", "client_random", 
        "therapist_and_client_random"
    ]

    context_response_pairs_output_dir = os.path.join(
        ANNO_MI_DATA_DIR,
        (
            "anno_mi.{}_quality".format(
                mi_quality if mi_quality in ["high", "low"] else "high_and_low"
            ) + 
            ".context_response_pairs" + 
            (
                ".{}_codes_in_context".format(prepend_codes_in_context)
            ) +
            (
                ".code_in_response" if prepend_code_in_response else ".no_code_in_response"
            ) +            
            f".context_window_{conv_history_window}.clean_separation_of_dialogues"
        )
    )
    if response_interlocutor is not None:
        assert (
            response_interlocutor in ["therapist", "client"]
        ), f"Unknown interlocutor: {response_interlocutor}"
        context_response_pairs_output_dir += \
            ".response_from_{}_only".format(
                response_interlocutor
            )
    if not os.path.isdir(context_response_pairs_output_dir):
        os.mkdir(context_response_pairs_output_dir)
    context_response_pairs_output_path = os.path.join(
        context_response_pairs_output_dir,
        "original.csv"
    )

    if os.path.isfile(context_response_pairs_output_path) and not overwrite:
        context_response_pairs_df = pd.read_csv(
            context_response_pairs_output_path,
            keep_default_na=False
        )
    else:
        context_response_pairs_df = build_annomi_context_and_response_pairs_df(
            anno_mi_data_path=anno_mi_data_path,
            conv_history_window=conv_history_window, 
            prepend_codes_in_context=prepend_codes_in_context,
            prepend_code_in_response=prepend_code_in_response,
            context_utterances_connector="|",
            mi_quality_filter=mi_quality,
            return_response_codes=True,
            random_seed=random_seed
        )
        if response_interlocutor is not None:
            context_response_pairs_df = context_response_pairs_df.loc[
                context_response_pairs_df["response"].str.startswith(
                    f"<{response_interlocutor}>"
                )
            ]
        context_response_pairs_df = context_response_pairs_df.reset_index(drop=True)
        context_response_pairs_df.to_csv(
            context_response_pairs_output_path,
            index=False
        )

    # If all the dialogues are of high or low quality, we can simply
    # get the folds in one go, considering only the size of each dialogue
    if len(set(context_response_pairs_df["mi_quality"])) == 1:
        all_folds = get_similarly_sized_folds(
            context_response_pairs_df=context_response_pairs_df,
            num_fold=num_fold
        )
    # Otherwise, we need to divide high- and low-quality dialogues
    # into K folds separately and then combine the each high- and low-quality
    # fold into a high & low fold.
    # This is to ensure that both high- and low-quality dialogues are
    # evenly distributed across all the folds
    else:
        high_quality_context_response_pairs_df = context_response_pairs_df.\
            loc[context_response_pairs_df["mi_quality"] == "high"].\
                reset_index(drop=True)
        high_quality_all_folds = get_similarly_sized_folds(
            context_response_pairs_df=high_quality_context_response_pairs_df,
            num_fold=num_fold
        )
        low_quality_context_response_pairs_df = context_response_pairs_df.\
            loc[context_response_pairs_df["mi_quality"] == "low"].\
                reset_index(drop=True)
        low_quality_all_folds = get_similarly_sized_folds(
            context_response_pairs_df=low_quality_context_response_pairs_df,
            num_fold=num_fold
        )

        # We merge a high-quality fold with a low-quality fold based on this principle:
        # The larger/smaller the high-quality fold is, the smaller/larger the low-quality fold should be
        # Hence, the merged folds should have similar sizes
        get_fold_size = lambda fold: sum([dialogue["utt_count"] for dialogue in fold])
        all_folds = []
        for high_quality_fold, low_quality_fold in zip(
            sorted(high_quality_all_folds, key=(lambda fold: get_fold_size(fold)), reverse=True),
            sorted(low_quality_all_folds, key=(lambda fold: get_fold_size(fold)), reverse=False), 
        ):
            all_folds.append(high_quality_fold + low_quality_fold)


    def get_k_fold_splits(all_folds: List[List[Dict[str, Any]]]):
        for test_fold_id in range(len(all_folds)):

            partial_dev_fold_id = test_fold_id + 1
            if partial_dev_fold_id >= len(all_folds):
                partial_dev_fold_id = 0

            partial_dev_fold_high_quality_dialogues = [
                dialogue for dialogue in all_folds[partial_dev_fold_id] \
                    if dialogue["mi_quality"] == "high"
            ]
            logger.info(
                f"""
                Sampling from the high-quality dialogues in
                split {partial_dev_fold_id} to create 
                validation data ...
                """
            )        
            dev_dialogues_high_quality, non_dev_dialogues_high_quality = \
                sample_dev_data_from_fold(
                    fold=partial_dev_fold_high_quality_dialogues,
                    dev_data_to_fold_size_ratio=dev_data_to_fold_size_ratio
                )

            partial_dev_fold_low_quality_dialogues = [
                dialogue for dialogue in \
                    all_folds[partial_dev_fold_id] \
                        if dialogue["mi_quality"] == "low"
            ]
            logger.info(
                f"""
                Sampling from the low-quality dialogues in
                split {partial_dev_fold_id} to create 
                validation data ...
                """
            )                
            dev_dialogues_low_quality, non_dev_dialogues_low_quality = \
                sample_dev_data_from_fold(
                    fold=partial_dev_fold_low_quality_dialogues,
                    dev_data_to_fold_size_ratio=dev_data_to_fold_size_ratio
                )

            dev_dialogues_in_partial_dev_fold = dev_dialogues_high_quality + dev_dialogues_low_quality
            non_dev_dialogues_in_partial_dev_fold = non_dev_dialogues_high_quality + non_dev_dialogues_low_quality

            train_folds_ids = [
                fold_id for fold_id in range(len(all_folds)) \
                    if fold_id not in [partial_dev_fold_id, test_fold_id]
            ]

            test_data_indexes = context_response_pairs_df.loc[
                context_response_pairs_df["dialogue_id"].isin([
                    utt["dialogue_id"] for utt in all_folds[test_fold_id]
                ])
            ].index.tolist()
            dev_data_indexes = context_response_pairs_df.loc[
                context_response_pairs_df["dialogue_id"].isin([
                    utt["dialogue_id"] for utt in dev_dialogues_in_partial_dev_fold
                ])
            ].index.tolist()    
            train_data_indexes = context_response_pairs_df.loc[
                context_response_pairs_df["dialogue_id"].isin(
                    [
                        utt["dialogue_id"] for utt in itertools.chain(*[
                            all_folds[fold_id] for fold_id in train_folds_ids
                        ])
                    ] +
                    [
                        utt["dialogue_id"] for utt in non_dev_dialogues_in_partial_dev_fold
                    ]                  
                )
            ].index.tolist()

            rand.shuffle(test_data_indexes)
            rand.shuffle(dev_data_indexes)
            rand.shuffle(train_data_indexes)

            yield train_data_indexes, dev_data_indexes, test_data_indexes


    # k_fold_data_dirs = dict()
    k_fold_splits_dir = os.path.join(
        context_response_pairs_output_dir,
        f"{num_fold}_fold_splits"
    )
    k_fold_data_creation_is_complete_flag_path = os.path.join(
        k_fold_splits_dir, "k_fold_data_creation_is_complete.flag"
    )
    if os.path.isfile(k_fold_data_creation_is_complete_flag_path) and not overwrite:
        return k_fold_splits_dir

    logger.info(f"Creating {num_fold} splits ...")
    if not os.path.isdir(k_fold_splits_dir):
        os.mkdir(k_fold_splits_dir)

    for split_id, (train_indexes, dev_indexes, test_indexes) in enumerate(
        get_k_fold_splits(all_folds=all_folds)
    ):
        train_data =  context_response_pairs_df.iloc[train_indexes]
        dev_data =  context_response_pairs_df.iloc[dev_indexes]
        test_data =  context_response_pairs_df.iloc[test_indexes]

        split_data_dir = os.path.join(
            k_fold_splits_dir,
            f"split_{split_id}"
        )
        if not os.path.isdir(split_data_dir):
            os.mkdir(split_data_dir)
        for subset_id, subset_data in zip(
            ["train", "dev", "test"],
            [train_data, dev_data, test_data]
        ):
            subset_output_path = os.path.join(
                split_data_dir,
                f"{subset_id}.csv"
            )
            subset_data.to_csv(subset_output_path, index=False)

    with open(k_fold_data_creation_is_complete_flag_path, 'w') as k_fold_data_creation_is_complete_flag_writer:
        k_fold_data_creation_is_complete_flag_writer.write("\n")    

    return k_fold_splits_dir


def build_mi_quality_balanced_context_response_pair_datasets_with_clean_separation_of_dialogues(
    anno_mi_data_path: str=ANNO_MI_NORMALIZED_PATH,
    anno_mi_augmented_data_path: str=ANNO_MI_NORMALIZED_AUGMENTED_PATH,
    mi_quality: str=None,
    conv_history_window: int=3,
    num_fold: int=10,
    # Basically, this means how much data from a fold we use
    # as the validation data of a particular split
    # Should be <= 1
    dev_data_to_fold_size_ratio: float=1.0,
    random_seed: int=42,
    response_interlocutor: str=None, # "therapist" or "client"
    prepend_codes_in_context: str=None,
    prepend_code_in_response: bool=False,
    overwrite: bool=False,
    mi_quality_balancing_method: str="augmentation"
):
    assert mi_quality is None
    assert prepend_codes_in_context == "no"
    if not mi_quality_balancing_method == "augmentation":
        raise NotImplementedError(
            f"""
            We haven't implemented the "{mi_quality_balancing_method}"
            method of MI quality balancing
            """
        )

    rand = random.Random(random_seed)
    context_response_pairs_output_dir = os.path.join(
        ANNO_MI_DATA_DIR,
        (
            "anno_mi.high_and_low_quality_{}_balanced".format(
                mi_quality_balancing_method
            ) + 
            ".context_response_pairs" + 
            (
                ".{}_codes_in_context".format(prepend_codes_in_context)
            ) +
            (
                ".code_in_response" if prepend_code_in_response else ".no_code_in_response"
            ) +            
            f".context_window_{conv_history_window}.clean_separation_of_dialogues"
        )
    )
    if response_interlocutor is not None:
        assert (
            response_interlocutor in ["therapist", "client"]
        ), f"Unknown interlocutor: {response_interlocutor}"
        context_response_pairs_output_dir += \
            ".response_from_{}_only".format(
                response_interlocutor
            )
    if not os.path.isdir(context_response_pairs_output_dir):
        os.mkdir(context_response_pairs_output_dir)
    original_context_response_pairs_output_path = os.path.join(
        context_response_pairs_output_dir,
        "original.csv"
    )
    augmented_context_response_pairs_output_path = os.path.join(
        context_response_pairs_output_dir,
        "augmented.csv"
    )    

    if not (
        os.path.isfile(original_context_response_pairs_output_path) and
        os.path.isfile(augmented_context_response_pairs_output_path) and 
        not overwrite
    ):
        original_context_response_pairs_df = build_annomi_context_and_response_pairs_df(
            anno_mi_data_path=anno_mi_data_path,
            conv_history_window=conv_history_window, 
            prepend_codes_in_context=prepend_codes_in_context,
            prepend_code_in_response=prepend_code_in_response,
            context_utterances_connector="|",
            mi_quality_filter=mi_quality,
            return_response_codes=True,
            random_seed=random_seed
        )
        augmented_context_response_pairs_df = build_augmented_annomi_context_and_response_pairs_df(
            anno_mi_augmented_data_path=anno_mi_augmented_data_path,
            conv_history_window=conv_history_window, 
            prepend_codes_in_context=prepend_codes_in_context,
            prepend_code_in_response=prepend_code_in_response,
            context_utterances_connector="|",
            mi_quality_filter=mi_quality,
            return_response_codes=True,
            # because #utt(high) is roughly 10 times #utt(low)
            num_augs_to_use=10,
            random_seed=random_seed
        )        
        for original_context_response_pairs_df, context_response_pairs_output_path in zip(
            [
                original_context_response_pairs_df, 
                augmented_context_response_pairs_df
            ],
            [
                original_context_response_pairs_output_path, 
                augmented_context_response_pairs_output_path
            ]
        ):
            if response_interlocutor is not None:
                original_context_response_pairs_df = original_context_response_pairs_df.loc[
                    original_context_response_pairs_df["response"].str.startswith(
                        f"<{response_interlocutor}>"
                    )
                ]
            original_context_response_pairs_df = original_context_response_pairs_df.reset_index(drop=True)
            original_context_response_pairs_df.to_csv(context_response_pairs_output_path, index=False)

    original_context_response_pairs_df = pd.read_csv(
        original_context_response_pairs_output_path,
        keep_default_na=False
    )
    augmented_context_response_pairs_df = pd.read_csv(
        augmented_context_response_pairs_output_path,
        keep_default_na=False
    )         

    # If all the dialogues are of high or low quality, we can simply
    # get the folds in one go, considering only the size of each dialogue
    if len(set(original_context_response_pairs_df["mi_quality"])) == 1:
        all_folds = get_similarly_sized_folds(
            context_response_pairs_df=original_context_response_pairs_df,
            num_fold=num_fold
        )
    # Otherwise, we need to divide high- and low-quality dialogues
    # into K folds separately and then combine the each high- and low-quality
    # fold into a high & low fold.
    # This is to ensure that both high- and low-quality dialogues are
    # evenly distributed across all the folds
    else:
        high_quality_context_response_pairs_df = original_context_response_pairs_df.\
            loc[original_context_response_pairs_df["mi_quality"] == "high"].\
                reset_index(drop=True)
        high_quality_all_folds = get_similarly_sized_folds(
            context_response_pairs_df=high_quality_context_response_pairs_df,
            num_fold=num_fold
        )
        low_quality_context_response_pairs_df = original_context_response_pairs_df.\
            loc[original_context_response_pairs_df["mi_quality"] == "low"].\
                reset_index(drop=True)
        low_quality_all_folds = get_similarly_sized_folds(
            context_response_pairs_df=low_quality_context_response_pairs_df,
            num_fold=num_fold
        )

        # We merge a high-quality fold with a low-quality fold based on this principle:
        # The larger/smaller the high-quality fold is, the smaller/larger the low-quality fold should be
        # Hence, the merged folds should have similar sizes
        get_fold_size = lambda fold: sum([dialogue["utt_count"] for dialogue in fold])
        all_folds = []
        for high_quality_fold, low_quality_fold in zip(
            sorted(high_quality_all_folds, key=(lambda fold: get_fold_size(fold)), reverse=True),
            sorted(low_quality_all_folds, key=(lambda fold: get_fold_size(fold)), reverse=False), 
        ):
            all_folds.append(high_quality_fold + low_quality_fold)


    def get_k_fold_splits(all_folds: List[List[Dict[str, Any]]]):
        for test_fold_id in range(len(all_folds)):

            partial_dev_fold_id = test_fold_id + 1
            if partial_dev_fold_id >= len(all_folds):
                partial_dev_fold_id = 0

            partial_dev_fold_high_quality_dialogues = [
                dialogue for dialogue in all_folds[partial_dev_fold_id] \
                    if dialogue["mi_quality"] == "high"
            ]
            logger.info(
                f"""
                Sampling from the high-quality dialogues in
                split {partial_dev_fold_id} to create 
                validation data ...
                """
            )        
            dev_dialogues_high_quality, non_dev_dialogues_high_quality = \
                sample_dev_data_from_fold(
                    fold=partial_dev_fold_high_quality_dialogues,
                    dev_data_to_fold_size_ratio=dev_data_to_fold_size_ratio
                )

            partial_dev_fold_low_quality_dialogues = [
                dialogue for dialogue in \
                    all_folds[partial_dev_fold_id] \
                        if dialogue["mi_quality"] == "low"
            ]
            logger.info(
                f"""
                Sampling from the low-quality dialogues in
                split {partial_dev_fold_id} to create 
                validation data ...
                """
            )                
            dev_dialogues_low_quality, non_dev_dialogues_low_quality = \
                sample_dev_data_from_fold(
                    fold=partial_dev_fold_low_quality_dialogues,
                    dev_data_to_fold_size_ratio=dev_data_to_fold_size_ratio
                )

            dev_dialogues_in_partial_dev_fold = dev_dialogues_high_quality + dev_dialogues_low_quality
            non_dev_dialogues_in_partial_dev_fold = non_dev_dialogues_high_quality + non_dev_dialogues_low_quality

            train_folds_ids = [
                fold_id for fold_id in range(len(all_folds)) \
                    if fold_id not in [partial_dev_fold_id, test_fold_id]
            ]

            test_data_indexes = original_context_response_pairs_df.loc[
                original_context_response_pairs_df["dialogue_id"].isin([
                    utt["dialogue_id"] for utt in all_folds[test_fold_id]
                ])
            ].index.tolist()
            dev_data_indexes = original_context_response_pairs_df.loc[
                original_context_response_pairs_df["dialogue_id"].isin([
                    utt["dialogue_id"] for utt in dev_dialogues_in_partial_dev_fold
                ])
            ].index.tolist()    

            train_original_data_indexes = original_context_response_pairs_df.loc[
                original_context_response_pairs_df["dialogue_id"].isin(
                    [
                        utt["dialogue_id"] for utt in itertools.chain(*[
                            all_folds[fold_id] for fold_id in train_folds_ids
                        ])
                    ] +
                    [
                        utt["dialogue_id"] for utt in non_dev_dialogues_in_partial_dev_fold
                    ]                  
                )
            ].index.tolist()

            train_augmented_data_indexes = augmented_context_response_pairs_df.loc[
                augmented_context_response_pairs_df["dialogue_id"].isin(
                    [
                        utt["dialogue_id"] for utt in itertools.chain(*[
                            all_folds[fold_id] for fold_id in train_folds_ids
                        ])
                    ] +
                    [
                        utt["dialogue_id"] for utt in non_dev_dialogues_in_partial_dev_fold
                    ]                  
                )
            ].index.tolist()

            rand.shuffle(test_data_indexes)
            rand.shuffle(dev_data_indexes)
            rand.shuffle(train_original_data_indexes)
            rand.shuffle(train_augmented_data_indexes)

            yield (
                train_original_data_indexes, 
                train_augmented_data_indexes, 
                dev_data_indexes, 
                test_data_indexes
            )


    # k_fold_data_dirs = dict()
    k_fold_splits_dir = os.path.join(
        context_response_pairs_output_dir,
        f"{num_fold}_fold_splits"
    )
    k_fold_data_creation_is_complete_flag_path = os.path.join(
        k_fold_splits_dir, "k_fold_data_creation_is_complete.flag"
    )
    if os.path.isfile(k_fold_data_creation_is_complete_flag_path) and not overwrite:
        return k_fold_splits_dir

    logger.info(f"Creating {num_fold} splits ...")
    if not os.path.isdir(k_fold_splits_dir):
        os.mkdir(k_fold_splits_dir)

    for split_id, (
        train_original_data_indexes, train_augmented_data_indexes, 
        dev_indexes, test_indexes
    ) in enumerate(
        get_k_fold_splits(all_folds=all_folds)
    ):
        train_original_data =  original_context_response_pairs_df\
            .iloc[train_original_data_indexes].copy(deep=True)
        train_original_data["augmented"] = False
        train_augmented_data =  augmented_context_response_pairs_df.\
            iloc[train_augmented_data_indexes]
        # train_data =  original_context_response_pairs_df.iloc[train_indexes]

        high_vs_low_quality_mi_num_examples_gap = (
            train_original_data["mi_quality"].value_counts()["high"] - 
            train_original_data["mi_quality"].value_counts()["low"]
        )
        train_augmented_low_mi_quality_data = train_augmented_data.loc[
            train_augmented_data["mi_quality"] == "low"
        ]

        if (
            train_augmented_low_mi_quality_data.shape[0] > 
            high_vs_low_quality_mi_num_examples_gap
        ):
            gap_filling_train_augmented_low_mi_quality_data = \
                train_augmented_low_mi_quality_data.iloc[
                    :high_vs_low_quality_mi_num_examples_gap
                ].copy(deep=True)
        elif (
            train_augmented_low_mi_quality_data.shape[0] ==
            high_vs_low_quality_mi_num_examples_gap            
        ):
            gap_filling_train_augmented_low_mi_quality_data = \
                train_augmented_low_mi_quality_data.copy(deep=True)
        else:
            # So long as the high-vs-low quality MI utt count gap
            # is not more than twice the number of low-quality-MI
            # augmentations, we just repeat some elements of the
            # augmentations to fill the rest of the gap
            assert (
                train_augmented_low_mi_quality_data.shape[0] >
                0.5 * high_vs_low_quality_mi_num_examples_gap
            )
            logger.warning(
                """
                The total number of augmentations for low-quality MI
                is inadequate but not terribly so (>50%), so we're gonna
                repeat some repetitions to fill the high-low quality MI
                utterance count gap
                """
            )
            gap_filling_train_augmented_low_mi_quality_data = pd.concat(
                [gap_filling_train_augmented_low_mi_quality_data] * 2,
            )
            gap_filling_train_augmented_low_mi_quality_data = \
                gap_filling_train_augmented_low_mi_quality_data.iloc[
                    :high_vs_low_quality_mi_num_examples_gap
                ].copy(deep=True)

        gap_filling_train_augmented_low_mi_quality_data["augmented"] = True

        train_data = pd.concat(
            [
                train_original_data,
                gap_filling_train_augmented_low_mi_quality_data
            ],
            ignore_index=True
        ).sample(frac=1)
        assert (
            train_data["mi_quality"].value_counts()["high"] == 
            train_data["mi_quality"].value_counts()["low"]
        )

        dev_data = original_context_response_pairs_df.iloc[dev_indexes].copy(deep=True)
        dev_data["augmented"] = False

        test_data = original_context_response_pairs_df.iloc[test_indexes].copy(deep=True)
        test_data["augmented"] = False

        split_data_dir = os.path.join(
            k_fold_splits_dir,
            f"split_{split_id}"
        )
        if not os.path.isdir(split_data_dir):
            os.mkdir(split_data_dir)
        for subset_id, subset_data in zip(
            ["train", "dev", "test"],
            [train_data, dev_data, test_data]
        ):
            subset_output_path = os.path.join(
                split_data_dir,
                f"{subset_id}.csv"
            )
            subset_data.to_csv(subset_output_path, index=False)

    with open(k_fold_data_creation_is_complete_flag_path, 'w') as k_fold_data_creation_is_complete_flag_writer:
        k_fold_data_creation_is_complete_flag_writer.write("\n")    

    return k_fold_splits_dir



def build_train_augmented_context_response_pair_datasets_with_clean_separation_of_dialogues(
    anno_mi_data_path: str=ANNO_MI_NORMALIZED_PATH,
    anno_mi_augmented_data_path: str=ANNO_MI_NORMALIZED_AUGMENTED_PATH,
    mi_quality: str=None,
    conv_history_window: int=3,
    num_fold: int=10,
    dev_data_to_fold_size_ratio: float=1.0,
    random_seed: int=42,
    response_interlocutor: str=None, # "therapist" or "client"
    prepend_codes_in_context: str=None,
    prepend_code_in_response: bool=False,
    num_augs_to_use_per_train_example: int=5,
    overwrite: bool=False,
):
    assert prepend_codes_in_context == "no"
    rand = random.Random(random_seed)
    context_response_pairs_output_dir = os.path.join(
        ANNO_MI_DATA_DIR,
        (
            "anno_mi.train_augmented" +
            ".{}_augs_per_train_example".format(num_augs_to_use_per_train_example) +
            ".{}_quality".format(
                mi_quality if mi_quality in ["high", "low"] else "high_and_low"
            ) + 
            ".context_response_pairs" + 
            (
                ".{}_codes_in_context".format(prepend_codes_in_context)
            ) +
            (
                ".code_in_response" if prepend_code_in_response else ".no_code_in_response"
            ) +            
            f".context_window_{conv_history_window}.clean_separation_of_dialogues"
        )
    )
    if response_interlocutor is not None:
        assert (
            response_interlocutor in ["therapist", "client"]
        ), f"Unknown interlocutor: {response_interlocutor}"
        context_response_pairs_output_dir += \
            ".response_from_{}_only".format(
                response_interlocutor
            )
    if not os.path.isdir(context_response_pairs_output_dir):
        os.mkdir(context_response_pairs_output_dir)
    original_context_response_pairs_output_path = os.path.join(
        context_response_pairs_output_dir,
        "original.csv"
    )
    augmented_context_response_pairs_output_path = os.path.join(
        context_response_pairs_output_dir,
        "augmented.csv"
    )    

    if not (
        os.path.isfile(original_context_response_pairs_output_path) and
        os.path.isfile(augmented_context_response_pairs_output_path) and 
        not overwrite
    ):
        original_context_response_pairs_df = build_annomi_context_and_response_pairs_df(
            anno_mi_data_path=anno_mi_data_path,
            conv_history_window=conv_history_window, 
            prepend_codes_in_context=prepend_codes_in_context,
            prepend_code_in_response=prepend_code_in_response,
            context_utterances_connector="|",
            mi_quality_filter=mi_quality,
            return_response_codes=True,
            random_seed=random_seed
        )
        augmented_context_response_pairs_df = build_augmented_annomi_context_and_response_pairs_df(
            anno_mi_augmented_data_path=anno_mi_augmented_data_path,
            conv_history_window=conv_history_window, 
            prepend_codes_in_context=prepend_codes_in_context,
            prepend_code_in_response=prepend_code_in_response,
            context_utterances_connector="|",
            mi_quality_filter=mi_quality,
            return_response_codes=True,
            num_augs_to_use=num_augs_to_use_per_train_example,
            random_seed=random_seed
        )
        for context_response_pairs_df, context_response_pairs_output_path in zip(
            [
                original_context_response_pairs_df, 
                augmented_context_response_pairs_df
            ],
            [
                original_context_response_pairs_output_path, 
                augmented_context_response_pairs_output_path
            ]
        ):
            if response_interlocutor is not None:
                context_response_pairs_df = context_response_pairs_df.loc[
                    context_response_pairs_df["response"].str.startswith(
                        f"<{response_interlocutor}>"
                    )
                ]
            context_response_pairs_df = context_response_pairs_df.reset_index(drop=True)
            context_response_pairs_df.to_csv(context_response_pairs_output_path, index=False)

    original_context_response_pairs_df = pd.read_csv(
        original_context_response_pairs_output_path,
        keep_default_na=False
    )
    augmented_context_response_pairs_df = pd.read_csv(
        augmented_context_response_pairs_output_path,
        keep_default_na=False
    )              

    # If all the dialogues are of high or low quality, we can simply
    # get the folds in one go, considering only the size of each dialogue
    if len(set(original_context_response_pairs_df["mi_quality"])) == 1:
        all_folds = get_similarly_sized_folds(
            context_response_pairs_df=original_context_response_pairs_df,
            num_fold=num_fold
        )
    # Otherwise, we need to divide high- and low-quality dialogues
    # into K folds separately and then combine the each high- and low-quality
    # fold into a high & low fold.
    # This is to ensure that both high- and low-quality dialogues are
    # evenly distributed across all the folds
    else:
        high_quality_context_response_pairs_df = original_context_response_pairs_df.\
            loc[original_context_response_pairs_df["mi_quality"] == "high"].\
                reset_index(drop=True)
        high_quality_all_folds = get_similarly_sized_folds(
            context_response_pairs_df=high_quality_context_response_pairs_df,
            num_fold=num_fold
        )
        low_quality_context_response_pairs_df = original_context_response_pairs_df.\
            loc[original_context_response_pairs_df["mi_quality"] == "low"].\
                reset_index(drop=True)
        low_quality_all_folds = get_similarly_sized_folds(
            context_response_pairs_df=low_quality_context_response_pairs_df,
            num_fold=num_fold
        )

        # We merge a high-quality fold with a low-quality fold based on this principle:
        # The larger/smaller the high-quality fold is, the smaller/larger the low-quality fold should be
        # Hence, the merged folds should have similar sizes
        get_fold_size = lambda fold: sum([dialogue["utt_count"] for dialogue in fold])
        all_folds = []
        for high_quality_fold, low_quality_fold in zip(
            sorted(high_quality_all_folds, key=(lambda fold: get_fold_size(fold)), reverse=True),
            sorted(low_quality_all_folds, key=(lambda fold: get_fold_size(fold)), reverse=False), 
        ):
            all_folds.append(high_quality_fold + low_quality_fold)

    def get_k_fold_splits(all_folds: List[List[Dict[str, Any]]]):
        for test_fold_id in range(len(all_folds)):

            partial_dev_fold_id = test_fold_id + 1
            if partial_dev_fold_id >= len(all_folds):
                partial_dev_fold_id = 0

            partial_dev_fold_high_quality_dialogues = [
                dialogue for dialogue in all_folds[partial_dev_fold_id] \
                    if dialogue["mi_quality"] == "high"
            ]
            logger.info(
                f"""
                Sampling from the high-quality dialogues in
                split {partial_dev_fold_id} to create 
                validation data ...
                """
            )        
            dev_dialogues_high_quality, non_dev_dialogues_high_quality = \
                sample_dev_data_from_fold(
                    fold=partial_dev_fold_high_quality_dialogues,
                    dev_data_to_fold_size_ratio=dev_data_to_fold_size_ratio
                )

            partial_dev_fold_low_quality_dialogues = [
                dialogue for dialogue in \
                    all_folds[partial_dev_fold_id] \
                        if dialogue["mi_quality"] == "low"
            ]
            logger.info(
                f"""
                Sampling from the low-quality dialogues in
                split {partial_dev_fold_id} to create 
                validation data ...
                """
            )                
            dev_dialogues_low_quality, non_dev_dialogues_low_quality = \
                sample_dev_data_from_fold(
                    fold=partial_dev_fold_low_quality_dialogues,
                    dev_data_to_fold_size_ratio=dev_data_to_fold_size_ratio
                )

            dev_dialogues_in_partial_dev_fold = dev_dialogues_high_quality + dev_dialogues_low_quality
            non_dev_dialogues_in_partial_dev_fold = non_dev_dialogues_high_quality + non_dev_dialogues_low_quality

            train_folds_ids = [
                fold_id for fold_id in range(len(all_folds)) \
                    if fold_id not in [partial_dev_fold_id, test_fold_id]
            ]

            # Use augmented data to create the training set and
            # real data to create the dev & test sets
            test_data_indexes = original_context_response_pairs_df.loc[
                original_context_response_pairs_df["dialogue_id"].isin([
                    utt["dialogue_id"] for utt in all_folds[test_fold_id]
                ])
            ].index.tolist()
            dev_data_indexes = original_context_response_pairs_df.loc[
                original_context_response_pairs_df["dialogue_id"].isin([
                    utt["dialogue_id"] for utt in dev_dialogues_in_partial_dev_fold
                ])
            ].index.tolist()
            train_data_indexes = augmented_context_response_pairs_df.loc[
                augmented_context_response_pairs_df["dialogue_id"].isin(
                    [
                        utt["dialogue_id"] for utt in itertools.chain(*[
                            all_folds[fold_id] for fold_id in train_folds_ids
                        ])
                    ] +
                    [
                        utt["dialogue_id"] for utt in non_dev_dialogues_in_partial_dev_fold
                    ]
                )
            ].index.tolist()

            rand.shuffle(test_data_indexes)
            rand.shuffle(dev_data_indexes)
            rand.shuffle(train_data_indexes)

            yield train_data_indexes, dev_data_indexes, test_data_indexes


    # k_fold_data_dirs = dict()
    k_fold_splits_dir = os.path.join(
        context_response_pairs_output_dir,
        f"{num_fold}_fold_splits"
    )
    k_fold_data_creation_is_complete_flag_path = os.path.join(
        k_fold_splits_dir, "k_fold_data_creation_is_complete.flag"
    )    
    if os.path.isfile(k_fold_data_creation_is_complete_flag_path) and not overwrite:
        return k_fold_splits_dir

    logger.info(f"Creating {num_fold} splits with augmented data ...")
    if not os.path.isdir(k_fold_splits_dir):
        os.mkdir(k_fold_splits_dir)

    for split_id, (train_indexes, dev_indexes, test_indexes) in enumerate(
        get_k_fold_splits(all_folds=all_folds)
    ):
        # Use augmented data to create the training set and
        # real data to create the dev & test sets
        train_data =  augmented_context_response_pairs_df.iloc[train_indexes]
        dev_data =  original_context_response_pairs_df.iloc[dev_indexes]
        test_data =  original_context_response_pairs_df.iloc[test_indexes]

        split_data_dir = os.path.join(
            k_fold_splits_dir,
            f"split_{split_id}"
        )
        if not os.path.isdir(split_data_dir):
            os.mkdir(split_data_dir)
        for subset_id, subset_data in zip(
            ["train", "dev", "test"],
            [train_data, dev_data, test_data]
        ):
            subset_output_path = os.path.join(
                split_data_dir,
                f"{subset_id}.csv"
            )
            subset_data.to_csv(subset_output_path, index=False)

    with open(k_fold_data_creation_is_complete_flag_path, 'w') as k_fold_data_creation_is_complete_flag_writer:
        k_fold_data_creation_is_complete_flag_writer.write("\n")

    return k_fold_splits_dir


def load_context_response_pair_datasets_with_augmented_train_data(
    anno_mi_data_path: str=ANNO_MI_NORMALIZED_PATH,
    anno_mi_augmented_data_path: str=ANNO_MI_NORMALIZED_AUGMENTED_PATH,
    mi_quality: str=None,
    conv_history_window: int=3,
    num_fold: int=10,
    random_seed: int=42,
    return_data_name: bool=False,
    response_interlocutor: str=None, # "therapist" or "client"
    prepend_codes_in_context: str=None,
    prepend_code_in_response: bool=False,   
    num_augs_to_use_per_train_example: int=5,  
    overwrite: bool=False,
    debug: bool=False,
    dev_data_to_fold_size_ratio: float=1.0
):
    k_fold_splits_dir = \
        build_train_augmented_context_response_pair_datasets_with_clean_separation_of_dialogues(
            anno_mi_data_path=anno_mi_data_path,
            anno_mi_augmented_data_path=anno_mi_augmented_data_path,
            mi_quality=mi_quality,
            conv_history_window=conv_history_window,
            num_fold=num_fold,
            random_seed=random_seed,
            response_interlocutor=response_interlocutor,
            prepend_codes_in_context=prepend_codes_in_context,
            prepend_code_in_response=prepend_code_in_response,
            num_augs_to_use_per_train_example=num_augs_to_use_per_train_example,
            overwrite=overwrite,
            dev_data_to_fold_size_ratio=dev_data_to_fold_size_ratio
        )

    data_dir, k_fold_name = os.path.split(k_fold_splits_dir)
    _, data_type = os.path.split(data_dir)
    data_name = f"{data_type}.{k_fold_name}"

    logger.info(
        f"""
        Getting the folder for each {num_fold}-split
        """
    )

    split_dataset_dicts = dict()
    for split_id in range((1 if debug else num_fold)):
        split_dir = os.path.join(k_fold_splits_dir, f"split_{split_id}")
        split_dataset_dict = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(split_dir, "train.csv"),
                "dev": os.path.join(split_dir, "dev.csv"),
                "test": os.path.join(split_dir, "test.csv"),
            },
            cache_dir=HF_DATASETS_CACHE_DIR
        )
        split_dataset_dicts[split_id] = split_dataset_dict

    if return_data_name:
        return split_dataset_dicts, data_name
    else:
        return split_dataset_dicts


def load_context_response_pair_datasets(
    anno_mi_data_path: str=ANNO_MI_NORMALIZED_PATH,
    mi_quality: str=None,
    conv_history_window: int=3,
    num_fold: int=10,
    random_seed: int=42,
    return_data_name: bool=False,
    response_interlocutor: str=None, # "therapist" or "client"
    prepend_codes_in_context: str=None,
    prepend_code_in_response: bool=False,     
    overwrite: bool=False,
    debug: bool=False,
    dev_data_to_fold_size_ratio: float=1.0,
):
    k_fold_splits_dir = build_context_response_pair_datasets_with_clean_separation_of_dialogues(
        anno_mi_data_path=anno_mi_data_path,
        mi_quality=mi_quality,
        conv_history_window=conv_history_window,
        num_fold=num_fold,
        random_seed=random_seed,
        response_interlocutor=response_interlocutor,
        prepend_codes_in_context=prepend_codes_in_context,
        prepend_code_in_response=prepend_code_in_response,        
        overwrite=overwrite,
        dev_data_to_fold_size_ratio=dev_data_to_fold_size_ratio
    )

    data_dir, k_fold_name = os.path.split(k_fold_splits_dir)
    _, data_type = os.path.split(data_dir)
    data_name = f"{data_type}.{k_fold_name}"

    logger.info(
        f"""
        Getting the folder for each {num_fold}-split
        """
    )

    split_dataset_dicts = dict()
    for split_id in range((1 if debug else num_fold)):
        split_dir = os.path.join(k_fold_splits_dir, f"split_{split_id}")
        split_dataset_dict = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(split_dir, "train.csv"),
                "dev": os.path.join(split_dir, "dev.csv"),
                "test": os.path.join(split_dir, "test.csv"),
            },
            cache_dir=HF_DATASETS_CACHE_DIR
        )
        split_dataset_dicts[split_id] = split_dataset_dict

    if return_data_name:
        return split_dataset_dicts, data_name
    else:
        return split_dataset_dicts


def load_mi_quality_balanced_context_response_pair_datasets(
    anno_mi_data_path: str=ANNO_MI_NORMALIZED_PATH,
    mi_quality: str=None,
    conv_history_window: int=3,
    num_fold: int=10,
    random_seed: int=42,
    return_data_name: bool=False,
    response_interlocutor: str=None, # "therapist" or "client"
    prepend_codes_in_context: str=None,
    prepend_code_in_response: bool=False,     
    overwrite: bool=False,
    debug: bool=False,
    dev_data_to_fold_size_ratio: float=1.0,
    mi_quality_balancing_method: str="augmentation",
):
    k_fold_splits_dir = build_mi_quality_balanced_context_response_pair_datasets_with_clean_separation_of_dialogues(
        anno_mi_data_path=anno_mi_data_path,
        mi_quality=mi_quality,
        conv_history_window=conv_history_window,
        num_fold=num_fold,
        random_seed=random_seed,
        response_interlocutor=response_interlocutor,
        prepend_codes_in_context=prepend_codes_in_context,
        prepend_code_in_response=prepend_code_in_response,        
        overwrite=overwrite,
        dev_data_to_fold_size_ratio=dev_data_to_fold_size_ratio,
        mi_quality_balancing_method=mi_quality_balancing_method
    )

    data_dir, k_fold_name = os.path.split(k_fold_splits_dir)
    _, data_type = os.path.split(data_dir)
    data_name = f"{data_type}.{k_fold_name}"

    logger.info(
        f"""
        Getting the folder for each {num_fold}-split
        """
    )

    split_dataset_dicts = dict()
    for split_id in range((1 if debug else num_fold)):
        split_dir = os.path.join(k_fold_splits_dir, f"split_{split_id}")
        split_dataset_dict = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(split_dir, "train.csv"),
                "dev": os.path.join(split_dir, "dev.csv"),
                "test": os.path.join(split_dir, "test.csv"),
            },
            cache_dir=HF_DATASETS_CACHE_DIR
        )
        split_dataset_dicts[split_id] = split_dataset_dict

    if return_data_name:
        return split_dataset_dicts, data_name
    else:
        return split_dataset_dicts


def build_dataset_of_entire_anno_mi_dialogues(
    dialogue_ids_in_splits: Dict[int, Dict[str, List[int]]],
    output_dir: str,
    anno_mi_data_path: str=ANNO_MI_NORMALIZED_PATH,
    overwrite: bool=False,
    debug: bool=False
):
    """
    dialogue_ids_in_folds = {
        0: {
            "train": [1, 2, 5, ...],
            "dev": [0, 6, ...],
            "test": [3, 4, ...],
        },
        1: {
            ...
        }
    }
    """
    logger.info(
        f"""
        Building a dataset of entire AnnoMI dialogues
        """
    )
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    processed_data_already_exists = True
    for split_id in range((1 if debug else len(dialogue_ids_in_splits))):
        split_dir = os.path.join(output_dir, f"split_{split_id}")
        if not os.path.isdir(split_dir):
            os.mkdir(split_dir)        
        for subset_name in ["train", "dev", "test"]:
            processed_data_path = os.path.join(split_dir, f"{subset_name}.csv")
            if not os.path.isfile(processed_data_path):
                processed_data_already_exists = False
                break
            
    if processed_data_already_exists and not overwrite:
        anno_mi_data = None
    else:
        anno_mi_data = pd.read_csv(anno_mi_data_path, keep_default_na=False)

    split_dataset_dicts = dict()
    for split_id in range((1 if debug else len(dialogue_ids_in_splits))):
        split_dir = os.path.join(output_dir, f"split_{split_id}")

        if overwrite or not processed_data_already_exists:
            if not os.path.isdir(split_dir):
                os.mkdir(split_dir)                
            split_dialogue_ids = dialogue_ids_in_splits[split_id]
            for subset_name in ["train", "dev", "test"]:
                processed_data_path = os.path.join(split_dir, f"{subset_name}.csv")
                dialogue_df_dict = {
                    "dialogue_id": [],
                    "dialogue_text_with_interlocutor_labels": []
                }
                subset_dialogue_ids = split_dialogue_ids[subset_name]
                for dialogue_id in subset_dialogue_ids:
                    dialogue_utts_df = anno_mi_data.loc[
                        anno_mi_data["transcript_id"] == dialogue_id
                    ].copy(deep=True)
                    dialogue_utts_df = dialogue_utts_df.sort_values(by=["utterance_id"])
                    dialogue_utts_df["prefixed_utt_text"] = dialogue_utts_df.apply(
                        lambda row: f"<{row['interlocutor']}>{row['utterance_text']}",
                        axis=1
                    )
                    dialogue_text_with_interlocutor_labels = '|'.join(dialogue_utts_df["prefixed_utt_text"])
                    dialogue_df_dict["dialogue_id"].append(dialogue_id)
                    dialogue_df_dict["dialogue_text_with_interlocutor_labels"].append(dialogue_text_with_interlocutor_labels)
                dialogue_df = pd.DataFrame.from_dict(dialogue_df_dict)
                dialogue_df.to_csv(processed_data_path, index=False)

        split_dataset_dict = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(split_dir, "train.csv"),
                "dev": os.path.join(split_dir, "dev.csv"),
                "test": os.path.join(split_dir, "test.csv"),
            },
            cache_dir=HF_DATASETS_CACHE_DIR
        )
        split_dataset_dicts[split_id] = split_dataset_dict        

    return split_dataset_dicts

        
def pad_to_max_seq_len(
    int_seqs: List[List[int]],
    padding_value: int,
    max_seq_len: int=None,
    padding_side: str="right",
    return_tensors: str="pt",
):
    """
    Manually right-pad `int_seqs` to an tensor of shape
    (int_seqs.shape[0], max_seq_len)

    `int_seqs` contains sequences of different lengths
    """
    assert padding_side in ["left", "right"], \
        f"Unknown padding side: {padding_side}"
    
    max_int_seq_len = max([len(int_seq) for int_seq in int_seqs])
    if max_seq_len is None:
        max_seq_len = max_int_seq_len
    else:
        assert max_int_seq_len <= max_seq_len

    if padding_side == "left":
        # If left padding is desired, then we first horizontally 
        # flip each sequence
        # So that we can later pad on the right, before eventually
        # flipping it back, so that effectively we will have padded
        # on the left
        tensor_seqs = [torch.tensor(int_seq[::-1]) for int_seq in int_seqs]
    else:
        tensor_seqs = [torch.tensor(int_seq) for int_seq in int_seqs]
        
    padded_tensor_seqs = pad_sequence(
        sequences=tensor_seqs,
        batch_first=True,
        padding_value=padding_value
    )

    if max_seq_len == max_int_seq_len:
        tensor_seqs_padded_to_max_seq_len = padded_tensor_seqs
    else:
        tensor_seqs_padded_to_max_seq_len = torch.full(
            size=(padded_tensor_seqs.shape[0], max_seq_len),
            fill_value=padding_value, 
            dtype=padded_tensor_seqs.dtype
        )
        tensor_seqs_padded_to_max_seq_len[:, :padded_tensor_seqs.shape[1]] = \
            padded_tensor_seqs

    if padding_side == "left":
        tensor_seqs_padded_to_max_seq_len = torch.fliplr(tensor_seqs_padded_to_max_seq_len)

    if return_tensors == "np":
        return tensor_seqs_padded_to_max_seq_len.detach().cpu().numpy()
    else:
        return tensor_seqs_padded_to_max_seq_len


def sample_dev_data_from_fold(
    fold: List[Dict[str, Any]],
    dev_data_to_fold_size_ratio: float
):
    dev_dialogues = None
    fold_utt_count = sum([
        dialogue["utt_count"] for dialogue in fold
    ])
    if fold_utt_count == 0:
        logger.info(
            "No dialogues in the fold, so we're returning empty samples"
        )
        return [], []
    # If we need the whole fold as validation data,
    # we simply return the fold itself
    # Also if there's only one dialogue in the fold,
    # we just take it
    elif dev_data_to_fold_size_ratio == 1.0:
        logger.info(
            f"""
            dev_data_to_fold_size_ratio = 1.0, so we are
            taking the whole fold as the dev set
            """
        )
        return fold, []
    elif len(fold) == 1:
        logger.info(
            f"""
            There's only one dialogue in the fold, so we are
            taking the whole fold as the dev set
            """
        )            
        return fold, []

    # Try error = 0.01, 0.02, ... 0.04 first,
    # if it doens't work, try 0.05, 0.10, 0.15, ...
    for max_error in [
        *np.arange(0.01, 0.05, 0.01), 
        *np.arange(0.05, 1.0, 0.05)
    ]:
        dev_data_to_fold_size_ratio_lower_bound = max(
            0.0, dev_data_to_fold_size_ratio - max_error
        )
        # This makes sure that we always sample at least
        # one utterance
        dev_data_num_utts_lower_bound = ceil(
            dev_data_to_fold_size_ratio_lower_bound * 
            fold_utt_count
        )
        dev_data_to_fold_size_ratio_upper_bound = min(
            1.0, dev_data_to_fold_size_ratio + max_error
        )
        dev_data_num_utts_upper_bound = ceil(
            dev_data_to_fold_size_ratio_upper_bound * 
            fold_utt_count
        )                
        # Sampling a subset of the dialogues in the fold
        # whose total utt count is roughly the same as
        # the desired dev data size in terms of utt count
        sampling_successful = False
        for _ in range(500):
            shuffled_dialogues = random.sample(fold, len(fold)) 
            sampled_dialogues_for_dev = []
            desired_dev_data_size_reached = False

            for dialogue in shuffled_dialogues:
                sampled_dialogues_for_dev_utt_count = sum([
                    dialogue["utt_count"] for dialogue in sampled_dialogues_for_dev
                ])
                if (
                    sampled_dialogues_for_dev_utt_count >= dev_data_num_utts_lower_bound and
                    sampled_dialogues_for_dev_utt_count <= dev_data_num_utts_upper_bound
                ):
                    desired_dev_data_size_reached = True
                    break
                sampled_dialogues_for_dev.append(dialogue)

            if desired_dev_data_size_reached:
                sampling_successful = True
                break
            
        if sampling_successful:
            dev_dialogues = sampled_dialogues_for_dev
            logger.info(
                f"""
                Dev data sampling was successful with an error of 
                {max_error}
                """
            )
            break
        
    if dev_dialogues is None:
        raise ValueError(
            f"""
            I wasn't able to sample dev dialogues.
            Something went wrong!
            """
        )

    non_dev_dialogues = [dialogue for dialogue in fold if dialogue not in dev_dialogues]
    return dev_dialogues, non_dev_dialogues
    

def get_similarly_sized_folds(
    context_response_pairs_df: pd.DataFrame,
    num_fold: int
):
    dialogue_level_utt_counts = context_response_pairs_df.\
        groupby(["dialogue_id"]).size().reset_index(name='utt_count')
        
    dialogue_mi_qualities = []
    for row in dialogue_level_utt_counts.itertuples():
        dialogue_id = getattr(row, "dialogue_id")
        dialogue_mi_quality = context_response_pairs_df.loc[
            context_response_pairs_df["dialogue_id"] == dialogue_id
        ]["mi_quality"].unique()
        assert len(dialogue_mi_quality) == 1
        dialogue_mi_quality = dialogue_mi_quality[0]
        dialogue_mi_qualities.append(dialogue_mi_quality)
    dialogue_level_utt_counts["mi_quality"] = dialogue_mi_qualities

    def get_smallest_fold_id(folds: List[List[Dict[str, Any]]]):
        smallest_fold = min(
            folds, 
            key=(
                lambda elems: sum([elem["utt_count"] for elem in elems])
            )
        )
        index = folds.index(smallest_fold)
        return index

    all_folds = [[] for _ in range(num_fold)]
    for row in dialogue_level_utt_counts.sort_values(
        by="utt_count", ascending=False
    ).itertuples():
        all_folds[get_smallest_fold_id(all_folds)].append({
            "dialogue_id": getattr(row, "dialogue_id"),
            "mi_quality": getattr(row, "mi_quality"),
            "utt_count": getattr(row, "utt_count")
        })

    return all_folds
