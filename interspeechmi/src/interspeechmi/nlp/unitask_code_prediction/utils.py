import logging
from typing import Dict, List
from interspeechmi.nlp.unitask_code_prediction.constants import (
    DEBERTA_MAX_SEQ_LEN,
    ROBERTA_MAX_SEQ_LEN
)
from transformers import (
    PreTrainedTokenizerFast,
)

logger = logging.getLogger(__name__)


def tokenize_sequence_with_left_truncation(
    sequence: str,
    textual_label: str,
    label2id: Dict[str, int],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int=ROBERTA_MAX_SEQ_LEN,
):
    # tokenize without truncation or padding
    input_ids = tokenizer(
        text=sequence,
        add_special_tokens=False,
        padding="do_not_pad",
        truncation="do_not_truncate",
        max_length=None,
        return_attention_mask=False
    )["input_ids"]
    assert isinstance(input_ids, list)
    unprocessed_input_ids = input_ids.copy()

    # [CLS] + Context + [SEP]
    input_ids_len = len(input_ids) + 2
    if input_ids_len > max_seq_len:
        input_ids = input_ids[(input_ids_len - max_seq_len):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "label": label2id[textual_label],
        "unprocessed_input_ids": unprocessed_input_ids,
    }    


def tokenize_sequence_with_left_truncation_and_mi_quality_info(
    sequence: str,
    textual_label: str,
    label2id: Dict[str, int],
    mi_quality: str,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int=ROBERTA_MAX_SEQ_LEN,
):
    if "roberta" not in tokenizer.name_or_path:
        raise ValueError(
            f"""
            This function, 
            namely `tokenize_sequence_with_left_truncation_and_mi_quality_info`, 
            only works with RoBERTa tokenizers because RoBERTa doesn't 
            use token type IDs!
            """
        )

    assert mi_quality in ["high", "low"]
    mi_quality_prefix = f"[{mi_quality}]"
    mi_quality_prefix_input_ids = tokenizer(
        text=mi_quality_prefix,
        add_special_tokens=True,
        padding="do_not_pad",
        truncation="do_not_truncate",
        max_length=None,
        return_attention_mask=False
    )["input_ids"]
    assert isinstance(mi_quality_prefix_input_ids, list)
    max_seq_len_for_actual_seq = (
        max_seq_len - 
        len(mi_quality_prefix_input_ids)
    )

    # tokenize without truncation or padding
    seq_input_ids = tokenizer(
        text=sequence,
        add_special_tokens=False,
        padding="do_not_pad",
        truncation="do_not_truncate",
        max_length=None,
        return_attention_mask=False
    )["input_ids"]
    assert isinstance(seq_input_ids, list)
    unprocessed_input_ids = seq_input_ids.copy()

    # [SEP] + Context + [EOS]
    seq_input_ids_len = len(seq_input_ids) + 2
    if seq_input_ids_len > max_seq_len_for_actual_seq:
        seq_input_ids = seq_input_ids[(seq_input_ids_len - max_seq_len_for_actual_seq):]
    seq_input_ids = [tokenizer.sep_token_id] + seq_input_ids + [tokenizer.eos_token_id]

    input_ids = mi_quality_prefix_input_ids + seq_input_ids

    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "label": label2id[textual_label],
        "unprocessed_input_ids": unprocessed_input_ids,
    }    
