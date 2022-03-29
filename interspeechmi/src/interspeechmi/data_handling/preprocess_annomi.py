import logging
import random
import re
import pandas as pd
import requests
import os
import zipfile
import spacy
import torch
import numpy as np
import json
import itertools
from interspeechmi.nlp.constants import TRANSFORMERS_CACHE_DIR
from typing import Any, Dict, List
from tqdm import tqdm
from transformers import (
    PegasusForConditionalGeneration, 
    PegasusTokenizer
)
from datasets import load_dataset
from interspeechmi.data_handling.constants import (
    ANNO_MI_DATASET_DIR, 
    ANNO_MI_DATASET_ZIP_PATH,
    ANNO_MI_NORMALIZED_AUGMENTED_PATH,
    ANNO_MI_NORMALIZED_PATH,
    ANNO_MI_PATH, 
    ANNO_MI_URL
)
from unidecode import unidecode

logger = logging.getLogger(__name__)


def preprocess_anno_mi(overwrite: bool=False):
    """
    The overall pipeline of preprocessing AnnoMI
    """
    download_anno_mi(overwrite=overwrite)
    unzip_anno_mi(overwrite=overwrite) 
    normalize_anno_mi(overwrite=overwrite)
    augment_anno_mi(
        batch_size=16,
        overwrite=overwrite,
        num_paraphrases_per_utt=10
    )


def download_anno_mi(
    anno_mi_url: str=ANNO_MI_URL,
    anno_mi_local_path: str=ANNO_MI_DATASET_ZIP_PATH,
    overwrite: bool=False
):
    if os.path.isfile(anno_mi_local_path) and not overwrite:
        logger.info(
            f"""
            AnnoMI .zip already exists at {anno_mi_local_path},
            so we're skipping downloading.
            """
        )
        return
    
    logger.info(f"Downloading AnnoMI to {anno_mi_local_path}")
    response = requests.get(anno_mi_url, allow_redirects=True)
    with open(anno_mi_local_path, 'wb') as anno_mi_writer:
        anno_mi_writer.write(response.content)
    

def unzip_anno_mi(
    anno_mi_local_path: str=ANNO_MI_DATASET_ZIP_PATH,
    anno_mi_dataset_dir: str=ANNO_MI_DATASET_DIR,
    overwrite: bool=False
):
    if (
        os.path.isdir(anno_mi_dataset_dir) and 
        os.listdir(anno_mi_dataset_dir) and 
        not overwrite
    ):
        logger.info(
            f"""
            AnnoMI has already been unzipped and its content
            is in {anno_mi_dataset_dir}, so we're skipping unzipping
            """
        )
        return
    
    logger.info(f"Unzipping AnnoMI to {anno_mi_dataset_dir}")

    if not os.path.isdir(anno_mi_dataset_dir):
        os.mkdir(anno_mi_dataset_dir)

    with zipfile.ZipFile(anno_mi_local_path, 'r') as zip_ref:
        zip_ref.extractall(anno_mi_dataset_dir)
    
    for filename in ["dataset.csv", "README.md"]:
        os.rename(
            os.path.join(anno_mi_dataset_dir, "AnnoMI-main", filename),
            os.path.join(anno_mi_dataset_dir, filename),
        )
    os.rmdir(os.path.join(anno_mi_dataset_dir, "AnnoMI-main"))


def normalize_anno_mi(
    anno_mi_path: str=ANNO_MI_PATH,
    output_path: str=ANNO_MI_NORMALIZED_PATH,
    overwrite: bool=False
):
    if (
        os.path.isfile(output_path) and 
        not overwrite
    ):
        logger.info(
            f"""
            Normalization has been conducted on the utterances.
            Please find the results at {output_path}
            """
        )
        return

    logger.info("Normalizing utterances in AnnoMI ...")

    assert os.path.isfile(anno_mi_path)
    anno_mi = pd.read_csv(anno_mi_path, keep_default_na=False)
    anno_mi["utterance_text"] = anno_mi["utterance_text"].\
        apply(normalize_utterance)
    anno_mi.to_csv(output_path, index=False)


def normalize_utterance(utt: str):
    """
    Replace special symbols and patterns like [chuckles]

    If an utterance becomes null after normalization, we just assign "Hmm." to it
    because this is probably the safest replacement without making the utterance's
    annotation totally useless
    """
    normalized_utt = utt.replace('£', '$').replace('\\', '').replace('\n', '')
    normalized_utt = unidecode(normalized_utt)
    ## Special handling of the NZ English word "whana"/"whanau" which means family
    normalized_utt.replace("whana", "family")
    normalized_utt.replace("whanau", "family")

    special_patterns = [
        r'\[unintelligible \d\d:\d\d:\d\d\]',
        r'\[unintelligible \d\d:\d:\d\d\]',
        r'\[inaudible \d\d:\d\d:\d\d\]',
        r'\[chuckles\]',
        r'\[chuckling\]',
        r'\[clears throat\]',
        r'\[clear throat\]',
        r'\[sigh\]',
        r'\[sighs\]',
        r'\[signs\]',
        r'\[laughs\]',
        r'\[laughing\]',
        r'\[laughter\]',
        r'\[scoffs\]',
        r'\[silence\]',
        r'\[coughs\]',
        r'\[crosstalk\]',
        r'\[croostalk\]',
        r'\[foreign language\]',
        r'\[nods\]',
        r'\[door closes\]',
        r'Speaker 1:', 
    ]
    for pattern in special_patterns:
        normalized_utt = re.sub(pattern, '', normalized_utt).replace("  ", ' ').strip()

    normalized_utt = normalized_utt.strip()

    if len(normalized_utt) == 0:
        normalized_utt = "Hmm."

    return normalized_utt


def augment_anno_mi(
    anno_mi_path: str=ANNO_MI_NORMALIZED_PATH,
    output_path: str=ANNO_MI_NORMALIZED_AUGMENTED_PATH,
    batch_size: int=8,
    num_paraphrases_per_utt: int=10,
    random_seed: int=42,
    overwrite: bool=False,
):
    if (
        os.path.isfile(output_path) and 
        not overwrite
    ):
        logger.info(
            f"""
            Augmentation on normalized AnnoMI has been done.
            Please find the results at {output_path}
            """
        )
        return

    anno_mi_utts_sentence_level_path = f"{anno_mi_path}.utts.sentence_level.csv"
    if (
        os.path.isfile(anno_mi_utts_sentence_level_path) and 
        not overwrite
    ):
        anno_mi_utts_sentence_level = pd.read_csv(
            anno_mi_utts_sentence_level_path, 
            keep_default_na=False
        )
    else:
        anno_mi = pd.read_csv(anno_mi_path, keep_default_na=False)

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

        anno_mi_ = anno_mi[["transcript_id", "utterance_id", "utterance_text"]]
        anno_mi_utts_sentence_level = {
            "transcript_id": [],
            "utterance_id": [],
            "sentence_id": [],
            "sentence": []
        }
        for row in anno_mi_.itertuples():
            utt = getattr(row, "utterance_text")
            for sent_id, sent in enumerate(nlp(utt).sents):
                anno_mi_utts_sentence_level["transcript_id"].append(getattr(row, "transcript_id"))
                anno_mi_utts_sentence_level["utterance_id"].append(getattr(row, "utterance_id"))
                anno_mi_utts_sentence_level["sentence_id"].append(sent_id)
                anno_mi_utts_sentence_level["sentence"].append(str(sent))
        anno_mi_utts_sentence_level = pd.DataFrame.from_dict(anno_mi_utts_sentence_level)    

        anno_mi_utts_sentence_level.to_csv(
            anno_mi_utts_sentence_level_path,
            index=False
        )

    anno_mi_utts_sentence_level_augmented_path = f"{anno_mi_utts_sentence_level_path}.augmented.csv"
    if (
        overwrite or
        not os.path.isfile(anno_mi_utts_sentence_level_augmented_path)
    ):
        anno_mi_utts_sentence_level = load_dataset(
            "csv", data_files=anno_mi_utts_sentence_level_path
        )["train"]
        model_name = 'tuner007/pegasus_paraphrase'
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = PegasusTokenizer.from_pretrained(
            model_name,
            cache_dir=TRANSFORMERS_CACHE_DIR
        )
        model = PegasusForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=TRANSFORMERS_CACHE_DIR
        ).to(torch_device)

        def paraphrase(examples: Dict[str, List[Any]]):
            batch = tokenizer(
                examples["sentence"],
                truncation=True,
                padding="longest",
                max_length=60, 
                return_tensors="pt"
            ).to(torch_device)
            generations = model.generate(
                **batch,
                max_length=60,
                num_beams=10, 
                num_return_sequences=10, 
                temperature=1.5
            )        
            augmentations = tokenizer.batch_decode(
                generations, 
                skip_special_tokens=True
            )
            augmentations = np.array(augmentations).\
                reshape(len(examples["sentence"]), -1).tolist()
            for example_id in range(len(augmentations)):
                augmentations[example_id] = json.dumps(
                    augmentations[example_id]
                )
            return {
                "augmentations": augmentations
            }

        # ### DEBUG ###
        # anno_mi_utts_sentence_level = anno_mi_utts_sentence_level.select(range(50))

        anno_mi_utts_sentence_level_augmented = \
            anno_mi_utts_sentence_level.map(
                paraphrase,
                batched=True,
                batch_size=batch_size
            )
        
        anno_mi_utts_sentence_level_augmented.to_csv(
            anno_mi_utts_sentence_level_augmented_path, 
            index=False
        )

    anno_mi_utts_sentence_level_augmented = pd.read_csv(
        anno_mi_utts_sentence_level_augmented_path, 
        keep_default_na=False
    )

    anno_mi = pd.read_csv(anno_mi_path, keep_default_na=False)
    rand = random.Random(random_seed)
    anno_mi_utt_augmentations = []
    for row in tqdm(anno_mi.itertuples(), total=anno_mi.shape[0]):
        # ############## DEBUG #############
        # if row.Index < 774:
        #     continue
        # ##################################

        transcript_id = getattr(row, "transcript_id")
        utterance_id = getattr(row, "utterance_id")
        utt_sentences_augmentations = anno_mi_utts_sentence_level_augmented.loc[
            (anno_mi_utts_sentence_level_augmented["transcript_id"] == transcript_id) &
            (anno_mi_utts_sentence_level_augmented["utterance_id"] == utterance_id)
        ]["augmentations"]

        if len(utt_sentences_augmentations) > 1: # utterance has more than one sentence
            utt_sentences_augmentations = [
                json.loads(augmentations) for augmentations \
                    in utt_sentences_augmentations.tolist()
            ]
            utt_augmentations = []
            while len(utt_augmentations) < num_paraphrases_per_utt:
                utt_augmentation = []
                for sent_augmentations in utt_sentences_augmentations:
                    sent_augmentation = rand.sample(sent_augmentations, 1)[0]
                    utt_augmentation.append(sent_augmentation)
                utt_augmentation = ' '.join(utt_augmentation)
                if utt_augmentation not in utt_augmentations:
                    utt_augmentations.append(utt_augmentation)
        else:
            utt_augmentations = rand.sample(
                    json.loads(utt_sentences_augmentations.iloc[0]), 
                    num_paraphrases_per_utt
                )
        anno_mi_utt_augmentations.append(json.dumps(utt_augmentations))

    anno_mi["utterance_augmentations"] = anno_mi_utt_augmentations
    anno_mi.to_csv(output_path, index=False)
