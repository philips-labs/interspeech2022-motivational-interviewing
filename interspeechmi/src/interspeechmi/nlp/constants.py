import os

from interspeechmi.constants import CACHE_DIR, MODELS_DIR, VISUALS_DIR
from interspeechmi.data_handling.constants import ANNO_MI_DATA_DIR

TRANSFORMERS_CACHE_DIR = os.path.join(CACHE_DIR, "transformers_cache")
HF_DATASETS_CACHE_DIR = os.path.join(CACHE_DIR, "hf_datasets_cache")
GENSIM_CACHE_DIR = os.path.join(CACHE_DIR, "gensim_cache")

SENTENCE_TRANSFORMERS_CACHE_DIR = os.path.join(CACHE_DIR, "sentence_transformers_cache")
# SENTENCE_TRANSFORMERS_EMBEDDING_EXPERIMENTS_CACHE_DIR = \
#     os.path.join(SENTENCE_TRANSFORMERS_CACHE_DIR, "embedding_experiments")

for cache_dir in [
    TRANSFORMERS_CACHE_DIR, HF_DATASETS_CACHE_DIR, 
    GENSIM_CACHE_DIR, SENTENCE_TRANSFORMERS_CACHE_DIR,
    # SENTENCE_TRANSFORMERS_EMBEDDING_EXPERIMENTS_CACHE_DIR
]:
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

ANNO_MI_TFIDF_FEATUERS_PATH = os.path.join(
    ANNO_MI_DATA_DIR, 
    "anno_mi_tfidf_features.npz"
)


THERAPIST_CODES = [
    "question", "input", "reflection", "other"
]

CLIENT_CODES = [
    "change", "neutral", "sustain"
]


HF_TOKENIZERS_DIR = os.path.join(MODELS_DIR, "hf_tokenizers")
if not os.path.isdir(HF_TOKENIZERS_DIR):
    os.mkdir(HF_TOKENIZERS_DIR)
