import logging
import pickle

import nltk
import spacy

from ner_system.config import NERConfig
from .pipeline import CRFPipeline


logger = logging.getLogger(__name__)

ner_config = NERConfig(
    label_types=[
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-MISC",
        "I-MISC",
    ],
    data_path="data",
    models_path="models",
)


def init_resources():
    """
    Initializes the resources required by the NER System.

    Parameters:
        config (NERConfig): The configuration object for the NER system.

    Returns:
    - dict: A dictionary containing the initialized resources.

    """
    # Load NLTK resources
    nltk.download("gazetteers")
    nltk.download("names")
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger")

    logger.info("NLTK resources loaded")

    # Load SpaCy model
    spacy_model = spacy.load("en_core_web_sm")

    # Load persisted model(s)
    with open(ner_config.MODEL_PATH / "crf_model_optim.pkl", "rb") as f:
        crf_model = pickle.load(f)

    logger.info("CRF model loaded")

    assert spacy_model is not None, "SpaCy model not loaded"
    assert crf_model is not None, "CRF model not loaded"

    resources = {
        "ner_config": ner_config,
        "spacy_model": spacy_model,
        "crf_model": crf_model,
    }

    return resources
