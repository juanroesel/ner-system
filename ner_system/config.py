import logging
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

import spacy

from datasets import load_dataset

import transformers

load_dotenv()


class NERConfig:
    """
    Class to hold all the configuration parameters along the NER pipeline.
    """

    def __init__(
        self,
        data_path: Optional[str | Path] = None,
        models_path: Optional[str | Path] = None,
        artifacts_path: Optional[str | Path] = None,
        label_types: Optional[List[str]] = None,
        dataset: Optional[str] = None,
        train_file: Optional[str] = None,
        dev_file: Optional[str] = None,
        test_file: Optional[str] = None,
        base_model_path: Optional[str] = None,
        tokenizer: Optional[transformers.BertTokenizer] = None,
        **kwargs,
    ):
        if data_path and isinstance(data_path, str):
            self.DATA_PATH = Path(data_path)
        if models_path and isinstance(models_path, str):
            self.MODEL_PATH = Path(models_path)
        if artifacts_path and isinstance(artifacts_path, str):
            self.ARTIFACTS_PATH = Path(artifacts_path)
        # DATA CONFIG
        if dataset:  # Must be a valid HuggingFace dataset
            self.dataset = load_dataset(dataset)
            self.LABEL_TYPES = self.dataset["train"].features["ner_tags"].feature.names
        else:
            if not label_types:
                raise ValueError("Label types must be provided if no dataset is given.")
            self.LABEL_TYPES = label_types
            self.dataset = None
        self.ID2LABEL = {i: label for i, label in enumerate(self.LABEL_TYPES)}
        if train_file:
            self.train_file = self.DATA_PATH / train_file
        if dev_file:
            self.dev_file = self.DATA_PATH / dev_file
        if test_file:
            self.test_file = self.DATA_PATH / test_file
        # BASE BERT CONFIG
        self.BASE_MODEL_PATH = base_model_path or "bert-base-cased"
        self.MAX_LEN = 128
        self.TRAIN_BATCH_SIZE = 64
        self.VALID_BATCH_SIZE = 32
        self.EPOCHS = 15
        self.OUT_DIM = 768  # bert-base-cased: 768, bert-large-cased: 1024
        self.TOKENIZER = tokenizer or transformers.BertTokenizer.from_pretrained(
            self.BASE_MODEL_PATH, do_lower_case=False
        )
        # BASE CRF CONFIG
        self.CRF_ALGORITHM = "lbfgs"
        self.CRF_C1 = 0.1
        self.CRF_C2 = 0.1
        self.CRF_MAX_ITER = 100
        self.logger = logging.getLogger(self.__class__.__name__)
        # KWARGS
        if kwargs:
            for attr, value in kwargs.items():
                setattr(self, attr, value)
