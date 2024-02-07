import sys
import time
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from tqdm import tqdm

from ner_system.config import NERConfig
from ner_system.data_models import Article
from ner_system.utils import get_root_directory

import nltk
from nltk import pos_tag
from nltk.corpus import names, gazetteers, stopwords


class CRFPipeline:
    """
    Class to model the end-to-end pipeline for the CRF model.
    """

    def __init__(
        self,
        config: NERConfig,
        resources: Dict[str, Any],
        debug: bool = False,
        persist_data: bool = False,
    ):
        self.config = config
        if not resources:
            raise ValueError(
                "A dictionary with API resources must be provided to run the pipeline"
            )
        self.resources = resources
        self.debug = debug
        self.persist_data = persist_data
        self.logger = logging.getLogger(self.__class__.__name__)

    def _token2features(self, tokens: List[str], idx: int):
        """
        Generates a feature dictionary for a given token, using the following features:
        - Features which looks at neighbouring words.
        - Features which looks at word morphology.
        - Features which considers the "shape" of word.
        - Features which include POS tags.
        - Gazetteer features using the nltk Gazetteer corpus.
        - Name features using the nltk Names corpus.
        - Stop words features using the nltk StopWords corpus.

        """
        # load reference vocabularies and POS tags
        _gazetteer_words = set(gazetteers.words())
        _names_words = set(names.words())
        _stop_words = set(stopwords.words())
        _pos_tags = pos_tag(tokens)

        token_features = {}

        # token
        token_features["token"] = tokens[idx]

        # POS tags
        token_features["pos_tag_full"] = _pos_tags[idx][1]
        token_features["pos_tag_short"] = _pos_tags[idx][1][:2]

        # lower case
        token_features["token_lower"] = tokens[idx].lower()

        # first word upper case (boolean)
        token_features["token_upper"] = True if idx == 0 else False

        # upper case (boolean)
        token_features["token_upper"] = tokens[idx].isupper()

        # tile case (boolean)
        if not tokens[idx].isupper() and not tokens[idx].islower() and idx != 0:
            token_features["token_tile"] = True
        else:
            token_features["token_tile"] = False

        # digit case (boolean)
        token_features["token_digit"] = tokens[idx].isdigit()

        # all alpha case (boolean)
        token_features["token_alpha"] = tokens[idx].isalpha()

        # token is title (boolean)
        token_features["token_title"] = tokens[idx].istitle()

        # space case (boolean)
        token_features["token_space"] = tokens[idx].isspace()

        # word morphology - 3 chars
        if len(tokens[idx]) > 3:
            token_features["token_morph_-3"] = tokens[idx][:3]
            token_features["token_morph_+3"] = tokens[idx][-3:]

        # word morphology - 2 chars
        if len(tokens[idx]) > 2:
            token_features["token_morph_-2"] = tokens[idx][:2]
            token_features["token_morph_+2"] = tokens[idx][-2:]

        # neighbours with corresponding POS tags
        if idx > 0:
            token_features["prev_token"] = tokens[idx - 1]
            token_features["prev_token_pos_full"] = _pos_tags[idx - 1][1]
            token_features["prev_token_pos_short"] = _pos_tags[idx - 1][1][:2]
        else:
            token_features["BOS"] = True

        if idx > 1:
            token_features["prev_prev_token"] = tokens[idx - 2]
            token_features["prev_prev_token_pos_full"] = _pos_tags[idx - 2][1]
            token_features["prev_prev_token_pos_short"] = _pos_tags[idx - 2][1][:2]

        if idx < len(tokens) - 2:
            token_features["next_next_token"] = tokens[idx + 2]
            token_features["next_next_token_pos_full"] = _pos_tags[idx + 2][1]
            token_features["next_next_token_pos_short"] = _pos_tags[idx + 2][1][:2]

        if idx < len(tokens) - 1:
            token_features["next_token"] = tokens[idx + 1]
            token_features["next_token_pos_full"] = _pos_tags[idx + 1][1]
            token_features["next_token_pos_short"] = _pos_tags[idx + 1][1][:2]
        else:
            token_features["EOS"] = True

        # gazetteer features
        if tokens[idx] in _gazetteer_words:
            token_features["gazetteer"] = tokens[idx]

        # name features
        if tokens[idx] in _names_words:
            token_features["name"] = tokens[idx]

        # stop words features
        if tokens[idx] in _stop_words:
            token_features["stop_word"] = tokens[idx]

        return token_features

    def _tokenize(self, text: str, method: str = "spacy"):
        """
        Tokenizes the given text using the specified tokenizer.
        """
        if method == "spacy":
            _nlp = self.resources.get("spacy_model")
            return [token.text for token in _nlp(text)]

        elif method == "nltk":
            from nltk.tokenize import word_tokenize

            return word_tokenize(text)

        elif method == "bert":
            return self.config.TOKENIZER.tokenize(text)

        else:
            raise ValueError(
                "Invalid tokenization method. Must be one of 'spacy', 'nltk', or 'bert'"
            )

    def _sentence2features(self, tokens: List[str]):
        """
        Generates a list of feature dictionaries for a given list of tokens.
        """
        return [self._token2features(tokens, idx) for idx in range(len(tokens))]

    def _labels_to_iob(self, label_ids: List[int]):
        """
        Convert a list of label ids to IOB format
        """
        return [self.config.ID2LABEL[label_id] for label_id in label_ids]

    def prepare_train_data(self, split: str):
        """
        Runs a simple data processing/transformation pipeline to convert
        the raw data into a format that can be used to train the CRF model.
        """
        if split not in ["train", "validation", "test"]:
            raise ValueError("split must be one of 'train', 'validation', 'test'")

        self.logger.info(f"Initializing data processing for {split} split...")
        _start_time = time.perf_counter()

        if self.debug:
            _data = self.config.dataset[split][:10]
            split_feature_dicts = [
                self._sentence2features(tokens) for tokens in tqdm(_data["tokens"])
            ]
            split_tags = [self._labels_to_iob(tags) for tags in tqdm(_data["ner_tags"])]

        else:
            _data = self.config.dataset[split]
            split_feature_dicts = [
                self._sentence2features(item["tokens"]) for item in tqdm(_data)
            ]
            split_tags = [self._labels_to_iob(item["ner_tags"]) for item in tqdm(_data)]

        _end_time = time.perf_counter()
        # self.logger.info(f"Data processing for {split} split completed in {_end_time - _start_time:0.4f} seconds")
        self.logger.info(
            f"Data processing for {split} split completed in {_end_time - _start_time:0.4f} seconds"
        )

        if self.persist_data:
            _output_path = self.config.ARTIFACTS_PATH / f"{split}_data.pkl"
            _output_path_tags = self.config.ARTIFACTS_PATH / f"{split}_tags.pkl"

            self.logger.info(f"Persisting {split} features to {_output_path}...")
            self.logger.info(f"Persisting {split} tags to {_output_path_tags}...")
            self.logger.info(f"Persisting {split} features to {_output_path}...")
            self.logger.info(f"Persisting {split} tags to {_output_path_tags}...")

            with open(get_root_directory() / _output_path, "wb") as f:
                pickle.dump(split_feature_dicts, f)
            with open(get_root_directory() / _output_path_tags, "wb") as f:
                pickle.dump(split_tags, f)

        return split_feature_dicts, split_tags

    def inference(self, article: Article, resources: Dict[str, Any]):
        """
        Takes an article and processes it to extract named entities.

        Parameters:
        - article (Article): The article to be processed.
        - resources (dict): The resources required to process the article (see api_init.py).
        """
        self.logger.info(f"Initializing inference for article {article.article_id}...")
        _start_time = time.perf_counter()

        spacy_model = resources.get("spacy_model")
        crf_model = resources.get("crf_model")

        article_resp = {"article_id": article.article_id, "entities": []}

        sents = [sent for sent in spacy_model(article.content).sents]
        for idx, sent in tqdm(enumerate(sents)):
            tokens = [token.text for token in sent]  # Sentencizer already tokenizes
            feature_dicts = self._sentence2features(tokens)

            assert len(tokens) == len(
                feature_dicts
            ), "Length of tokens and feature dicts must be the same"

            labels = crf_model.predict([feature_dicts])
            resp = {"sent_id": idx, "tokens": tokens, "labels": labels}

            article_resp["entities"].append(resp)

        _end_time = time.perf_counter()
        self.logger.info(
            f"Inference for article {article.article_id} completed in {_end_time - _start_time:0.4f} seconds"
        )

        return article_resp
