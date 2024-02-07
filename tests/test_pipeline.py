import pytest

from ner_system.model import init_resources
from ner_system.config import NERConfig
from ner_system.data_models import Article
from ner_system.model.pipeline import CRFPipeline

### Unit tests for the CRFPipeline class ###

resources = init_resources()


@pytest.fixture
def pipeline():
    test_config = NERConfig(
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
        dataset="conll2003",
    )
    return CRFPipeline(
        config=test_config, resources=resources, debug=False, persist_data=False
    )


def test_token2features(pipeline):
    tokens = ["This", "is", "a", "test"]
    idx = 1
    features = pipeline._token2features(tokens, idx)
    assert isinstance(features, dict)
    assert "token" in features


def test_tokenize(pipeline):
    text = "This is a test."
    method = "spacy"
    tokens = pipeline._tokenize(text, method)
    assert isinstance(tokens, list)
    assert "This" in tokens


def test_sentence2features(pipeline):
    tokens = ["This", "is", "a", "test"]
    features = pipeline._sentence2features(tokens)
    assert isinstance(features, list)
    assert isinstance(features[0], dict)
    assert "token" in features[0]


def test_labels_to_iob(pipeline):
    label_ids = [0, 1, 2, 3]
    labels = pipeline._labels_to_iob(label_ids)
    assert isinstance(labels, list)
    assert "O" in labels
