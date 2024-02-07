from fastapi.testclient import TestClient

from ner_system.main import app


client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_entities():
    dummy_request = {
        "news_source": "test",
        "articles": [
            {
                "article_id": "1",
                "content": "This is a test article."
            }
        ]
    }
    response = client.post("/api/v0/ner/predict", json=dummy_request)
    assert response.status_code == 200