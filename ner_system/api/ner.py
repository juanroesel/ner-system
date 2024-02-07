import logging
from typing import Annotated

from ner_system.config import NERConfig
from ner_system.models.pipeline import CRFPipeline
from ner_system.data_models import NERRequest, NERResponse
from ner_system.models import init_resources

from fastapi import APIRouter, Depends

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/v0/ner",
    tags=["NER"],
    responses={404: {"description": "Not found"}},
)


@router.post("/predict", response_model=NERResponse, tags=["NER"])
async def predict_entities(
    request: NERRequest, resources: dict = Depends(init_resources)
) -> NERResponse:
    """
    Predicts the named entities in the given articles.

    Parameters:
    - request (NERRequest): The request object containing the articles to be processed.

    Returns:
    - NERResponse: The response object containing the named entities extracted from the articles.
    """
    logger.info(
        f"Received request to predict entities for: {request.news_source} with {len(request.articles)} articles..."
    )

    ner_config = resources.get("ner_config")
    crf_pipeline = CRFPipeline(
        ner_config, resources=resources, debug=False, persist_data=False
    )
    response = {"news_source": request.news_source, "articles": []}

    for article in request.articles:
        article_resp = crf_pipeline.inference(article, resources)
        response["articles"].append(article_resp)

    logger.info(f"Returning response for {request.news_source}...")

    return response
