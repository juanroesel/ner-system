from typing import List, Dict, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class RequestBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    params: Optional[dict] = None
    debug: Optional[bool] = False


class Article(BaseModel):
    # NOTE: More params could be added as needed
    article_id: Union[int, str]
    content: str
    title: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[str] = None


class NERRequest(RequestBaseModel):
    news_source: str
    articles: List[Article]


class Entities(BaseModel):
    sent_id: Union[int, str]
    tokens: List[str]
    labels: List[List[str]]


class ArticleResponse(BaseModel):
    article_id: Union[int, str]
    entities: List[Entities]


class NERResponse(BaseModel):
    news_source: str
    articles: List[ArticleResponse]
