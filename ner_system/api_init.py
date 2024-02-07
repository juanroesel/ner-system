from typing import Callable

from fastapi import FastAPI

from ner_system.api import ner


def initialize_app(lifespan: Callable = None) -> FastAPI:
    """
    Initializes the FastAPI application with the specified lifespan callable.

    Parameters:
        lifespan (Callable): A callable that is invoked when the application
            starts and shuts down. This is useful for initializing and cleaning
            up resources that are required during the application's lifespan.

    Returns:
    - FastAPI: The initialized FastAPI application.

    """
    app = FastAPI(lifespan=lifespan)

    # API routing of NER endpoint
    app.include_router(ner.router)

    return app
