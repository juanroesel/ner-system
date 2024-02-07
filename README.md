# Named Entity Recognition (NER) System

## Requirements
In order to run the system, you need to have the following tools installed locally:

* Python (3.11.0)
* Docker [[Mac](https://docs.docker.com/desktop/install/mac-install/), [Windows](https://docs.docker.com/desktop/install/windows-install/)]
* [Poetry (1.6+)](https://python-poetry.org/docs/#installation)

## Set up

Clone the project repo:
```
https://github.com/juanroesel/ner-system.git
```

Navigate to the project root directory and install the `ner-system` package using Poetry:

```
poetry install
```

Activate the Poetry Shell, which enables the project's virtual environment:

```
poetry shell
```

Download SpaCy's English language model using the Poetry CLI:
```
poetry run python -m spacy download en_core_web_sm
```
> NOTE: Using a lightweight version for demonstration purposes. You can download a more robust English model folowing the guidelines from [this list](https://spacy.io/models/en).

Finally, unzip the provided `ner-system.zip` file and copy/paste the `data`, `models` and `artifacts` directories at the project's root level. If no such file was provided, create these directories at the root project level and follow the instructions on the `Reproducibility` section.

The following should be the final project structure after completing all these steps:

```
C:.
│   .env
│   .gitignore
│   LICENSE.md
│   poetry.lock
│   pyproject.toml
│   README.md
│   
├───.pytest_cache
│
├───artifacts
│       test_data.pkl
│       test_tags.pkl
│       train_data.pkl
│       train_tags.pkl
│       validation_data.pkl
│       validation_tags.pkl
│
├───data
│   ├───CoNLL003
│   │       metadata
│   │       test.txt
│   │       train.txt
│   │       valid.txt
│   │
│   └───DataWorld
│           cnn_data.json
│           cnn_data_sample.txt
│
├───models
│       crf_model_vanilla.pkl
|       crf_model_optim.pkl
│
├───ner_system
│   │   api_init.py
│   │   config.py
│   │   data_models.py
│   │   main.py
│   │   utils.py
│   │   __init__.py
│   │
│   ├───api
│   │   │   ner.py
│   │   │   sample_request.json
│   │   │   sample_response.json
│   │   │   __init__.py
│   │   │
│   │   └───__pycache__
│   │
│   ├───models
│   │   │   pipeline.py
│   │   │   __init__.py
│   │   │
│   │   └───__pycache__
│   │        
│   └───__pycache__
│
├───notebooks
│       R&D_BERT+CRF_Model.ipynb
│       R&D_CRF_Model.ipynb
│
└───tests
    │   test_api.py
    │   test_pipeline.py
    │   test_request.py
    │   __init__.py
    │
    └───__pycache__

```

## Usage

First, launch the NER System Application:
```
poetry run python ner-system/main.py

INFO:     Started server process [5980]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

Open your browser and navigate to the URL displayed on the terminal: http://localhost:8000. It should redirect to the API Documentation page located at URL http://localhost:8000/docs.

The endpoint `/api/v0/ner/predict` is responsible for extracting entities from a payload containing news articles. Sample JSON files `sample_request.json` and `sample_response.json` have been provided inside the `./api` directory to illustrate the API schema as documented in localhost:8000/docs.

## Tests
A simple unit test suite has been provided under the folder `./tests` and can be run using `PyTest`:

```
poetry run python pytest

===================================== test session starts =====================================
platform win32 -- Python 3.11.0, pytest-8.0.0, pluggy-1.4.0
rootdir: path/to/root/dir/ner-system
plugins: anyio-4.2.0
collected 6 items

tests\test_api.py ..                                                                     [ 33%]
tests\test_pipeline.py ....                                                              [100%]
```

To run an end-to-end integration test, run `test_request.py` as follows:

```
poetry run python tests/test_request.py
```


## Reproducibility
- Run the notebook `./notebooks/R&D_CRF_Model.ipynb` from top to bottom, which will produce a trained CRF model on the CoNNL2003 dataset, persisted into the `./models` folder. The corresponding artifacts are persisted on the `./artifacts` folder.

    > NOTE: Make sure to run the notebook using the Kernel that contains the virtual environment activated by Poetry. For reference, it should start with the name `ner-system-XXXX`.

- Launch the NER System API using the instructions provided in the `Usage` stage. Once it's running, send a request to the API by running the file `./tests/test_request.py`. You can use the Poetry shell as follows:

```
poetry run python tests/test_request.py
```

- Alternatively, you can use a software like Postman to submit the request, using the `sample_request.json` file located inside the `./api` folder as reference to fill the `Body` parameters.

## Contact
For any questions, comments or bug reporting, please submit an issue or contact me at juan at helelab dot org.