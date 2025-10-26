# Build Tool definitions for OpenAI Function Calling API

The OpenAI [function calling API](https://platform.openai.com/docs/guides/function-calling)
provides a specification for extracting details from text and
representing the outputs in a json object. This is very useful in
situations where you want to provide the extracted features to a
downstream process for analysis.

The purpose of this application is to provide a user interface for
composing a tool specification to perform a specific data extraction
task.

Test the application here: https://toolbuilder.streamlit.app

Note the option to load example data to demonstrate the application.

## Run locally

Clone the project and enter the project directory.

```
git clone https://github.com/nhoffman/toolbuilder.git
cd toolbuilder
```

You will need to provide an OpenAI API key, either using an
environment variable, or within the app. For example:

```
export OPENAI_API_KEY="..."
```

You can also specify an alternative API endpoint using `OPENAI_BASE_URL`.

The easiest way to launch the app is with [uv](https://github.com/astral-sh/uv):

```
uv run --with-requirements requirements.txt streamlit run toolbuilder.py
```

Without uv:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run toolbuilder.py
```

The functions for interacting with the OpenAI API are defined in
`utils.py`. You can test the function calling API from the command
line like this (requires that `OPENAI_API_KEY` is defined):

```
% uv -q run --script utils.py
{'label': 'A', 'gleason_score': '3+4=7', 'biopsy_length': 2.1, 'cancer_length': 0.1, 'diagnosis': 'malignant'}
{'label': 'B', 'gleason_score': '', 'biopsy_length': 0, 'cancer_length': 0, 'diagnosis': 'benign'}
{'label': 'C', 'gleason_score': '', 'biopsy_length': 0, 'cancer_length': 0, 'diagnosis': 'benign'}
{'label': 'D', 'gleason_score': '3+4=7', 'biopsy_length': 2.5, 'cancer_length': 0.8, 'diagnosis': 'malignant'}
{'label': 'E', 'gleason_score': '3+4=7', 'biopsy_length': 2.2, 'cancer_length': 1.4, 'diagnosis': 'malignant'}
{'label': 'F', 'gleason_score': '3+4=7', 'biopsy_length': 2.9, 'cancer_length': 0.3, 'diagnosis': 'malignant'}
{'label': 'G', 'gleason_score': '3+4=7', 'biopsy_length': 2.0, 'cancer_length': 1.2, 'diagnosis': 'malignant'}
```

## Deploy

Deployment to Streamlit community cloud is performed by launching the
app locally and using the interactive "deploy" action.
