# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai",
# ]
# ///

import logging
import json
from pathlib import Path

from openai import OpenAI

log = logging.getLogger(__name__)


def get_features(client: OpenAI,
                 context: str,
                 prompt: str,
                 tools: list,
                 model: str = 'gpt-4o-mini',
                 **kwargs) -> dict:

    log.info(f'using model {model}')

    messages = [
        {'role': 'user',
         'content': context},
        {'role': 'user',
         'content': prompt},
    ]

    print(tools)

    response = client.responses.create(
        model=model,
        input=messages,
        tools=tools,
        tool_choice='required',
        **kwargs
    )

    return response.to_dict()


def feature_table(response: dict) -> list[dict]:
    print(response)
    return [o['arguments'] for o in response['output'] if 'arguments' in o]


def test_chat():
    client = OpenAI()

    example = Path('examples/ishlt_features')

    with open(example / 'specification.json') as f:
        tools = json.load(f)

    with open(example / 'context.txt') as f:
        context = f.read()

    response = get_features(
        client=client,
        context=context,
        prompt="Extract features from this pathology report",
        tools=[tools],
        model='gpt-4.1',
        temperature=1.0,
        n=1,
    )

    features = feature_table(response)
    for feature in features:
        print(feature)


if __name__ == "__main__":
    test_chat()
