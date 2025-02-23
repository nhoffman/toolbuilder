# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai",
# ]
# ///

import logging
import json

from openai import OpenAI

log = logging.getLogger(__name__)


example_context = """
A) Prostate, Right Base, core biopsy:
- Prostatic adenocarcinoma.
- Gleason score: 3+4=7
- 0.1 cm total cancer of 2.1 cm total core length, involving 1 of 2 cores
B) Prostate, Right Mid, core biopsy:
- Benign prostatic tissue.
C) Prostate, Right Apex, core biopsy:
- Benign prostatic tissue.
D) Prostate, Left Base, core biopsy:
- Prostatic adenocarcinoma.
- Gleason score: 3+4=7
- 0.8 cm total cancer of 2.5 cm total core length, involving 2 of 2 cores
E) Prostate, Left Mid, core biopsy:
- Prostatic adenocarcinoma.
- Gleason score: 3+4=7
- 1.4 cm total cancer of 2.2 cm total core length, involving 2 of 2 cores
F) Prostate, Left Apex, core biopsy:
- Prostatic adenocarcinoma.
- Gleason score: 3+4=7
- 0.3 cm total cancer of 2.9 cm total core length, involving 1 of 2 cores
G) Prostate, Lesion 1 (left base), core biopsy:
- Prostatic adenocarcinoma.
- Gleason score: 3+4=7
- 1.2 cm total cancer of 2.0 cm total core length, involving 3 of 3 cores
""".strip()

example_prompt = """
Extract the features from the pathology report
""".strip()


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

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice='required',
        **kwargs
    )

    return response.to_dict()


def feature_table(response: dict) -> list[dict]:
    output = []
    for choice in response['choices']:
        for tool_call in choice['message']['tool_calls']:
            d = {}
            if len(response['choices']) > 1:
                d['choice_index'] = choice['index']

            d.update(json.loads(tool_call['function']['arguments']))
            output.append(d)

    return output


def test_chat():
    client = OpenAI()

    with open('get_prostate_biopsies.json') as f:
        tools = json.load(f)

    response = get_features(
        client=client,
        context=example_context,
        prompt=example_prompt,
        tools=[tools],
        model='gpt-4o-mini',
        temperature=1.0,
        n=1,
    )

    features = feature_table(response)
    for feature in features:
        print(feature)


if __name__ == "__main__":
    test_chat()
