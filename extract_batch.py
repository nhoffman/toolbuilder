#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.13"
# dependencies = ["openai"]
# ///

"""Extract features from one or more input files

Expects OPENAI_API_KEY to be set in the environment.
OPENAI_BASE_URL can be used to set a custom API base URL.

This script is intended to be run using uv
(https://github.com/astral-sh/uv); follow instructions to install if
you don't have it already. Recommended method for macos and linux is

  curl -LsSf https://astral.sh/uv/install.sh | sh

Set environment variables:

  export OPENAI_API_KEY="sk-..."
  export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional

Run the script:

  extract=https://raw.githubusercontent.com/nhoffman/toolbuilder/refs/heads/main/extract_batch.py
  uv run $extract -h
"""

import argparse
import sys
import json
from pathlib import Path
import csv

from openai import OpenAI


def get_features(client: OpenAI,
                 content: str,
                 tools: list,
                 model: str,
                 prompt: str = None,
                 **kwargs) -> dict:

    messages = [
        {'role': 'user',
         'content': content},
    ]

    if prompt:
        messages.append({'role': 'user', 'content': prompt})

    response = client.responses.create(
        model=model,
        input=messages,
        tools=tools,
        tool_choice='required',
        **kwargs
    )

    return response.to_dict()


def feature_table(response: dict) -> list[dict]:
    output = (o for o in response['output'] if 'arguments' in o)
    return [json.loads(o['arguments']) for o in output]


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('schema', help="json file with feature schema",
                        type=argparse.FileType('r'))
    parser.add_argument('-i', '--infile', help="A single input file")
    parser.add_argument('-d', '--dirname', help="A directory of input files")
    parser.add_argument('-o', '--outfile', help="Output file",
                        default=sys.stdout, type=argparse.FileType('w'))
    parser.add_argument('-m', '--model', help="Model name",
                        default='gpt-4.1')

    args = parser.parse_args(arguments)

    if not (args.infile or args.dirname):
        parser.error('Either -i/--infile or -d/--dirname must be specified')

    files = [Path(args.infile)] if args.infile else []
    if args.dirname:
        files.extend(
            p for p in Path(args.dirname).iterdir() if p.is_file()
        )

    client = OpenAI()

    schema = json.load(args.schema)

    fieldnames = ['filename', 'item'] + list(schema['parameters']['properties'].keys())
    writer = csv.DictWriter(args.outfile, fieldnames=fieldnames)
    writer.writeheader()

    for infile in files:
        print(f'Processing {infile}...', file=sys.stderr)
        features = get_features(
            client=client,
            content=infile.read_text(),
            tools=[schema],
            model=args.model,
        )

        for i, feature in enumerate(feature_table(features), 1):
            tab = {'filename': infile.name, 'item': i}
            tab.update({k: '' for k in fieldnames})  # ensure all fields present
            tab.update(feature)
            writer.writerow(tab, extrasaction='ignore')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
