from enum import Enum
import json
from pathlib import Path

from openai import OpenAI, OpenAIError
import streamlit as st

import utils

DEFAULT_MODEL = 'gpt-5.1-chat'

MODELS = [
    'gpt-5',
    'gpt-5-mini',
    'gpt-5.1-chat',
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4o',
    'gpt-4o-mini'
    ]

st.set_page_config(layout="wide")


class ParameterType(Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    # OBJECT = "object"


def getval(key, default=None):
    return st.session_state.get(key, default)


def unwrap(text):
    return ' '.join(text.split())


def submit_query():
    if getval("context"):
        tool_spec = getval('tool_spec')
        model = getval('model', DEFAULT_MODEL)
        args = {
            'client': getval("client"),
            'context': getval('context'),
            'model': model,
            'prompt': getval('prompt'),
            'temperature': getval('temperature', 1.0),
            'tools': [tool_spec]
            }
        try:
            response = utils.get_features(**args)
            st.session_state['response'] = response
            st.session_state['features'] = utils.feature_table(response)
        except Exception as e:
            st.error(e)


def get_nested(d, *args):
    d = d.copy()
    for key in args:
        if d := d.get(key):
            continue
        else:
            return None

    return d


def set_tool_spec(tool_spec):
    output = {}
    output['func_name'] = get_nested(tool_spec, 'name')
    output['func_desc'] = get_nested(tool_spec, 'description')

    properties = get_nested(tool_spec, 'parameters', 'properties')

    required = set(get_nested(tool_spec, 'parameters', 'required'))

    output['num_features'] = len(properties)

    for i, (name, d) in enumerate(properties.items(), 1):
        output[f"feat_name_{i}"] = name
        output[f"feat_required_{i}"] = name in required
        st.session_state["feat_required_{i}_changed"] = False
        for key, value in d.items():
            if key == "enum":
                output[f"feat_{key}_{i}"] = ', '.join(value)
            else:
                output[f"feat_{key}_{i}"] = value

    st.session_state["num_features_changed"] = False
    st.session_state['uploaded_data'] = output


def set_uploaded_data():
    if uploaded_file := st.session_state.get('uploaded_file'):
        set_tool_spec(json.loads(uploaded_file.read()))


def get_uploaded(key):
    """Return the value from uploaded_data, but only if the
    corresponding field is empty.

    """
    if st.session_state.get('uploaded_data') and not st.session_state.get(key):
        return st.session_state['uploaded_data'].get(key)


def load_example_data():
    data_dir = Path('examples') / st.session_state['example_data']
    with open(data_dir / 'specification.json') as f:
        set_tool_spec(json.load(f))

    with open(data_dir / 'context.txt') as f:
        st.session_state["context"] = f.read()

    st.session_state["prompt"] = "Extract data from this report"


def get_or_reset(key, default=None, condition=True):
    if val := get_uploaded(key):
        if key in st.session_state and condition:
            del st.session_state[key]
    else:
        val = getval(key, default)

    return val


def on_click_num_features():
    st.session_state["num_features_changed"] = True


def on_click_required_changed(i):
    st.session_state[f"feat_required_{i}_changed"] = True


def get_openai_client():
    try:
        st.session_state["client"] = OpenAI(
            api_key=getval('api_key'),
            base_url=getval('base_url')
        )
    except OpenAIError:
        st.error('Set environment variables OPENAI_API_KEY, OPENAI_BASE_URL')


@st.dialog("Load Example Data")
def load_example_modal():
    st.write("""Load the example input text, prompt, and function
    definition? This will overwrite any data that you have entered.""")

    st.selectbox(
        "Load Example",
        ["ishlt_features", "prostate_biopsies"],
        key="example_data"
    )

    if st.button("Load", on_click=load_example_data):
        st.session_state['example_data_was_loaded'] = True
        st.rerun()


with st.sidebar:
    st.title("Feature Workbench")
    st.write(
        "This app allows you to define a function specification and "
        "extract features from a document using OpenAI's function calling "
        "capabilities. "
        "See [OpenAI's documentation](https://platform.openai.com/docs/guides/gpt/function-calling)")

    try:
        st.session_state['client'] = OpenAI()
    except OpenAIError:
        st.text_input(
            "OpenAI API Key", type="password",
            key="api_key",
            on_change=get_openai_client)

        st.text_input(
            "Base URL (optional)",
            placeholder="https://api.openai.com/v1",
            key="base_url",
            on_change=get_openai_client)

        get_openai_client()

    if st.button("Test API Key"):
        try:
            completion = getval('client').responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "user",
                        "content": "What is the capital of France?"
                        }
                ]
            )
            # st.write(completion.choices[0].message.content)
            st.success("API key is valid")
        except OpenAIError as e:
            st.error(e)


st.header('Feature extraction using OpenAI function calling')

# st.write(st.session_state.get('uploaded_data'))

with st.container(border=True):
    form_col1, form_col2 = st.columns(2)
    with form_col1:
        if "context" not in st.session_state:
            st.session_state["context"] = ""

        context = st.text_area(
            "Document Content", key="context",
            placeholder="Enter the document content here",
            height=300)

    with form_col2:
        st.text_area(
            "Prompt", key="prompt",
            placeholder=unwrap(
                """Optional. Use this area to provide additional
                instructions or examples for representing the output.
                """))
        model = getval('model', DEFAULT_MODEL)
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox(
                'Model',
                MODELS,
                index=MODELS.index(model),
                key='model'
                )
        with c2:
            temperature = st.slider(
                "Temperature", 0.0, 2.0, 1.0, 0.1, key="temperature")
        submitted = st.button("Submit", on_click=submit_query)

col1, __ = st.columns(2)
with col1:
    if st.session_state.get('example_data_was_loaded'):
        st.write('Reload page to clear')
        st.button("Clear all data", on_click=st.session_state.clear)
    else:
        if st.button("Load Example Data"):
            load_example_modal()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Function Definition", divider=True)
    subcol1, subcol2 = st.columns(2)

    with subcol1:
        func_name = get_or_reset('func_name')
        st.text_input(
            "Function name", key="func_name",
            value=func_name,
            placeholder="lowercase_with_underscores")

    with subcol2:
        if 'num_features' not in st.session_state:
            st.session_state['num_features'] = 1
        if 'num_features_changed' not in st.session_state:
            st.session_state['num_features_changed'] = False

        if ('uploaded_data' in st.session_state and
                not st.session_state['num_features_changed']):
            uploaded_data = st.session_state['uploaded_data']
            st.session_state['num_features'] = uploaded_data['num_features']

        number_of_features = st.number_input(
            "Number of features",
            key='num_features',
            min_value=1, max_value=20,
            on_change=on_click_num_features
        )

    st.text_area(
        "Function description", key="func_desc", height=68,
        value=get_or_reset('func_desc'),
        placeholder=unwrap("""
        Describe the purpose of this function. This description will
        be used to determine the context in which the function is
        called.
        """),
    )

    for i in range(1, number_of_features + 1):
        feat_name = get_or_reset(f"feat_name_{i}")
        st.subheader(
            f"Feature {i}" + (f": {feat_name}" if feat_name else ""),
            divider=True)

        subcol1, subcol2, subcol3 = st.columns(3)
        with subcol1:
            st.text_input(
                "Feature name", key=f"feat_name_{i}",
                value=feat_name,
                placeholder="lowercase_with_underscores")
            if not feat_name:
                st.error('A name is required')

        with subcol2:
            feat_type = get_or_reset(f"feat_type_{i}", "string")
            feat_type_options = [t.value for t in ParameterType]
            st.selectbox(
                "Feature type", key=f"feat_type_{i}",
                index=feat_type_options.index(feat_type),
                options=feat_type_options)

        with subcol3:
            st.toggle(
                "Required", key=f"feat_required_{i}",
                value=get_or_reset(
                    f"feat_required_{i}",
                    condition=not getval(f"feat_required_{i}_changed")
                ),
                on_change=on_click_required_changed, args=(i,))

        if getval(f"feat_type_{i}") == ParameterType.STRING.value:
            st.text_input(
                "Enum values", key=f"feat_enum_{i}",
                value=get_or_reset(f"feat_enum_{i}"),
                placeholder="Comma-separated list of values")

        if getval(f"feat_type_{i}") == ParameterType.ARRAY.value:
            items_type_options = [
                t.value for t in ParameterType if t is not ParameterType.ARRAY]
            items_type = get_or_reset(f"items_type_{i}", "number")
            st.selectbox(
                "Array items type", key=f"feat_array_{i}",
                index=items_type_options.index(items_type),
                options=items_type_options)

        feat_desc = st.text_area(
            "Feature description",
            key=f"feat_description_{i}", height=68,
            value=get_or_reset(f"feat_description_{i}"),
            placeholder=unwrap(
                """Describe the feature to be extracted into this field.
                """))
        if not feat_desc:
            st.error('A description is required')

with col2:
    # assemble the tool specification
    properties = {}
    required = []
    for i in range(1, number_of_features + 1):
        property = {
            "type": getval(f"feat_type_{i}"),
            "description": getval(f"feat_description_{i}")
        }
        if enum_vals := getval(f"feat_enum_{i}"):
            property["enum"] = list(
                set(s.strip() for s in enum_vals.split(",")))
        if items_type := getval(f"feat_array_{i}"):
            property["items"] = {"type": items_type}

        feat_name = getval(f"feat_name_{i}") or f"feat_name_{i}"
        properties[feat_name] = property
        if getval(f"feat_required_{i}", False):
            required.append(feat_name)

    st.session_state['tool_spec'] = {
        "name": st.session_state.get("func_name", "function_name"),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        },
        "type": "function",
        "description": st.session_state.get("func_desc")
    }

    # display the pretty-printed value of tool_spec
    if features := getval('features'):
        st.dataframe(
            features,
            width='stretch',
        )

    if response := st.session_state.get('response'):
        if st.toggle("Show API response"):
            response_json = json.dumps(response, indent=2)
            st.markdown(f"```json\n{response_json}\n```")

    if st.toggle("Show tool specification"):
        spec_json = json.dumps(st.session_state['tool_spec'], indent=2)
        st.markdown(f"```json\n{spec_json}\n```")

    if getval("func_name") and getval("tool_spec"):
        st.download_button(
            f"Download {func_name}.json",
            data=json.dumps(st.session_state['tool_spec'], indent=2),
            mime="application/json",
            file_name=f"{func_name}.json")

    st.file_uploader(
        "Upload a JSON file containing a tool specification.",
        key="uploaded_file", type="json", on_change=set_uploaded_data)
