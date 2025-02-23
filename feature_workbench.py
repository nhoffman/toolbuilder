import os
from enum import Enum
import json

from openai import OpenAI, OpenAIError
import streamlit as st

import utils

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
        try:
            response = utils.get_features(
                client=getval("client"),
                context=getval('context'),
                prompt=getval('prompt'),
                tools=[getval('tool_spec')],
                model=getval('model', 'gpt-4o'),
                temperature=getval('temperature', 1.0),
                n=getval('n_choices', 1),
            )
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


def set_uploaded_data():
    if uploaded_file := st.session_state.get('uploaded_file'):
        uploaded = json.loads(uploaded_file.read())

        output = {}
        output['func_name'] = get_nested(
            uploaded, 'function', 'name')
        output['func_desc'] = get_nested(
            uploaded, 'function', 'description')

        properties = get_nested(
            uploaded, 'function', 'parameters', 'properties')

        required = set(get_nested(
            uploaded, 'function', 'parameters', 'required'))

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


def get_uploaded(key):
    """Return the value from uploaded_data, but only if the
    corresponding field is empty.

    """
    if st.session_state.get('uploaded_data') and not st.session_state.get(key):
        return st.session_state['uploaded_data'].get(key)


def load_examples():
    st.session_state["context"] = utils.example_context
    st.session_state["prompt"] = utils.example_prompt


def get_or_reset(key, default=None, condition=True):
    if val := get_uploaded(key):
        if key in st.session_state and condition:
            del st.session_state[key]
    else:
        val = getval(key, default)

    return val


def get_num_features():
    if (st.session_state.get('uploaded_data')
        and not getval('num_features_changed')):
        return st.session_state['uploaded_data']['num_features']
    else:
        getval('num_features')


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


with st.sidebar:
    st.title("Feature Workbench")
    st.write(
        "This app allows you to define a function specification and "
        "extract features from a document using OpenAI's function calling "
        "capabilities."
        )

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
            completion = getval('client').chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "What is the capital of France?"}
                ]
            )
            # st.write(completion.choices[0].message.content)
            st.success("API key is valid")
        except OpenAIError as e:
            st.error(e)


st.header('Feature extraction using OpenAI function calling')

# st.write(st.session_state.get('uploaded_data'))

with st.form("content_form"):
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

        c1, c2, c3 = st.columns(3)

        with c1:
            model = st.selectbox("Model", ['gpt-4o', 'gpt-4o-mini'], key="model")
        with c2:
            temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1, key="temperature")
        with c3:
            n_choices = st.number_input(
                "Number of choices",
                key='n_choices',
                value=1,
                min_value=1, max_value=10,
            )

        submitted = st.form_submit_button("Submit", on_click=submit_query)


st.button("Load Example Text", on_click=load_examples)


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
        if num_features := get_num_features():
            del st.session_state['num_features']
        else:
            num_features = getval('num_features', 1)

        number_of_features = st.number_input(
            "Number of features",
            key='num_features',
            value=num_features,
            min_value=1, max_value=20,
            on_change=on_click_num_features,
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

        feat_name = getval(f"feat_name_{i}") or f"feat_name_{i}"
        properties[feat_name] = property
        if getval(f"feat_required_{i}", False):
            required.append(feat_name)

    st.session_state['tool_spec'] = {
        "type": "function",
        "function": {
            "name": st.session_state.get("func_name", "function_name"),
            "description": st.session_state.get("func_desc"),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

    # display the pretty-printed value of tool_spec
    if features := getval('features'):
        st.dataframe(
            features,
            use_container_width=True,
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
