import base64
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.helper import sample_prompt

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# Initialize session state variables
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "result" not in st.session_state:
    st.session_state.result = None


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_gpt4_model_for_analysis(file_name: str, sample_prompt=sample_prompt):
    base64_image = encode_image(file_name)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": sample_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        max_tokens=1500
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content


st.title("Medical Diseases Analysis")

with st.expander("About this application"):
    st.write("Upload an image to get an analysis from GPT-4 vision model")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=['jpg', 'jpeg', 'png']
)

# Temporary file handling
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state["filename"] = tmp_file.name

    st.image(uploaded_file, caption='Uploaded Image')

# Process Button
if st.button("Analyze Image"):
    if (
        'filename' in st.session_state and
        os.path.exists(st.session_state['filename'])
    ):
        st.session_state['result'] = call_gpt4_model_for_analysis(
            st.session_state['filename']
        )
        st.markdown(st.session_state['result'], unsafe_allow_html=True)
        os.unlink(st.session_state['filename'])  # Del tempfile after process
