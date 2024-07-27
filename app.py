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

# print(sample_prompt)

# Initialize session state variables
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "result" not in st.session_state:
    st.session_state.result = None


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read().decode('utf-8'))


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
        model="gpt-4-vision-preview",
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
