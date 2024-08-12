"""
command:
streamlit run streamlit_fastapi.py --server.port 8788

http://192.168.2.186:8788/
"""
import asyncio
import os
import sys
import streamlit as st
import requests
import random
import time
import glob
from pprint import pprint
import json
from retrieve_visualisation import bbox_visualisation
import uvicorn
from db_management import (load_weaviate_class_list, 
                        del_weaviate_class
                        )
async def get_response(query,file_path):
    url = "http://192.168.2.186:8789/OCR_RAG"
    query_params ={"user_input": query}
    file_name=file_path.split("/")[-1]
    files = {
        'file': (file_name, open(file_path, 'rb'))
    }
    response = requests.post(url, params=query_params, files=files)
    # print(response.status_code)
    return response.json()

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 50px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# st.sidebar.button('Clear History', on_click=clear_chat_history)
            
async def try_sample(sample):
    file_name=sample
    file_location=f"./pdf_examples/{file_name}"
    uploaded_file=None
    return file_location,uploaded_file

async def main_process(user_input,sample,uploaded_file):
    saved_class_list, class_name_list, file_name_list=load_weaviate_class_list()
    sample_op=list(map(lambda x : f"{x}.pdf", file_name_list))
    if user_input: 
        with st.spinner("Generating response..."):
            try:
                if uploaded_file is not None:
                    file_name=uploaded_file.name
                    file_location = f"./pdf_examples/{file_name}"
                    exist_file_list = glob.glob("./pdf_examples/"+ "*.pdf")
                    exist_file_list=list(map(lambda x: x.split("/")[-1], exist_file_list))
                    if file_name not in exist_file_list:
                        print("--- Save Input File ---")
                        file_location = f"./pdf_examples/{file_name}"
                        os.makedirs(os.path.dirname(file_location), exist_ok=True)
                        file = uploaded_file.read()
                        with open(file_location, "wb") as f:
                            f.write(file) 
                elif sample is not None:
                    file_location,uploaded_file= await try_sample(sample[0])
                response = await get_response(user_input,file_location)

                answer=response["response"]
                vis_res= await bbox_visualisation(response)
                st.write(str(answer))
                st.image(vis_res, use_column_width=True, caption=["reference page"] * len(vis_res))
            
            except Exception as e:
                st.error(f"Error: {e}")

async def run_convo():
    saved_class_list, class_name_list, file_name_list=load_weaviate_class_list()
    sample_op=list(map(lambda x : f"{x}.pdf", file_name_list))
    st.title("""OCR-RAG""")
    with st.form(key="form"):
        user_input = st.text_input('검색어를 입력하세요.')
        file_flag=False
        sample = st.multiselect(
            label="sample PDF",
            options=sample_op,
            max_selections=1,
        )
        uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
        submit = st.form_submit_button(label="Submit")
        if submit:
            await main_process(user_input,sample,uploaded_file)

if __name__ == '__main__':
    asyncio.run(run_convo())

