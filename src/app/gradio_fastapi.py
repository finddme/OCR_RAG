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
import gradio as gr 
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

async def get_response(query,file,file_name):
    url = "http://192.168.2.186:8789/OCR_RAG"
    query_params ={"user_input": query}
    file_name=file_name.split("/")[-1]
    file_path=f"./pdf_examples/{file_name}"
    files = {
        'file': (file_name, open(file_path, 'rb'))
    }
    response = requests.post(url, params=query_params, files=files)
    # print(response.status_code)
    return response.json()

async def main_process(user_input,uploaded_file):
    file_name=uploaded_file.name
    file_name=file_name.split("/")[-1]
    # with open("tt.txt","w") as f:
    #     f.write(file_name)
    exist_file_list = glob.glob("./pdf_examples/"+ "*.pdf")
    exist_file_list=list(map(lambda x: x.split("/")[-1], exist_file_list))
    file_location = f"./pdf_examples/{file_name}"
    if file_name not in exist_file_list:
        print("--- Save Input File ---")
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as f:
            f.write(uploaded_file) 
    response = await get_response(user_input,uploaded_file,file_location)

    answer=response["response"]
    vis_res= await bbox_visualisation(response)

    return answer,vis_res


# os.environ["GRADIO_TEMP_DIR"]="./pdf_examples/"
with gr.Blocks() as demo:
    # gr.Markdown(DESCRIPTION)
    with gr.Row(equal_height=True):
        with gr.Column(scale=0.5):
            user_input = gr.Textbox(label="Query")
        with gr.Column(scale=0.5):
            with gr.Row():
                uploaded_file = gr.File(label="Input pdf",height=0.1)
                saved_class_list, class_name_list, file_name_list=load_weaviate_class_list()
                gr.Examples(
                    examples=[
                        # os.path.join("pdf_examples", f"{fnl}.pdf") for fnl in file_name_list
                        os.path.join("pdf_examples", img_name) for img_name in sorted(os.listdir("pdf_examples"))
                    ],
                    inputs=[uploaded_file],
                    label="Examples",
                    cache_examples=False,
                    examples_per_page=12,
                )
                # print("111111111111111111111111111",str(uploaded_file))
    with gr.Row():
        run_button = gr.Button()
    with gr.Row():
        answer=gr.Markdown(label="Output")
        # vis_res = gr.Image(label="Result", show_label=False,multiple=True)
        vis_res = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery",
            object_fit="contain", height="auto")
    run_button.click(main_process, inputs=[user_input,uploaded_file], outputs=[answer,vis_res])

demo.queue()
demo.launch(server_name = "0.0.0.0",share=True,server_port=8986)