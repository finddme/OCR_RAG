from ocr.ocr_core import detect_and_log_layouts, PAGE_LIMIT
import re, sys
from PIL import Image, ImageDraw
import pdfplumber
from threading import Thread
import weaviate
import json,os,random
import string
from tqdm import tqdm
class_name_path="/".join(__file__.split("/")[:-1])+"/DB_class_name.txt"
client = weaviate.Client("http://192.168.0.202:8080") 

import openai
from langchain_community.embeddings.openai import OpenAIEmbeddings

def get_embedding_openai(text, engine="text-embedding-3-large") : 
    Openai_API_KEY = ""
    os.environ["OPENAI_API_KEY"] =  Openai_API_KEY
    openai.api_key =os.getenv("OPENAI_API_KEY")

    # res = openai.Embedding.create(input=text,engine=engine)['data'][0]['embedding']
    from openai import OpenAI
    embedding_client = OpenAI()
    res= embedding_client.embeddings.create(input = text, model=engine).data[0].embedding
    return res

def load_weaviate_class_list():
    saved_class_list=[]
    with open(class_name_path, 'r') as ff:
        class_data=ff.readlines()
        for cd in class_data:
            saved_class_list.append({"class":cd.split("/////")[0],"file":cd.split("/////")[1].replace("\n","")})
    class_name_list=list(dict.fromkeys([cdl['class'] for cdl in saved_class_list]))
    file_name_list=list(dict.fromkeys([cdl['file'] for cdl in saved_class_list]))
    return saved_class_list,class_name_list, file_name_list

def save_weaviate(ocr_result,pdf_path):
    global client
    print("--- Save DB ---")
    class_name="".join(random.choice(string.ascii_lowercase) for _ in range(8))
    class_obj = {
        "class": class_name,
        "vectorizer": "none",
    }
    pdf_temp_name=pdf_path.split("/")[-1].rstrip(".pdf")
    temp_save=f"{class_name}/////{pdf_temp_name}"
    with open(class_name_path, 'a') as ff:
        ff.write(f"{temp_save}\n")
    # Add the class to the schema
    client.schema.create_class(class_obj)
    for d in tqdm(ocr_result):
        page_num=d["page_num"]
        for dr in d["detection_res"]:
            layout_type=dr["layout_type"]
            content=dr["text"]
            bbox="//".join([str(i) for i in dr["bbox"][0]])
            box_color="//".join([str(i) for i in dr["box_color"]])
            title=dr["title"]
            data={"page_num":page_num,"layout_type":layout_type,"content":content,
                "bbox":bbox,"box_color":box_color,"title":title}
            embedding_target=""
            if len(title)!=0:
                embedding_target+=f"[{title}] {content}"
            else:embedding_target += content
            embed = get_embedding_openai(content)
            with client.batch as batch:
                batch.add_data_object(data_object=data, class_name=class_name, vector=embed)
    print("--- vecotr DB store complete ---")
    return class_name

def del_weaviate_class(file_name):
    global client
    saved_class_list, class_name_list, file_name_list=load_weaviate_class_list()
    del_class_name=[cdl['class'] for cdl in saved_class_list if cdl['file'] ==file_name]
    for dcn in del_class_name:
        print(f"--- delte db class {dcn} ---")
        client.schema.delete_class(dcn)

def db_class_sync_check():
    global client
    print("--- Synchronizing vecotr DB ---")
    db_classes=list(map(lambda x : x["class"].lower(), client.schema.get()["classes"]))
    _, class_name_list, _=load_weaviate_class_list()
    for dc in db_classes:
        if dc not in class_name_list:
            print(f"--- delte db class {dc} ---")
            client.schema.delete_class(dc)