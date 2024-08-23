from ocr.core import detect_and_log_layouts, PAGE_LIMIT
import re
from PIL import Image, ImageDraw
import pdfplumber
from threading import Thread
import weaviate
import json,os,random
import string
from tqdm import tqdm
import openai
from langchain_community.embeddings.openai import OpenAIEmbeddings
from rag.db_management import (get_embedding_openai,
                        load_weaviate_class_list, 
                        save_weaviate, 
                        del_weaviate_class,
                        db_class_sync_check)

extracted_txt=[]
full_text = ""
result=[]
final_res=[]

def file_ocr(file_path: str, start_page: int, end_page: int):
    layouts=detect_and_log_layouts(file_path, start_page, end_page)
    rere=re.compile("\(recording.{0,100}?\)")
    global extracted_txt
    global full_text
    global final_res
    global result
    extracted_txt.clear()
    full_text=""
    result.clear()
    final_res.clear()
    collection_record_list=[]
    
    for idx,l in enumerate(layouts):
        resres={"page_num":l.page_number,"detection_res":[]}
        for l_idx,(layout_type,record_list) in enumerate(l.records.items()):
            collection_record_list.append(record_list)
            for rec_idx,record_res in enumerate(record_list):
                res={"layout_type":layout_type.value[1],"text":"","bbox":[],"box_color":layout_type.value[2]}
                for detection in record_res["detections"]:
                    res["text"]+=detection["text"].replace("�","").replace("Ÿ","")
                    res["bbox"].append(list(detection["bbox_converted"]))     
                    
                resres["detection_res"].append(res)
        result.append(resres)

    for fr in result:
        text_text = ""
        for dr in fr["detection_res"]:
            if dr["layout_type"]=="text":
                text_text += dr['text']

        for dr in fr["detection_res"]:
            if dr['layout_type'] == 'table':
                dr['text'] += "\n\n" + text_text
    for fr in result:
        title = ""
        fin_resres={"page_num":fr["page_num"],"detection_res":[]}
        for dr in fr["detection_res"]:
            if dr["layout_type"]=="title":
                title=dr['text']
            elif dr["layout_type"]=="text" or dr["layout_type"]=="table":
                if len(dr["text"])>55: # text minimum length
                    dr["title"] = title
                    fin_resres["detection_res"].append(dr)
        final_res.append(fin_resres)
                
    print("--- OCR done ---")

def threading_ocr_process(file_path: str, start_page: int, end_page: int):
    handle = Thread(target=file_ocr, args=[str(file_path), start_page, end_page])
    handle.start()
    handle.join()

def ocr_processing(file_name,start_page, end_page):
    global final_res
    db_class_sync_check()
    pdf_path=f"./pdf_examples/{file_name}"
    # file_name=pdf_path.split("/")[-1].rstrip(".pdf")
    file_name=file_name.rstrip(".pdf")

    saved_class_list, class_name_list, file_name_list=load_weaviate_class_list()
    print("--- check DB ---")
    if file_name in file_name_list:
        class_name = list(map(lambda x: x['class'], filter(lambda x: x['file']==file_name, saved_class_list)))
        print(f"--- PDF exist, calss name: {class_name}---")
        return class_name[0]
    else:
        print(f"--- PDF not exist, OCR start---")
        threading_ocr_process(pdf_path, start_page, end_page)
        # with open(output_path, 'w', encoding='utf-8') as outfile:
        #     json.dump(final_res, outfile,indent="\t",ensure_ascii=False)
        class_name=save_weaviate(final_res,pdf_path)
        print(f"--- weaviate save complete, calss name: {class_name}---")
        return class_name


