from ocr.ocr_processing import ocr_processing
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from typing import List
from langchain_core.prompts import ChatPromptTemplate
import weaviate, os
import openai
import pdfplumber
import io, json
import base64
from io import BytesIO
import base64,io
from PIL import Image, ImageDraw

async def page_to_img(pdf_path, retrieval_res):
    pdf_path=f"./pdf_examples/{pdf_path}"
    page_num_list=[]
    pages=[]

    for ra in retrieval_res["annotations"]:
        page_num_list.append(ra["page_num"])
    page_num_list=list(dict.fromkeys(page_num_list))

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            for pn in page_num_list:
                if page_num+1== pn:
                    im = page.to_image()
                    img_org = im.original
                    img_byte_arr = io.BytesIO()
                    img_org.save(img_byte_arr, format='PNG')
                    png_data = img_byte_arr.getvalue() # PNG됐다

                    # PNG data to base64
                    img_org.seek(0)
                    img_base64 = base64.b64encode(png_data).decode('utf-8')
                    
                    """api 호출해서 image display할 때 참고
                    # 1) decode base64 image and 2)Converting binary data to PIL image
                    img_data = base64.b64decode(base64_str)
                    img = Image.open(io.BytesIO(img_data))
                    """
                    pages.append({"page_num":pn, "page":img_base64})
    return pages

async def bbox_visualisation(response):
    vis_res=[]
    for page in response["pages"]:
        img_data = base64.b64decode(page["page"])
        page_img = Image.open(io.BytesIO(img_data))
        # display(page_img)
        for annotation in response["annotations"]:
            if annotation["page_num"]==page["page_num"]:
                bbox=list(map(lambda x : float(x),annotation["bbox"]))
                box_color=tuple(map(lambda x : int(x),annotation["box_color"]))
                # page_img.draw_rect(bbox, stroke=box_color, stroke_width=3)
                draw = ImageDraw.Draw(page_img)
                draw.rectangle(bbox, outline=box_color, width = 3)
        vis_res.append(page_img)
    return vis_res

async def comprehensive_vis_process(pdf_path, retrieval_res):
    pdf_path=f"./pdf_examples/{pdf_path}"
    bboxes={}
    bbox_vis=[]
    # yield json.dumps({
    #     "response": retrieval_res["response"],
    #     "annotations": retrieval_res["annotations"]
    #     })
    for ra in retrieval_res["annotations"]:
        bbox=list(map(lambda x : float(x),ra["bbox"]))
        box_color=tuple(map(lambda x : int(x),ra["box_color"]))
        if ra["page_num"] not in list(bboxes.keys()):
            bboxes[ra["page_num"]]=[{"bbox":bbox, "box_color":box_color}]
        else:bboxes[ra["page_num"]].append({"bbox":bbox, "box_color":box_color})

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            for b_page_num,b_infos in bboxes.items():
                if page_num+1== b_page_num:
                    im = page.to_image()
                    for bi in b_infos:
                        im.draw_rect(bi["bbox"], stroke=bi["box_color"], stroke_width=1)
                    # im.save(f"./img{page_num}.png")
                    # PageImage to PNG
                    pil_image = im.original
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    png_data = img_byte_arr.getvalue() # PNG됐다

                    # PNG data to base64
                    pil_image.seek(0)
                    img_base64 = base64.b64encode(png_data).decode('utf-8')

                    bbox_vis.append(img_base64)
    return bbox_vis
