import copy
import logging
import os,re
from enum import Enum
from pathlib import Path
from queue import SimpleQueue
# from typing import Any, Final, Iterable, Optional, TypeAlias
from typing import Any, Final, Iterable, Optional, Tuple, List
from pytesseract import pytesseract
import cv2 as cv2
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore
import pdf2image  # type: ignore
from paddleocr import PPStructure  # type: ignore
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes  # type: ignore
import pymupdf
from PIL import Image
import json
import cv2
import random
import string
import camelot

def bbox_control(pdf_page,box,img_width,img_height):
    x0, y0, x1, y1 = map(float, box)
    page_height = pdf_page.rect.height
    page_width = pdf_page.rect.width
    # print("page_height,page_width",page_height,page_width,img_height,img_width)

    scale_width = page_width / img_width
    scale_height = page_height / img_height

    pdf_x0 = x0 * scale_width
    pdf_y0 = y0 * scale_height
    pdf_x1 = x1 * scale_width
    pdf_y1 = y1 * scale_height

    bbox_converted = (x0, page_height - y1, x1, page_height - y0)
    bbox_converted = (pdf_x0, pdf_y0, pdf_x1, pdf_y1)
    return bbox_converted,pdf_x0


def table_location_control(box_list,image_list,pdf_page,pdf_doc,pdf_x0):
    closest_list = None
    min_diff = float('inf')
    text_from_pdf=""
    for b_lst in box_list:
        diff = abs(b_lst["box"][0] - pdf_x0)
        if diff < min_diff:
            min_diff = diff
            closest_list = b_lst
    if closest_list!=None:
        img=image_list[closest_list["idx"]]
        image_bbox = pdf_page.get_image_bbox(img)
        x0, y0, x1, y1 = map(float, image_bbox)
        xref = img[0]
        base_image = pdf_doc.extract_image(xref)
        image_bytes = base_image["image"]
        file_name="".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        save_file= f"./pngs/{file_name}.png"
        with open(save_file, "wb") as image_file:
            image_file.write(image_bytes)
        table_image = Image.open(save_file)
        table_image = table_image.resize((400,200))
        text_res = pytesseract.image_to_string(table_image, lang='kor')
        s_check=re.compile(r"\n\n\s{3,}")
        text_res=re.sub(s_check," ",text_res)
        text_from_pdf+=text_res
        return {"id": 0, "text": text_from_pdf, "confidence": "", "box": box, "bbox_converted":bbox_converted},
    else: return {"id": 0, "text": " ", "confidence": "", "box": [0,0,0,0], "bbox_converted":(0,0,0,0)}


