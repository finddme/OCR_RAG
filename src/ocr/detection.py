from __future__ import annotations
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
from ocr.controller import *

def get_detections(line: dict[str, Any],pdf_page,main_box,img_height,img_width,page_number,file_path) -> list[dict[str, Any]]:
    def draw_bbox(img, start_point, end_point, ratio=1):
        start_point = tuple(map(lambda x: round(x * ratio), start_point))
        end_point = tuple(map(lambda x: round(x * ratio), end_point))
        return start_point,end_point
    detections = []
    table_check_flag=False
    if pdf_page is not None:
        bbox_converted,pdf_x0 = bbox_control(pdf_page,main_box,img_width,img_height)

        text=""
        # table check ################################################
        tables = camelot.read_pdf(file_path, pages=str(page_number), backend="poppler")
        for table in tables:
            if table.df[0][0]!="":
                table_bbox=table._bbox
                t_x0, t_y0, t_x1, t_y1 = map(float, table_bbox)
                diff = abs(t_x0+10 - pdf_x0)

                img = table._image[0]
                # ratio = 298 / 72
                ratio=0.895
                new_tmp_img = copy.deepcopy(img)
                pdf_height = img.shape[0] / ratio
                start_point,end_point=draw_bbox(new_tmp_img,
                        start_point=(t_x0, pdf_height - t_y0),
                        end_point=(t_x1, pdf_height - t_y1),
                        ratio=ratio)

                diff= abs(start_point[0]-pdf_x0)
                
                if diff < 3:
                    # print(page_number,diff)
                    text+=table.df.to_markdown(index=False)
                    table_check_flag=True

        if table_check_flag==False:
            text+=pdf_page.get_textbox(bbox_converted)

        detections.append({"id": 0, "text": text, "confidence": "", "box": main_box, "bbox_converted":bbox_converted})

    else: 
        results = line.get("res")
        if results is not None:
            for i, result in enumerate(results):
                text = result.get("text")

                confidence = result.get("confidence")
                box = result.get("text_region")
                x_min, y_min = box[0]
                x_max, y_max = box[2]
                new_box = [x_min, y_min, x_max, y_max]

                detections.append({"id": i, "text": text, "confidence": confidence, "box": new_box, "bbox_converted":()})
    return detections,table_check_flag

# Safely attempt to extract the HTML table from the results
def get_table_markdown(line: dict[str, Any],box,pdf_page,img_height,img_width,pdf_doc) -> str:
    print("============Table============")
    bbox_converted,pdf_x0 = bbox_control(pdf_page,box,img_width,img_height)

    # extract ###############################################
    image_list = pdf_page.get_images(full=True)

    box_list=[]
    for img_index, img in enumerate(image_list):
        image_bbox = pdf_page.get_image_bbox(image_list[img_index])
        x0, y0, x1, y1 = map(float, image_bbox)
        box_list.append({"box":[x0, y0, x1, y1],"idx":img_index})

    detections=[table_location_control(box_list,image_list,pdf_page,pdf_doc,pdf_x0)]

    return detections