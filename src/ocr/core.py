from __future__ import annotations
import copy
import logging
import os,re
from enum import Enum
from pathlib import Path
from queue import SimpleQueue
# import os,sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from ocr.layout import *

EXAMPLE_DIR: Final = Path(os.path.dirname(__file__))
DATASET_DIR: Final = EXAMPLE_DIR / "dataset"

PAGE_LIMIT = 10

def detect_and_log_layouts(file_path: str, start_page: int = 1, end_page: int | None = -1) -> None:
    if end_page == -1:
        end_page = start_page + PAGE_LIMIT-1
    if end_page < start_page:
        end_page = start_page
    print(start_page, end_page)

    images: list[npt.NDArray[np.uint8]] = []
    if file_path.endswith(".pdf"):
        # convert pdf to images
        images.extend(np.array(img, dtype=np.uint8) for img in pdf2image.convert_from_path(file_path, first_page=start_page, last_page=end_page))
        print(len(images))

        PDF_FOR_TXT_EXTRACTION = pymupdf.open(file_path)
    else:
        # read image
        PDF_FOR_TXT_EXTRACTION_page=None
        img = cv2.imread(file_path)
        coloured_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(coloured_image.astype(np.uint8))

    # Extracte the layout from each image
    layouts: list[Layout] = []
    page_numbers = [i + start_page for i in range(len(images))]
    for i, (image, page_number) in enumerate(zip(images, page_numbers)):
        if PDF_FOR_TXT_EXTRACTION:
            PDF_FOR_TXT_EXTRACTION_page = PDF_FOR_TXT_EXTRACTION.load_page(page_number-1)
            layouts.append(detect_and_log_layout(image, page_number,PDF_FOR_TXT_EXTRACTION_page,PDF_FOR_TXT_EXTRACTION,file_path))
        else:
            layouts.append(detect_and_log_layout(image, page_number,PDF_FOR_TXT_EXTRACTION_page,PDF_FOR_TXT_EXTRACTION,file_path))

    return layouts

def detect_and_log_layout(coloured_image: npt.NDArray[np.uint8], page_number: int, pdf_page,pdf_doc,file_path) -> Layout:
    # Layout Object - This will contain the detected layouts and their detections
    layout = Layout(page_number)
    page_path = f'page_{page_number}'
    print("page_number",page_number)
    # Paddle Model - Getting Predictions
    logging.info("Start detection... (It usually takes more than 10-20 seconds per page)")
    ocr_model_pp = PPStructure(show_log=False, recovery=True)
    logging.info("model loaded")
    result_pp = ocr_model_pp(coloured_image)
    # print("result_pp", result_pp)
    _, w, _ = coloured_image.shape
    result_pp = sorted_layout_boxes(result_pp, w)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1result_pp2", result_pp)
    logging.info("Detection finished...")

    # Add results to the layout
    img_height,img_width, _=coloured_image.shape
    layout.save_all_layouts(result_pp,pdf_page,img_height,img_width,pdf_doc,page_number,file_path)
    logging.info("All results are saved...")

    return layout