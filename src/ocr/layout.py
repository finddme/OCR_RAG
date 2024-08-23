from __future__ import annotations
import copy
import logging
import os,re
from enum import Enum
from pathlib import Path
from queue import SimpleQueue
from typing import Any, Final, Iterable, Optional, Tuple, List
# import os,sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from ocr.detection import *

class Color:
    Red = (255, 0, 0)
    Green = (0, 255, 0)
    Blue = (0, 0, 255)
    Yellow = (255, 255, 0)
    Cyan = (0, 255, 255)
    Magenta = (255, 0, 255)
    Purple = (128, 0, 128)
    Orange = (255, 165, 0)

class LayoutType(Enum):
    UNKNOWN = (0, "unknown", Color.Purple)
    TITLE = (1, "title", Color.Red)
    TEXT = (2, "text", Color.Green)
    FIGURE = (3, "figure", Color.Blue)
    FIGURE_CAPTION = (4, "figure_caption", Color.Yellow)
    TABLE = (5, "table", Color.Cyan)
    TABLE_CAPTION = (6, "table_caption", Color.Magenta)
    REFERENCE = (7, "reference", Color.Purple)
    FOOTER = (8, "footer", Color.Orange)

    def __str__(self) -> str:
        return str(self.value[1])  # Returns the string part (type)

    @property
    def number(self) -> int:
        return self.value[0]  # Returns the numerical identifier

    @property
    def type(self) -> str:
        return self.value[1]  # Returns the type

    @property
    def color(self) -> tuple[int, int, int]:
        return self.value[2]  # Returns the color

    @staticmethod
    def get_class_id(text: str) -> int:
        try:
            return LayoutType[text.upper()].number
        except KeyError:
            logging.warning(f"Invalid layout type {text}")
            return 0

    @staticmethod
    def get_type(text: str) -> LayoutType:
        try:
            return LayoutType[text.upper()]
        except KeyError:
            logging.warning(f"Invalid layout type {text}")
            return LayoutType.UNKNOWN

    @classmethod
    def get_annotation(cls) -> list[tuple[int, str, tuple[int, int, int]]]:
        return [(layout.number, layout.type, layout.color) for layout in cls]

class Layout:
    def __init__(self, page_number: int, show_unknown: bool = False):
        self.counts = {layout_type: 0 for layout_type in LayoutType}
        self.records: dict[LayoutType, Any] = {layout_type: [] for layout_type in LayoutType}
        self.recovery = """"""
        self.page_number = page_number
        self.show_unknown = show_unknown

    def add(
        self,
        pdf_page,
        layout_type: LayoutType,
        bounding_box: list[int],
        detections: Optional[Iterable[dict[str, Any]]] = None,
        table: Optional[str] = None,
        figure: Optional[dict[str, Any]] = None,
    ) -> None:
        if layout_type in LayoutType:
            self.counts[layout_type] += 1
            name = f"{layout_type}{self.counts[layout_type]}"
            logging.info(f"Saved layout type {layout_type} with name: {name}")
            self.records[layout_type].append({
                "type": layout_type,
                "name": name,
                "bounding_box": bounding_box,
                "detections": detections,
                "table": table,
            })
            if layout_type != LayoutType.UNKNOWN or self.show_unknown:  # Discards the unknown layout types detections
                path = f"recording://page_{self.page_number}/Image/{layout_type.type.title()}/{name.title()}"
                self.recovery += f"\n\n## [{name.title()}]({path})\n\n"  # Log Type as Heading
                # Enhancement - Logged image for Figure type TODO(#6517)
                if layout_type == LayoutType.TABLE:
                    # if table:
                        # self.recovery += table  # Log details (table)
                    for index, detection in enumerate(detections):
                        self.recovery += f'{detection["text"]}'
                elif detections:
                    for index, detection in enumerate(detections):
                        path_text = f"recording://page_{self.page_number}/Image/{layout_type.type.title()}/{name.title()}/Detections/{index}"
                        self.recovery += f' [{detection["text"]}]({path_text})'  # Log details (text)
        else:
            logging.warning(f"Invalid layout type detected: {layout_type}")

    def get_count(self, layout_type: LayoutType) -> int:
        if layout_type in LayoutType:
            return self.counts[layout_type]
        else:
            raise ValueError("Invalid layout type")

    def get_records(self) -> dict[LayoutType, list[dict[str, Any]]]:
        return self.records

    def save_all_layouts(self, results: list[dict[str, Any]],pdf_page,img_height,img_width,pdf_doc,page_number,file_path) -> None:
        res=[]
        for line in results:
            layout_type, box, detections, table, img=self.save_layout_data(line,pdf_page,img_height,img_width,pdf_doc,page_number,file_path)
            res.append({"layout_type":layout_type, "box":box, "detections":detections, "table":table, "img":img})
            with open ("./ex.txt","a",encoding="utf-8") as e:
                e.write(f"{page_number}===============layout_type:{layout_type}, box:{box}, detections:{detections}, table:{table}, img:{img}")

        # 중복 제거
        unique_data_dict = {item['detections'][0]['text']: item for item in res}

        # 일부 중복 제거
        exclude_list=[]
        udd_check_list=list(unique_data_dict.keys())
        for e_idx,udd in enumerate(udd_check_list):
            if e_idx+1 != len(unique_data_dict) and len(udd_check_list[e_idx+1].split("\n")) >3:
                if udd_check_list[e_idx+1].split("\n")[2] in udd:
                    exclude_list.append(udd_check_list[e_idx+1])

        exclude_list=dict.fromkeys(exclude_list)
        for el in exclude_list:
            del unique_data_dict[el]

        unique_data_list = list(unique_data_dict.values())

        # bbox y0 기준 재정렬
        unique_data_list.sort(key=lambda x: x['detections'][0]['bbox_converted'][1])

        for udl in unique_data_list:
            self.add(pdf_page,udl["layout_type"], udl["box"], detections=udl["detections"], table=udl["table"], figure=udl["img"])
        for layout_type in LayoutType:
            logging.info(f"Number of detections for type {layout_type}: {self.counts[layout_type]}")

    def save_layout_data(self, line: dict[str, Any],pdf_page,img_height,img_width,pdf_doc,page_number,file_path) -> None:
        type = line.get("type", "empty")
        box = line.get("bbox", [0, 0, 0, 0])
        layout_type = LayoutType.get_type(type)
        detections, table, img = [], None, None
        if layout_type == LayoutType.TABLE:
            detections = get_table_markdown(line,box,pdf_page,img_height,img_width,pdf_doc)
        elif layout_type == LayoutType.FIGURE:
            detections, table_check_flag = get_detections(line,pdf_page,box,img_height,img_width,page_number,file_path)
            img = line.get("img")  # Currently not in use
        else:
            detections, table_check_flag = get_detections(line,pdf_page,box,img_height,img_width,page_number,file_path)
            if table_check_flag==True:layout_type=LayoutType.TABLE
        return layout_type, box, detections, table, img
