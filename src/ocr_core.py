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

EXAMPLE_DIR: Final = Path(os.path.dirname(__file__))
DATASET_DIR: Final = EXAMPLE_DIR / "dataset"

SAMPLE_IMAGE_URLs = ["https://storage.googleapis.com/rerun-example-datasets/ocr/paper.png"]

PAGE_LIMIT = 10

# LayoutStructure = Tuple[
#     List[str], List[str], List[rrb.Spatial2DView], List[rrb.Spatial2DView], List[rrb.Spatial2DView]
# ]
# Supportive Classes

class Color:
    Red = (255, 0, 0)
    Green = (0, 255, 0)
    Blue = (0, 0, 255)
    Yellow = (255, 255, 0)
    Cyan = (0, 255, 255)
    Magenta = (255, 0, 255)
    Purple = (128, 0, 128)
    Orange = (255, 165, 0)


"""
LayoutType:
    Defines an enumeration for different types of document layout elements, each associated with a unique number, name,
    and color. Types:
    - UNKNOWN: Default type for undefined or unrecognized elements, represented by purple.
    - TITLE: Represents the title of a document, represented by red.
    - TEXT: Represents plain text content within the document, represented by green.
    - FIGURE: Represents graphical or image content, represented by blue.
    - FIGURE_CAPTION: Represents captions for figures, represented by yellow.
    - TABLE: Represents tabular data, represented by cyan.
    - TABLE_CAPTION: Represents captions for tables, represented by magenta.
    - REFERENCE: Represents citation references within the document, also represented by purple.
    - Footer: Represents footer of the document, represented as orange.
"""
 

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
        # if page_number==5:
        #     for item in res:
        #         with open ("./ex.txt","a",encoding="utf-8") as e:
        #             e.write(str(item['detections'][0]['text'])+"\n\n====================================================")
        
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
            # table = self.get_table_markdown(line,box,pdf_page,img_height,img_width,pdf_doc)
            detections = self.get_table_markdown(line,box,pdf_page,img_height,img_width,pdf_doc)
        elif layout_type == LayoutType.FIGURE:
            detections, table_check_flag = self.get_detections(line,pdf_page,box,img_height,img_width,page_number,file_path)
            img = line.get("img")  # Currently not in use
        else:
            detections, table_check_flag = self.get_detections(line,pdf_page,box,img_height,img_width,page_number,file_path)
            if table_check_flag==True:layout_type=LayoutType.TABLE
        # detections=dict.fromkeys(detections)
        # self.add(pdf_page,layout_type, box, detections=detections, table=table, figure=img)
        return layout_type, box, detections, table, img

    @staticmethod
    def get_detections(line: dict[str, Any],pdf_page,main_box,img_height,img_width,page_number,file_path) -> list[dict[str, Any]]:
        def draw_bbox(img, start_point, end_point, ratio=1):
            start_point = tuple(map(lambda x: round(x * ratio), start_point))
            end_point = tuple(map(lambda x: round(x * ratio), end_point))
            return start_point,end_point
        detections = []
        table_check_flag=False
        if pdf_page is not None:
            # bbox control ################################################
            x0, y0, x1, y1 = map(float, main_box)
            page_height = pdf_page.rect.height
            page_width = pdf_page.rect.width
            # print("page_height,page_width",page_height,page_width,img_height,img_width)

            scale_width = page_width / img_width
            scale_height = page_height / img_height

            pdf_x0 = x0 * scale_width
            pdf_y0 = y0 * scale_height
            pdf_x1 = x1 * scale_width
            pdf_y1 = y1 * scale_height
            # bbox_converted = (x0, page_height - y1, x1, page_height - y0)
            bbox_converted = (pdf_x0, pdf_y0, pdf_x1, pdf_y1)

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
                    
                                        # with open("./ex.txt","a",encoding="utf-8")as e:
                    #     e.write(f"\n\n\n===================={x0}///{pdf_x0}\n{diff}====================")
                    # print(f"\n\n\n===================={t_x0} /// {t_x0 * scale_width} /// {pdf_x0}====================")
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
    @staticmethod
    def get_table_markdown(line: dict[str, Any],box,pdf_page,img_height,img_width,pdf_doc) -> str:
        print("============Table============")
        detections=[]
        x0, y0, x1, y1 = map(float, box)
        page_height = pdf_page.rect.height
        page_width = pdf_page.rect.width

        scale_width = page_width / img_width
        scale_height = page_height / img_height

        pdf_x0 = x0 * scale_width
        pdf_y0 = y0 * scale_height
        pdf_x1 = x1 * scale_width
        pdf_y1 = y1 * scale_height

        bbox_converted = (x0, page_height - y1, x1, page_height - y0)
        bbox_converted = (pdf_x0, pdf_y0, pdf_x1, pdf_y1)

        # extract ###############################################
        text_from_pdf=""
        image_list = pdf_page.get_images(full=True)

        box_list=[]
        closest_list = None
        min_diff = float('inf')
        for img_index, img in enumerate(image_list):
            image_bbox = pdf_page.get_image_bbox(image_list[img_index])
            x0, y0, x1, y1 = map(float, image_bbox)
            box_list.append({"box":[x0, y0, x1, y1],"idx":img_index})

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
            detections.append({"id": 0, "text": text_from_pdf, "confidence": "", "box": box, "bbox_converted":bbox_converted})
        else: detections.append({"id": 0, "text": " ", "confidence": "", "box": [0,0,0,0], "bbox_converted":(0,0,0,0)})
        """
        try:
            html_table = line.get("res", {}).get("html")
            if not html_table:
                return "No table found."

            dataframes = pd.read_html(html_table)
            if not dataframes:
                return "No data extracted from the table."

            markdown_table = dataframes[0].to_markdown()
            return markdown_table  # type: ignore[no-any-return]

        except Exception as e:
            return f"Error processing the table: {str(e)}"
        """
        return detections

def update_zoom_paths(
    layout: Layout,
    layout_type: LayoutType,
    record: dict[str, Any],
    paths: list[str],
    page_path: str,
    zoom_paths: list[rrb.Spatial2DView],
    zoom_paths_figures: list[rrb.Spatial2DView],
    zoom_paths_tables: list[rrb.Spatial2DView],
    zoom_paths_texts: list[rrb.Spatial2DView],
) -> None:
    if layout_type in [LayoutType.FIGURE, LayoutType.TABLE, LayoutType.TEXT]:
        current_paths = paths.copy()
        current_paths.remove(f"-{page_path}/Image/{layout_type.type.title()}/{record['name'].title()}/**")
        bounds = rrb.VisualBounds2D(
            x_range=[record["bounding_box"][0] - 10, record["bounding_box"][2] + 10],
            y_range=[record["bounding_box"][1] - 10, record["bounding_box"][3] + 10],
        )

        # Add to zoom paths
        view = rrb.Spatial2DView(
            name=record["name"].title(), contents=[f"{page_path}/Image/**"] + current_paths, visual_bounds=bounds
        )
        zoom_paths.append(view)

        # Add to type-specific zoom paths
        if layout_type == LayoutType.FIGURE:
            zoom_paths_figures.append(view)
        elif layout_type == LayoutType.TABLE:
            zoom_paths_tables.append(view)
        elif layout_type != LayoutType.UNKNOWN or layout.show_unknown:
            zoom_paths_texts.append(view)


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
