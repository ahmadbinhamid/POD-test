




import os
import json
import requests
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
from jinja2 import Template
import threading
from typing import List, Dict
from pydantic import BaseModel
import torch
from pathlib import Path
from fastapi import HTTPException
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import save_one_box
from io import BytesIO
import base64
import re
from langsmith import traceable
from collections import defaultdict, Counter
import copy
from dotenv import load_dotenv
import cv2
from doctr.io import DocumentFile
from doctr.models import page_orientation_predictor, mobilenet_v3_small_page_orientation
from doctr.models._utils import estimate_orientation
import httpx      # Added for asynchronous HTTP requests
import asyncio    # Added for async capabilities


# Import utility functions
from utils import (
    dynamic_preprocess,
    build_transform,
    convert_image_to_base64,
    save_image_to_tmp_dir,
    make_image_square,
    base64topil,
    preprocess_date,
    process_customer_numbers,
    adjust_date,
    resize_large_image
)
import models
import prompts

# Configuration
OCR_PORT = 3203
load_dotenv()


bol_region_detector = YOLO("Models/bol_regions_best.pt").to("cpu")
page_classifier = YOLO("Models/doc_classify_best.pt").to("cpu")
receipt_region_detector = YOLO("Models/receipts_regions_best.pt").to("cpu")
stamp_detector = YOLO("Models/stamp_existence.pt").to("cpu")

    # Initialize page orientation detection
page_det_model = mobilenet_v3_small_page_orientation(
        pretrained=False, 
        pretrained_backbone=False
    )
page_det_params = torch.load("Models/page_orientation.pt", map_location="cpu")
page_det_model.load_state_dict(page_det_params)
page_orient_predictor = page_orientation_predictor(
    arch=page_det_model, 
    pretrained=False
)
# Ensure temporary directory exists
os.makedirs("./tmp", exist_ok=True)

# Model mappings
region2model = {
    "carrier": models.CarrierInfo,
    "stamp": models.Stamp,
    "customer_order_info": models.CustomerOrderInfo,
    "signatures": models.Signatures,
    "bill_of_lading": models.BillOfLadings,
    "date": models.DeliveryReceipt,
}

# Prompt mappings
region2prompt = {
    "stamp": prompts.STAMP_LATEST_PROMPT,
    "customer_order_info": prompts.CUSTOMER_ORDER_INFO_PROMPT,
    "signatures": prompts.SIGNATURE_PROMPT,
    "bill_of_lading": prompts.BILL_OF_LADING_PROMPT,
    "receipt": prompts.DELIVERY_RECEIPT_PROMPT,
    "total_received": prompts.RECEIPT_TOTAL_RECEIVED_PROMPT,
    "damage": prompts.RECEIPT_DAMAGE_PROMPT,
    "refused": prompts.RECEIPT_REFUSED_PROMPT,
    "customer_order_num": prompts.RECEIPT_CUSTOMER_ORDER_NUMBER_PROMPT,
    "pod_date": prompts.RECEIPT_DATE_PROMPT,
    "pod_sign": prompts.RECEIPT_SIGNATURE_PROMPT,
}

@traceable(run_type="tool", name="straighten-img", dangerously_allow_filesystem=True)
async def straighten_img(img: Image.Image):
    """
    Straighten an image using document orientation detection
    Args:
        img: Base64 encoded image string
    Returns:
        dict: Containing straightened image, rotation angle, and confidence
    """
    # img=base64topil(img)
    # docs = DocumentFile.from_images(save_image_to_tmp_dir(img))
    docs = DocumentFile.from_images(save_image_to_tmp_dir(base64topil(img)))

    _, classes, probs = zip(page_orient_predictor(docs))
    page_orientations = [(cls, prob) for cls, prob in zip(classes[0], probs[0])]
    angle, confidence = page_orientations[0]
    img =base64topil(img)
    return {
        "img": convert_image_to_base64(
            img.rotate(angle, expand=True, resample=Image.BICUBIC)
        ),
        "angle": angle,
        "confidence": confidence,
    }

@traceable(run_type="tool", name="detect-regions-receipt", dangerously_allow_filesystem=True)
def receipt_detect_regions(image: Image) -> models.ReceiptRegions:
    """
    Detect and extract regions from receipt images
    Args:
        image: Input image to process
    Returns:
        ReceiptRegions: Object containing detected region images
    """
    # Define region mapping
    class_names = {
        0: "customer_order_num",
        1: "damage",
        2: "pod_date",
        3: "total_received",
        4: "refused",
        6: "pod_sign",  # Note: class 5 ('sched') is skipped
    }
    
    # Detect regions using YOLO model
    results: List[Results] = receipt_region_detector.predict(
        base64topil(image), 
        conf=0.8
    )
    
    # Process detected regions
    boxes = {}
    for result in results:
        for class_id, box in zip(result.boxes.cls.tolist(), result.cpu().boxes.xyxy):
            if class_id in list(class_names.keys()):
                boxes[class_names[class_id]] = convert_image_to_base64(
                    make_image_square(
                        Image.fromarray(
                            save_one_box(box, im=result.orig_img, save=False)
                        )
                    )
                )
    return models.ReceiptRegions(**boxes)

@traceable(run_type="tool", name="detect-regions-bol", dangerously_allow_filesystem=True)
def bol_detect_regions(image: Image) -> models.Regions:
    """
    Detect and extract regions from Bill of Lading (BOL) images
    Args:
        image: Input image to process
    Returns:
        Regions: Object containing detected region images
    """
    # Define region mapping
    class_names = {
        0: "stamp",
        1: "bill_of_lading",
        2: "customer_order_info",
        3: "signatures",
    }

    # Detect regions using YOLO model
    results: List[Results] = bol_region_detector.predict(base64topil(image), conf=0.51)
    boxes = {}
    
    for result in results:
        orig_img = result.orig_img
        h, w, _ = orig_img.shape

        for class_id, box in zip(result.boxes.cls.tolist(), result.cpu().boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            
            # Special handling for bill_of_lading region
            if class_names[class_id] == "bill_of_lading":
                x1, y1 = 0, 0  # Extend to top-left corner
                x2 = min(x2, w)  # Limit to image width
                y2 = min(y2, h)  # Limit to image height
            
            # Convert coordinates to tensor for save_one_box
            box_tensor = torch.tensor([x1, y1, x2, y2])
            cropped_img = save_one_box(box_tensor, im=orig_img, save=False)
            
            boxes[class_names[class_id]] = convert_image_to_base64(
                Image.fromarray(cropped_img)
            )
    
    return models.Regions(**boxes)

async def pdf_to_images(pdf_path_or_bytes):
    """
    Convert PDF file or bytes to list of PIL Images
    Args:
        pdf_path_or_bytes: Path to PDF file or PDF bytes
    Returns:
        list: List of PIL Images
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path_or_bytes)
    images = []
    
    for i in range(len(pdf)):
        page = pdf[i]
        # Render page at higher resolution
        rendered_image = page.render(scale=4).to_pil()
        # Resize if image is too large
        resized_image =resize_large_image(rendered_image)
        images.append(resized_image)
    
    return images

# Removed the individual `ocr` and `ocr_request` functions as `batch_ocr` will replace their use.
# @traceable(run_type="llm", name="OCR", dangerously_allow_filesystem=True)
# def ocr(image, question, port=None):
#     ...
# @traceable(run_type="llm", name="OCR", dangerously_allow_filesystem=True)
# async def ocr_request(ocr_batch: models.OCRBatch, port=None):
#     ...

@traceable(run_type="llm", name="OCR-Batch", dangerously_allow_filesystem=True)
async def batch_ocr(ocr_batches: List[models.OCRBatch], port=None) -> List[models.OCRBatchResponse]:
    """
    Process multiple OCR requests as a single batch using pixtral's batch inference endpoint.
    This replaces the previous client-side concurrency with a single HTTP batch request.
    """
    global OCR_PORT
    url = f"http://localhost:{port or OCR_PORT}/batch-pixtral-inference"

    # Prepare requests for the batch endpoint
    pixtral_requests = []
    # Store metadata to reconstruct OCRBatchResponse later
    metadata_list = []
    for ocr_batch in ocr_batches:
        # The pixtral service expects the base64 string without the "data:image/jpeg;base64," prefix
        image_data = ocr_batch.image.split(",")[1] if "," in ocr_batch.image else ocr_batch.image
        
        pixtral_requests.append(models.InterVL2Request(
            image_url_or_path_or_base64=image_data,
            prompt=ocr_batch.prompt
        ))
        metadata_list.append({
            "page_type": ocr_batch.page_type,
            "region_name": ocr_batch.region_name,
            "stamp_exist": ocr_batch.stamp_exist
        })

    if not pixtral_requests:
        return []

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            # Send a single POST request with a list of InterVL2Request objects
            response = await client.post(url, json=[req.model_dump() for req in pixtral_requests])
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # Parse the list of InterVL2Response objects from the batch response
        pixtral_batch_responses = [models.InterVL2Response(**item) for item in response.json()]

        final_ocr_responses = []
        for i, pixtral_resp in enumerate(pixtral_batch_responses):
            metadata = metadata_list[i]
            json_to_parse = pixtral_resp.response
            
            # Extract JSON content from response (same logic as original ocr_request)
            pattern = r"```json(.*?)```"
            match = re.search(pattern, json_to_parse, re.DOTALL)
            if match:
                json_to_parse = match.group(1)
            json_content = json_to_parse.strip("```json").strip("```").replace("'null'", "null")
            
            final_ocr_responses.append(
                models.OCRBatchResponse(
                    page_type=metadata["page_type"],
                    region_name=metadata["region_name"],
                    ocr_response=json.loads(json_content),
                    stamp_exist=metadata["stamp_exist"]
                )
            )
        return final_ocr_responses

    except httpx.HTTPStatusError as e:
        print(f"HTTP error during batch OCR: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Batch OCR service error: {e.response.text}"
        )
    except httpx.RequestError as e:
        print(f"Network error during batch OCR: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Batch OCR service unreachable or network error: {e}"
        )
    except json.JSONDecodeError as e:
        print(f"JSON decode error from Pixtral batch response: {e}. Raw response: {response.text}")
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON response from batch OCR service: {e}"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error in batch_ocr: {str(e)}")


def classify_documents(images):
    """
    Classify document types in images
    Args:
        images: List of base64 encoded images
    Returns:
        dict: Classified document regions by type
    """
    class_names = {0: "BOL", 1: "others", 2: "receipt"}
    processed_images = []

    print(f"start of classify _documents: {datetime.now()}")
    
    # Process and validate images
    for image in images:
        try:
            pil_image = base64topil(image)
            if pil_image is not None:
                pil_image = resize_large_image(pil_image)
                processed_images.append(pil_image)
        except Exception as e:
            print(f"Error processing image: {e}")
            continue
    
    if not processed_images:
        raise HTTPException(
            status_code=400,
            detail="No valid images to process after resizing"
        )
    
    # Classify document regions
    results: List[Results] = page_classifier.predict(processed_images, conf=0.55)
    boxes = defaultdict(list)
    
    for result in results:
        if result.boxes:
            for class_id, box in zip(
                result.boxes.cls.tolist(),
                result.cpu().boxes.xyxy
            ):
                if class_id in list(class_names.keys()):
                    boxes[class_names[class_id]].append(
                        convert_image_to_base64(
                            Image.fromarray(
                                save_one_box(box, im=result.orig_img, save=False)
                            )
                        )
                    )
    print(f"End of classify _documents: {datetime.now()}")

    return boxes

def preprocess_image(img):
    """
    Enhance image quality for better OCR results
    Args:
        img: PIL Image to process
    Returns:
        PIL Image: Processed image
    """
    image = np.array(img)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sharpened = cv2.morphologyEx(threshold, cv2.MORPH_GRADIENT, kernel)
    return Image.fromarray(sharpened)

def merge_bol_data(data):
    """
    Merge multiple BOL entries for the same bill number
    Args:
        data: List of BOL data entries
    Returns:
        list: Merged BOL data
    """
    merged_data = []
    
    def resolve_conflict(values):
        """Helper function to resolve conflicting values"""
        values_counter = Counter(values)
        return "null" if values_counter["null"] > values_counter["empty"] else "empty"

    # Group BOLs by bill number
    grouped = {}
    for bol in data:
        bill_no = bol["bill_of_lading"]["bill_no"]
        if bill_no not in grouped:
            grouped[bill_no] = []
        grouped[bill_no].append(bol)

    # Merge data for each bill number
    for bill_no, bol_list in grouped.items():
        merged = copy.deepcopy(bol_list[0])
        for key in merged["stamp"].keys():
            values = [bol["stamp"][key] for bol in bol_list]
            if all(value == values[0] for value in values):
                merged["stamp"][key] = values[0]
            else:
                non_null_or_empty = [v for v in values if v not in ["null", "empty"]]
                merged["stamp"][key] = (
                    non_null_or_empty[0] if non_null_or_empty 
                    else resolve_conflict(values)
                )
        merged_data.append(merged)

    return merged_data

async def bol_stamp_exist(image):
    """
    Detect if a stamp exists in the BOL image
    Args:
        image: Base64 encoded image
    Returns:
        str: 'yes' or 'no' indicating stamp presence
    """
    class_names = {0: "no", 1: "yes"}
    results = stamp_detector.predict(base64topil(image), conf=0.95)
    
    for result in results:
        classes = result.boxes.cls.tolist()
        if classes:
            return class_names[result.boxes.cls.tolist()[0]]

async def batch_list(data, batch_size):
    """
    Split data into batches of specified size
    Args:
        data: List to be batched
        batch_size: Size of each batch
    Returns:
        list: List of batches
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def group_by_page_type(data: List[models.OCRBatchResponse]) -> Dict[str, models.OCRBatchResponse]:
    """
    Group OCR responses by page type
    Args:
        data: List of OCR batch responses
    Returns:
        dict: Responses grouped by page type
    """
    grouped_data = defaultdict(list)
    for item in data:
        grouped_data[item.page_type].append(item)
    return dict(grouped_data)

def get_yesterday_date():
    """
    Get yesterday's date in MM/DD/YY format
    Returns:
        str: Formatted date string
    """
    yesterday = datetime.now() - timedelta(days=1)
    return f"{yesterday.month:02d}/{yesterday.day:02d}/{yesterday.year % 100:02d}"

@traceable(run_type="chain", name="run-ocr", dangerously_allow_filesystem=True)
async def run_ocr(file_url_or_path: str, batch_size=4, port=None) -> List[models.OCRResponse]:
    # Initialize ML Models
   
    
    try:
        print(f"start of process: {datetime.now()}")
        if os.path.isfile(str(file_url_or_path)):
            ext = os.path.splitext(file_url_or_path)[1].lower()
            if ext == ".pdf":
                pages =await pdf_to_images(file_url_or_path)
            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                with Image.open(file_url_or_path) as img:
                    with BytesIO() as output:
                        img.convert("RGB").save(output, format="JPEG")
                        output.seek(0)
                        img_jpg = Image.open(output).copy()
                pages = [img_jpg]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {ext}"
                )
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(file_url_or_path)
            # response = requests.get(file_url_or_path)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to fetch the file from URL: {file_url_or_path}",
                )
            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" in content_type:
                pages =await pdf_to_images(response.content)
            elif any(img_fmt in content_type for img_fmt in ["jpeg", "png", "bmp", "tiff", "webp"]):
                img = Image.open(BytesIO(response.content))
                with BytesIO() as output:
                    img.convert("RGB").save(output, format="JPEG")
                    output.seek(0)
                    img_jpg = Image.open(output).copy()
                pages = [img_jpg]
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported content type in URL."
                )
        print(f"start of overall_classified pages: {datetime.now()}")


        classified_pages = classify_documents(
            [convert_image_to_base64(page) for page in pages]
        )
        print(f"End of overall_classified pages: {datetime.now()}")

        
        ocr_output_pdf = defaultdict(list)
        all_requests = []
        for cls, cls_pages in classified_pages.items():
            if cls == "BOL":
                for page_idx, page in enumerate(cls_pages):
                    straighten_result =await straighten_img(page)
                    page =straighten_result["img"]
                    stamp_exist =await bol_stamp_exist(page)
                    bill_of_ladings_regions = bol_detect_regions(page).model_dump()
                    ocr_output_image = {}
                    #_____________________________________________________________________________________________________________________________________________________FIX-NEEDED-21
                    if bill_of_ladings_regions:
                        for region_name, image_base64 in bill_of_ladings_regions.items():
                            region: BaseModel = region2model[region_name]
                            if not image_base64:
                                image_base64 = page
                            print("Region Name: ", region_name)
                            prompt = region2prompt[region_name]
                            OCR_PORT = 3203     
                            all_requests.append(
                                models.OCRBatch(
                                    **{
                                        "page_type": f"BOL_{page_idx}",
                                        "region_name": region_name,
                                        "prompt": prompt,
                                        "image": image_base64,
                                        "stamp_exist": stamp_exist,
                                    }
                                )
                            )
            if cls == "receipt":
                receipt_page= await straighten_img(cls_pages[0])
                receipt_page = receipt_page["img"]
                receipt_regions = receipt_detect_regions(receipt_page).model_dump()
                ocr_output_image = {}
                if receipt_regions:
                    for region_name, image_base64 in receipt_regions.items():
                        if not image_base64:
                            print("no image>>>>>>>>>>>>>>>>>>>>>")
                            image_base64 = receipt_page
                        print(">>>>>>>>>>>>>>>>>", region_name)
                        prompt = region2prompt[region_name]
                        all_requests.append(
                            models.OCRBatch(
                                **{
                                    "page_type": "receipt",
                                    "region_name": region_name,
                                    "prompt": prompt,
                                    "image": image_base64,
                                }
                            )
                        )
        print(f"start of batch_list: {datetime.now()}")

        
        # batches =await batch_list(all_requests, batch_size)
        # print(f"End of batch_list: {datetime.now()}")

        # batch_responses = []
        # for batch in batches:
        #     print(f"start of batch: {datetime.now()}")

        #     # This line now calls the modified batch_ocr for true batch inference
        #     batch_ocr_result=await batch_ocr(batch) 
        #     batch_responses.extend(batch_ocr_result)
        #     print(f"End of batch: {datetime.now()}")



        
        # batches =await batch_list(all_requests, batch_size)
        print(f"End of batch_list: {datetime.now()}")

        # batch_responses = []
        # for batch in batches:
        print(f"start of batch: {datetime.now()}")

            # This line now calls the modified batch_ocr for true batch inference
        batch_responses=await batch_ocr(all_requests) 
            # batch_responses.extend(batch_ocr_result)
        print(f"End of batch: {datetime.now()}")


        
            
        
        grouped_batch_responses: Dict[str, List[models.OCRBatchResponse]] = group_by_page_type(batch_responses)
        print(">>>>>>>>>>>>", grouped_batch_responses)
        print(f"start of post_process: {datetime.now()}")

        for page_type, responses in grouped_batch_responses.items():
            ocr_output_image = {}
            #__________________________________________________________________________________________________________________________________________________________FIX-20
            if "BOL" in page_type:
                for response in responses:
                    region: BaseModel = region2model[response.region_name]
                    response_parsed = region(**response.ocr_response)
                    if response.region_name == "stamp":
                        response_parsed.stamp_exist = response.stamp_exist
                        response_parsed.seal_intact = response.ocr_response.get("seal_intact", "null")
                    if response.region_name == "stamp" and response_parsed.total_received not in ["empty", "null", "", "N/A"]:
                        if isinstance(response_parsed.total_received, str):
                            total_rec = re.findall(r"-?\d+", response_parsed.total_received)
                            if isinstance(total_rec, list) and len(total_rec) > 1:
                                response_parsed.total_received = int(total_rec[0])
                    # FIX: Add signature fallback logic - if stamp pod_sign is null/empty, try signatures region
                    if (response.region_name == "stamp" and 
                        response.ocr_response.get("pod_sign") in ["null", "empty", "", None] and
                        "signatures" in ocr_output_image and ocr_output_image["signatures"].get("receiver_signature")):
                        response_parsed.pod_sign = ocr_output_image["signatures"]["receiver_signature"]
                        print(f"Using signatures region fallback for stamp: {response_parsed.pod_sign}")
                    ocr_output_image.update({response.region_name: response_parsed})
                try:
                    ocr_output_pdf["BOL"].append(
                        models.OCRResponse(**ocr_output_image).model_dump()
                    )
                except Exception as e:
                    print(f"Validation failed for OCRResponse: {e}")
                    raise HTTPException(status_code=500, detail=f"Invalid response structure: {e}")
            elif page_type == "receipt":
                for response in responses:
                    if (
                        response.region_name == "total_received"
                        and response.ocr_response["total_received"] not in ["empty", "null", "N/A", ""]
                    ):
                        response.ocr_response["total_received"] = sum(response.ocr_response["total_received"])
                    elif response.region_name == "damage" and response.ocr_response["damage"] not in ["empty", "null", "N/A", ""]:
                        response.ocr_response["damage"] = sum(response.ocr_response["damage"])
                    elif response.region_name == "refused" and response.ocr_response["refused"] not in ["empty", "null", "N/A", ""]:
                        response.ocr_response["refused"] = sum(response.ocr_response["refused"])
                    ocr_output_image.update(response.ocr_response)
                try:
                    ocr_output_pdf["receipt"].append(
                        models.DeliveryReceipt(**ocr_output_image).model_dump()
                    )
                except Exception as e:
                    print(f"Validation failed for OCRResponse: {e}")
                    raise HTTPException(status_code=500, detail=f"Invalid response structure: {e}")
        
        # Merge bol pages data
        ocr_output_pdf["BOL"] = merge_bol_data(ocr_output_pdf["BOL"])

        print(">>>>>>>>>>>>", ocr_output_pdf)
        POD_FIELDS = {
            "B/L Number": "null",
            "Stamp Exists": "null",
            "Seal Intact": "null",
            "POD Date": "null",
            "Signature Exists": "null",
            "Issued Qty": "null",
            "Received Qty": "null",
            "Damage Qty": "null",
            "Short Qty": "null",
            "Over Qty": "null",
            "Refused Qty": "null",
            "Customer Order Num": "null",
        }

        # Process the aggregated OCR results to produce final POD_FIELDS output.
        # If both receipt and BOL data exist, extract the date from bill_of_lading,
        # parse it as MM-DD-YYYY, and adjust using adjust_date from utils.
        if len(ocr_output_pdf["receipt"]) == 1:
            delivery_receipt_data = ocr_output_pdf["receipt"][0]
            bol_data = ocr_output_pdf["BOL"][0]
            
            # If there's a delivery receipt, use its complete date
            POD_FIELDS["POD Date"] = preprocess_date(delivery_receipt_data["pod_date"])
            
            # Add new condition for POD Date when stamp is invalid, no receipt, and issue date is null
            if (bol_data["stamp"]["stamp_exist"] != "yes" and 
                len(ocr_output_pdf.get("receipt", [])) == 0 and 
                bol_data["bill_of_lading"].get("date", "null") in ["null", "empty", "", None]):
                
                yesterday = datetime.now() - timedelta(days=1)
                POD_FIELDS["POD Date"] = f"{yesterday.month:02d}/{yesterday.day:02d}/{yesterday.year % 100:02d}"
            
            POD_FIELDS["Signature Exists"] = delivery_receipt_data["pod_sign"]
            POD_FIELDS["Customer Order Num"] = process_customer_numbers(delivery_receipt_data["customer_order_num"])
            POD_FIELDS["Issued Qty"] = bol_data["customer_order_info"]["total_order_quantity"]
            POD_FIELDS["B/L Number"] = bol_data["bill_of_lading"]["bill_no"]
            POD_FIELDS["Stamp Exists"] = bol_data["stamp"]["stamp_exist"]
        
            if delivery_receipt_data["total_received"] not in ["null", "empty", "0"] and POD_FIELDS["Issued Qty"] not in ["null", "empty", "N/A", "", "-", "0"]:
                POD_FIELDS["Received Qty"] = int(delivery_receipt_data["total_received"])
                if int(POD_FIELDS["Issued Qty"]) == POD_FIELDS["Received Qty"]:
                    (POD_FIELDS["Refused Qty"], POD_FIELDS["Short Qty"], POD_FIELDS["Damage Qty"], POD_FIELDS["Over Qty"]) = 0, 0, 0, 0
                else:
                    (POD_FIELDS["Refused Qty"], POD_FIELDS["Short Qty"], POD_FIELDS["Damage Qty"], POD_FIELDS["Over Qty"]) = (
                        bol_data["stamp"]["refused"],
                        bol_data["stamp"]["short"],
                        bol_data["stamp"]["damage"],
                        bol_data["stamp"]["over"],
                    )
            else:
                # Add fallback logic for 'Received Qty' when it's invalid or missing in the receipt
                received_qty = str(delivery_receipt_data["total_received"]).lower()
                issued_qty = str(POD_FIELDS["Issued Qty"]).lower()

                if received_qty in ["null", "empty", "", "n/a", "0"] and issued_qty not in ["null", "empty", "", "n/a", "0"]:
                    if bol_data["stamp"]["stamp_exist"] == "yes":
                        # Check if total_received field exists and has a value in the stamp
                        stamp_total_received = bol_data["stamp"].get("total_received", "null")
                        
                        if stamp_total_received in ["null", "empty", "", "N/A"]:
                            # total_received is not present on stamp - keep as null
                            POD_FIELDS['Received Qty'] = "null"
                            POD_FIELDS['Damage Qty'] = "null"
                            POD_FIELDS['Short Qty'] = "null"
                            POD_FIELDS['Over Qty'] = "null"
                            POD_FIELDS['Refused Qty'] = "null"
                            POD_FIELDS["Status"] = "incomplete_data"
                        else:
                            # total_received is present on stamp but invalid in receipt - assume full delivery
                            POD_FIELDS['Received Qty'] = int(POD_FIELDS['Issued Qty'])
                            POD_FIELDS['Damage Qty'] = 0
                            POD_FIELDS['Short Qty'] = 0
                            POD_FIELDS['Over Qty'] = 0
                            POD_FIELDS['Refused Qty'] = 0
                            POD_FIELDS["Status"] = "valid"
                    else:
                        # Existing logic for invalid stamp
                        POD_FIELDS['Received Qty'] = "null"
                        POD_FIELDS['Damage Qty'] = "null"
                        POD_FIELDS['Short Qty'] = "null"
                        POD_FIELDS['Over Qty'] = "null"
                        POD_FIELDS['Refused Qty'] = "null"
                        POD_FIELDS["POD Date"] = "null"
                        POD_FIELDS["Signature Exists"] = "null"
                        POD_FIELDS["Stamp Exists"] = "null"
                        POD_FIELDS["Status"] = "invalid"
                        
                else:
                    # Fallback to BOL stamp if available
                    if (bol_data["stamp"]["total_received"] not in ["null", "empty"]
                        and bol_data["customer_order_info"]["total_order_quantity"] not in ["null", "empty", "N/A", "", "-", "0"]):
                        if int(bol_data["customer_order_info"]["total_order_quantity"]) == int(bol_data["stamp"]["total_received"]):
                            POD_FIELDS["Received Qty"] = int(bol_data["stamp"]["total_received"])
                            POD_FIELDS["Damage Qty"] = 0
                            POD_FIELDS["Short Qty"] = 0
                            POD_FIELDS["Over Qty"] = 0
                            POD_FIELDS["Refused Qty"] = 0
                            POD_FIELDS["Status"] = "valid" 
                        else:
                            POD_FIELDS["Received Qty"] = int(bol_data["stamp"]["total_received"])
                            (POD_FIELDS["Refused Qty"], POD_FIELDS["Short Qty"], POD_FIELDS["Damage Qty"], POD_FIELDS["Over Qty"]) = (
                                bol_data["stamp"]["refused"],
                                bol_data["stamp"]["short"],
                                bol_data["stamp"]["damage"],
                                bol_data["stamp"]["over"],
                            )
                            POD_FIELDS["Status"] = "valid" 
                    else:
                        POD_FIELDS["Received Qty"] = "null"
                        POD_FIELDS["Status"] = "invalid"
                        
            POD_FIELDS_LIST = [POD_FIELDS]
        elif len(ocr_output_pdf["BOL"]) > 1 and len(ocr_output_pdf["receipt"]) == 1:
                    POD_FIELDS["Status"] = "failed"
                    POD_FIELDS_LIST = [POD_FIELDS]
        else:
            POD_FIELDS_LIST = []
            for bol_data in ocr_output_pdf["BOL"]:
                POD_FIELDS = {
                    "B/L Number": "null",
                    "Stamp Exists": "null",
                    "Seal Intact": "null",
                    "POD Date": "null",
                    "Signature Exists": "null",
                    "Issued Qty": "null",
                    "Received Qty": "null",
                    "Damage Qty": "null",
                    "Short Qty": "null",
                    "Over Qty": "null",
                    "Refused Qty": "null",
                    "Customer Order Num": "null",
                }
                
                # Parse bill_of_lading date to get month, day, year
                bill_date_str = bol_data["bill_of_lading"].get("date", None)
                
                # Add new condition: if stamp invalid, no receipt, and issue date null

                if (bol_data["stamp"]["stamp_exist"] != "yes" and 
                    len(ocr_output_pdf.get("receipt", [])) == 0 and 
                    (bill_date_str is None or bill_date_str in ["null", "empty", ""])):
                    
                    yesterday = datetime.now() - timedelta(days=1)
                    POD_FIELDS["POD Date"] = f"{yesterday.month:02d}/{yesterday.day:02d}/{yesterday.year % 100:02d}"
                    # Set other required fields
                    POD_FIELDS["B/L Number"] = bol_data["bill_of_lading"]["bill_no"]
                    POD_FIELDS["Issued Qty"] = bol_data["customer_order_info"]["total_order_quantity"]
                    POD_FIELDS["Stamp Exists"] = bol_data["stamp"]["stamp_exist"]
                    POD_FIELDS["Seal Intact"] = bol_data["stamp"]["seal_intact"]
                    POD_FIELDS["Signature Exists"] = bol_data["stamp"]["pod_sign"]
                    
                    # FIX: Add signature fallback logic - if stamp pod_sign is null/empty, try signatures region
                    if (bol_data["stamp"]["pod_sign"] in ["null", "empty", "", None] and 
                        "signatures" in bol_data and bol_data["signatures"].get("receiver_signature")):
                        POD_FIELDS["Signature Exists"] = bol_data["signatures"]["receiver_signature"]
                        print(f"Using signatures region fallback: {POD_FIELDS['Signature Exists']}")
                    
                    # Since stamp is invalid, set quantities to null
                    POD_FIELDS["Received Qty"] = "null"
                    POD_FIELDS["Damage Qty"] = "null"
                    POD_FIELDS["Short Qty"] = "null"
                    POD_FIELDS["Over Qty"] = "null"
                    POD_FIELDS["Refused Qty"] = "null"
                    POD_FIELDS["POD Date"] = "null"
                    POD_FIELDS["Signature Exists"] = "null"
                    POD_FIELDS["Stamp Exists"] = "null"
                    
                    # Set status to invalid
                    POD_FIELDS["Status"] = "invalid"
                    
                    print(f"Using current date-1 due to invalid stamp, no receipt, and null issue date: {POD_FIELDS['POD Date']}")
                    
                else:
                    # Continue with existing date processing logic
                    # First process the POD date to get its month
                    pod_date = preprocess_date(bol_data["stamp"]["pod_date"])

                
                    # First process the POD date to get its month
                    pod_date = preprocess_date(bol_data["stamp"]["pod_date"])
                    pod_month = None
                    if pod_date not in ["null", "empty", "Error Processing Date"]:
                        try:
                            pod_parts = pod_date.split('/')
                            if len(pod_parts) >= 2:
                                pod_month = int(pod_parts[0])
                        except:
                            pod_month = None
                    
                    # Handle special case where bill_date is in format "DD/YYYY"
                    if bill_date_str and "/" in bill_date_str:
                        parts = bill_date_str.split('/')
                        if len(parts) == 2 and len(parts[1]) == 4:  # Format is "DD/YYYY"
                            try:
                                day = int(parts[0])
                                year = int(parts[1])
                                # Use POD month if available, otherwise current month
                                month = pod_month if pod_month is not None else datetime.now().month
                                bill_date_str = f"{month}/{day}/{year}"
                            except ValueError:
                                bill_date_str = None
                    
                    # Convert date format from MM-DD-YYYY to MM/DD/YY
                    if bill_date_str and "-" in bill_date_str:
                        try:
                            bill_date = datetime.strptime(bill_date_str, "%m-%d-%Y")
                            bill_date_str = bill_date.strftime("%m/%d/%y")
                        except ValueError:
                            bill_date_str = None
                    
                    processed_bill_date = preprocess_date(bill_date_str) if bill_date_str else "null"
                    extracted_year = None
                    month_bill = None
                    day_bill = None
                    
                    if processed_bill_date not in ["null", "empty", "Error Processing Date"]:
                        try:
                            parts = processed_bill_date.split('/')
                            if len(parts) >= 2:
                                # If bill date is missing month (first part is day), use POD month
                                if pod_month is not None and int(parts[0]) > 12:
                                    month_bill = pod_month
                                    day_bill = int(parts[0])
                                else:
                                    month_bill = int(parts[0])
                                    day_bill = int(parts[1])
                                
                                if len(parts) >= 3:
                                    year_str = parts[2]
                                    extracted_year = 2000 + int(year_str) if len(year_str) == 2 else int(year_str)
                                else:
                                    extracted_year = datetime.now().year
                        except Exception as e:
                            print(f"Error parsing bill_of_lading date: {e}")
                
                    # Process POD date from stamp
                    # Add these debug prints at the very start
                    print("DEBUG - Initial values:")
                    print(f"processed_bill_date: {processed_bill_date}")
                    print(f"month_bill: {month_bill}")
                    print(f"day_bill: {day_bill}")
                    print(f"extracted_year: {extracted_year}")
                    
                    # Process POD date from stamp
                    pod_date = preprocess_date(bol_data["stamp"]["pod_date"])
                    print(f"Initial POD date after preprocessing: {pod_date}")
                    if bol_data["stamp"]["stamp_exist"] and bol_data["stamp"]["pod_sign"]:
        
                        if pod_date in ["null", "empty", "Error Processing Date"] and processed_bill_date in ["null", "empty", "Error Processing Date"]:
                            POD_FIELDS["POD Date"] = get_yesterday_date()
                            print(f"Using yesterday's date due to invalid POD and bill date: {POD_FIELDS['POD Date']}")
                        else:
                            if pod_date not in ["null", "empty", "Error Processing Date"]:
                                try:
                                    date_parts = pod_date.split('/')
                                    if len(date_parts) >= 2:
                                        # Check if pod_date is in DD/YY format (needs month from bill date)
                                        first_num = int(date_parts[0])
                                        if first_num > 12 and month_bill is not None: # First number must be a day
                                            pod_month = month_bill
                                            pod_day = first_num
                                        else:
                                            pod_month = int(date_parts[0])
                                            pod_day = int(date_parts[1])
                                        pod_year = extracted_year
                                        
                                        print(f"Initial parsed POD date values - Month: {pod_month}, Day: {pod_day}, Year: {pod_year}")
                                        
                                        # Only compare with bill date if it's not null
                                        if processed_bill_date not in ["null", "empty", "Error Processing Date"]:
                                            print(f"Bill date values for comparison - Month: {month_bill}, Day: {day_bill}, Year: {extracted_year}")
                            
                                            # If month_bill is None or invalid (12/31), use POD month
                                            if month_bill is None or (month_bill == 12 and day_bill == 31):
                                                month_bill = pod_month
                                                if day_bill is None:
                                                    day_bill = pod_day
                            
                                            # Modified logic for date replacement - only if bill date exists
                                            if month_bill is not None and day_bill is not None:
                                                if pod_month > month_bill:
                                                    if pod_month == month_bill + 1 and pod_day < day_bill:
                                                        print("Special case: POD month is one month greater but POD day is less than bill day")
                                                        pass
                                                    else:
                                                        print(f"Adjusting POD month from {pod_month} to bill month {month_bill}")
                                                        pod_month = month_bill
                                                        # When adjusting to bill month, POD day must be >= bill day
                                                        if pod_day < day_bill:
                                                            print(f"Adjusting POD day from {pod_day} to bill day {day_bill} as POD day cannot be less than bill day")
                                                            pod_day = day_bill
                                                elif pod_month < month_bill:
                                                    print(f"POD month {pod_month} is less than bill month {month_bill}, adjusting")
                                                    pod_month = month_bill
                                                    # When adjusting to bill month, POD day must be >= bill day
                                                    if pod_day < day_bill:
                                                        print(f"Adjusting POD day from {pod_day} to bill day {day_bill} as POD day cannot be less than bill day")
                                                        pod_day = day_bill
                                                elif pod_month == month_bill and pod_day < day_bill:
                                                    print(f"Same month, adjusting POD day from {pod_day} to bill day {day_day}")
                                                    pod_day = day_bill
                                        else:
                                            # If bill date is null, use POD date as is
                                            print("Bill date is null, using POD date without modifications")
                                            
                                        print(f"Pre-adjustment date values - Month: {pod_month}, Day: {pod_day}, Year: {pod_year}")
                                        # Ensure valid date after adjustments
                                        adjusted_month, adjusted_day, adjusted_year = adjust_date(pod_month, pod_day, pod_year or datetime.now().year)
                                        print(f"Post-adjustment date values - Month: {adjusted_month}, Day: {adjusted_day}, Year: {adjusted_year}")
                        
                                        current_year = datetime.now().year
                                        current_month = datetime.now().month
                                        current_day = datetime.now().day
                                        
                                        if adjusted_year > current_year or str(adjusted_year)[-2:] == "23" or str(adjusted_year)[-2:] < "20":
                                            print(f"Adjusting year from {adjusted_year} to current year {current_year}")
                                            adjusted_year = current_year
        
                                        # If year is current year, ensure month is not greater than current month
                                        if adjusted_year == current_year:
                                            if adjusted_month > current_month:
                                                print(f"Month {adjusted_month} is greater than current month {current_month}, adjusting month to current month")
                                                adjusted_month = current_month
                                                # When adjusting to current month, set day to current day
                                                if adjusted_day > current_day:
                                                    adjusted_day = current_day - 1
                                                    print(f"Adjusted day to {adjusted_day} to match current month")
                                        
                                        # Format POD Date
                                        POD_FIELDS["POD Date"] = f"{adjusted_month:02d}/{adjusted_day:02d}/{adjusted_year % 100:02d}"
                                        print(f"Final POD Date set to: {POD_FIELDS['POD Date']}")
                                    else:
                                        if month_bill is not None and day_bill is not None and extracted_year is not None:
                                            POD_FIELDS["POD Date"] = f"{month_bill:02d}/{day_bill:02d}/{extracted_year % 100:02d}"
                                            print(f"Using bill date due to invalid POD date parts: {POD_FIELDS['POD Date']}")
                                        else:
                                            POD_FIELDS["POD Date"] = get_yesterday_date()
                                            print(f"Using yesterday's date due to invalid date parts: {POD_FIELDS['POD Date']}")
                                except (ValueError, IndexError) as e:
                                    print(f"Error processing POD date: {e}")
                                    if month_bill is not None and day_bill is not None and extracted_year is not None:
                                        POD_FIELDS["POD Date"] = f"{month_bill:02d}/{day_bill:02d}/{extracted_year % 100:02d}"
                                        print(f"Using bill date due to processing error: {POD_FIELDS['POD Date']}")
                                    else:
                                        POD_FIELDS["POD Date"] = get_yesterday_date()
                                        print(f"Using yesterday's date due to processing error: {POD_FIELDS['POD Date']}")
                            else:
                                if month_bill is not None and day_bill is not None and extracted_year is not None:
                                    POD_FIELDS["POD Date"] = f"{month_bill:02d}/{day_bill:02d}/{extracted_year % 100:02d}"
                                    print(f"Using bill date due to invalid POD date: {POD_FIELDS['POD Date']}")
                                else:
                                    POD_FIELDS["POD Date"] = get_yesterday_date()
                                    print(f"Using yesterday's date due to invalid dates: {POD_FIELDS['POD Date']}")
            
                    POD_FIELDS["B/L Number"] = bol_data["bill_of_lading"]["bill_no"]
                    POD_FIELDS["Issued Qty"] = bol_data["customer_order_info"]["total_order_quantity"]
                    POD_FIELDS["Stamp Exists"] = bol_data["stamp"]["stamp_exist"]
                    POD_FIELDS["Seal Intact"] = bol_data["stamp"]["seal_intact"]
                    POD_FIELDS["Signature Exists"] = bol_data["stamp"]["pod_sign"]
    
                    # FIX: Add signature fallback logic - if stamp pod_sign is null/empty, try signatures region
                    if (bol_data["stamp"]["pod_sign"] in ["null", "empty", "", None] and 
                        "signatures" in bol_data and bol_data["signatures"].get("receiver_signature")):
                        POD_FIELDS["Signature Exists"] = bol_data["signatures"]["receiver_signature"]
                        print(f"Using signatures region fallback: {POD_FIELDS['Signature Exists']}")
    
                    if bol_data["stamp"]["total_received"] not in ["null", "empty", "0"]:
                        if int(bol_data["customer_order_info"]["total_order_quantity"]) == int(bol_data["stamp"]["total_received"]):
                            (POD_FIELDS["Refused Qty"], POD_FIELDS["Short Qty"], POD_FIELDS["Damage Qty"], POD_FIELDS["Over Qty"]) = 0, 0, 0, 0
                            POD_FIELDS["Received Qty"] = int(bol_data["stamp"]["total_received"])
                        else:
                            (POD_FIELDS["Refused Qty"], POD_FIELDS["Short Qty"], POD_FIELDS["Damage Qty"], POD_FIELDS["Over Qty"]) = (
                                bol_data["stamp"]["refused"],
                                bol_data["stamp"]["short"],
                                bol_data["stamp"]["damage"],
                                bol_data["stamp"]["over"],
                            )
                            POD_FIELDS["Received Qty"] = int(bol_data["stamp"]["total_received"])
                    else:
                        (POD_FIELDS["Refused Qty"], POD_FIELDS["Short Qty"], POD_FIELDS["Damage Qty"], POD_FIELDS["Over Qty"]) = (
                            bol_data["stamp"]["refused"],
                            bol_data["stamp"]["short"],
                            bol_data["stamp"]["damage"],
                            bol_data["stamp"]["over"],
                        )
                        POD_FIELDS["Received Qty"] = bol_data["stamp"]["total_received"]
                    #______________________________________________________________________________________________________________________________________________FIX-18
    
    
                    # Updated check with type handling
                    received_qty = str(POD_FIELDS['Received Qty']).lower()
                    issued_qty = str(POD_FIELDS['Issued Qty']).lower()

                    print(f"End of post_process: {datetime.now()}")

                    
                    # Check and handle invalid 'Received Qty'
                        #__________________________________________________________________________________________________________________________________________________________FIX-18
                    if received_qty in ["null", "empty", "", "n/a", "0"] and issued_qty not in ["null", "empty", "", "n/a", "0"]:
                        if bol_data["stamp"]["stamp_exist"] == "yes":
                            # Check if total_received field exists and has a value in the stamp
                            stamp_total_received = bol_data["stamp"].get("total_received", "null")
                            
                            if stamp_total_received in ["null", "empty", "", "N/A"]:
                                # total_received is not present on stamp - keep as null
                                POD_FIELDS['Received Qty'] = "null"
                                POD_FIELDS['Damage Qty'] = "null"
                                POD_FIELDS['Short Qty'] = "null"
                                POD_FIELDS['Over Qty'] = "null"
                                POD_FIELDS['Refused Qty'] = "null"
                                POD_FIELDS["Status"] = "incomplete_data"
                            else:
                                # total_received is present on stamp but invalid in receipt - assume full delivery
                                POD_FIELDS['Received Qty'] = int(POD_FIELDS['Issued Qty'])
                                POD_FIELDS['Damage Qty'] = 0
                                POD_FIELDS['Short Qty'] = 0
                                POD_FIELDS['Over Qty'] = 0
                                POD_FIELDS['Refused Qty'] = 0
                                POD_FIELDS["Status"] = "valid"
                        else:
                            # Existing logic for invalid stamp
                            POD_FIELDS['Received Qty'] = "null"
                            POD_FIELDS['Damage Qty'] = "null"
                            POD_FIELDS['Short Qty'] = "null"
                            POD_FIELDS['Over Qty'] = "null"
                            POD_FIELDS['Refused Qty'] = "null"
                            POD_FIELDS["POD Date"] = "null"
                            POD_FIELDS["Signature Exists"] = "null"
                            POD_FIELDS["Stamp Exists"] = "null"
                            POD_FIELDS["Status"] = "invalid"
                    else:
                        POD_FIELDS["Status"] = "valid"
                        
                    POD_FIELDS_LIST.append(POD_FIELDS)
                    
        return POD_FIELDS_LIST

    except Exception as e:
        from traceback import print_exc
        print(print_exc())
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")
