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
import logging

# Basic structured logging for API
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

# Detection Configuration Class
class DetectionConfig:
    """Configuration for detection thresholds and quality parameters"""
    RECEIPT_CONF = 0.45  # Minimum confidence for receipt regions
    BOL_CONF = 0.60      # Minimum confidence for BOL regions
    PAGE_CONF = 0.60     # Minimum confidence for page classification
    STAMP_CONF = 0.75    # Minimum confidence for stamp detection
    IOU_THRESHOLD = 0.40 # IOU for NMS
    MIN_REGION_SIZE = (80, 80)  # Minimum region dimensions
    BLUR_THRESHOLD = 100  # Laplacian variance threshold

def validate_image_quality(image: Image.Image, min_size=(100, 100)) -> bool:
    """Validate if image meets minimum quality requirements"""
    if image.width < min_size[0] or image.height < min_size[1]:
        logger.warning(f"Image too small: {image.size}")
        return False
    
    # Check if image is too blurry
    img_array = np.array(image.convert('L'))
    laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
    if laplacian_var < DetectionConfig.BLUR_THRESHOLD:
        logger.warning(f"Image too blurry: {laplacian_var}")
        return False
    
    return True

def log_detection_stats(results, detection_type):
    """Log statistics about detections"""
    if results and len(results) > 0:
        confidences = results[0].boxes.conf.tolist() if results[0].boxes else []
        if confidences:
            logger.info(f"{detection_type} - Detections: {len(confidences)}, "
                       f"Avg Conf: {np.mean(confidences):.2f}, "
                       f"Min: {min(confidences):.2f}, Max: {max(confidences):.2f}")

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
    resize_large_image,
    # New helper functions for OCR type conversion
    safe_int_conversion,
    safe_str_conversion,
    safe_numeric_operation,
    normalize_ocr_data
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
    
    # Detect regions using YOLO model with improved thresholds
    results: List[Results] = receipt_region_detector.predict(
        base64topil(image), 
        conf=DetectionConfig.RECEIPT_CONF,
        iou=DetectionConfig.IOU_THRESHOLD
    )
    
    # Log detection statistics
    log_detection_stats(results, "Receipt Region Detection")
    
    # Process detected regions with quality filtering
    boxes = {}
    region_scores = {}  # Track confidence scores
    
    for result in results:
        for class_id, box, conf in zip(
            result.boxes.cls.tolist(), 
            result.cpu().boxes.xyxy,
            result.boxes.conf.tolist()  # Get confidence scores
        ):
            if class_id in list(class_names.keys()):
                region_name = class_names[class_id]
                
                # Only keep highest confidence detection for each region
                if region_name not in region_scores or conf > region_scores[region_name]:
                    cropped = Image.fromarray(
                        save_one_box(box, im=result.orig_img, save=False)
                    )
                    
                    # Validate cropped region quality
                    if validate_image_quality(cropped, min_size=DetectionConfig.MIN_REGION_SIZE):
                        boxes[region_name] = convert_image_to_base64(
                            make_image_square(cropped)
                        )
                        region_scores[region_name] = conf
                        logger.info(f"Detected {region_name} with confidence: {conf:.2f}")
                    else:
                        logger.warning(f"Rejected {region_name} due to poor quality")
    
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

    # Detect regions using YOLO model with improved thresholds
    results: List[Results] = bol_region_detector.predict(
        base64topil(image), 
        conf=DetectionConfig.BOL_CONF,
        iou=DetectionConfig.IOU_THRESHOLD
    )
    
    # Log detection statistics
    log_detection_stats(results, "BOL Region Detection")
    
    # Enhanced logging for BOL region detection
    detected_regions = []
    for result in results:
        if result.boxes:
            for class_id, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                if class_id in class_names:
                    detected_regions.append(f"{class_names[class_id]} (conf: {conf:.2f})")
    logger.info(f"\n\n🎯 BOL REGIONS DETECTED\nRegions: {detected_regions}\nClass mapping: {class_names}\n\n")
    
    boxes = {}
    region_scores = {}  # Track confidence scores
    
    for result in results:
        orig_img = result.orig_img
        h, w, _ = orig_img.shape

        for class_id, box, conf in zip(
            result.boxes.cls.tolist(), 
            result.cpu().boxes.xyxy,
            result.boxes.conf.tolist()  # Get confidence scores
        ):
            region_name = class_names[class_id]
            
            # Only keep highest confidence detection for each region
            if region_name not in region_scores or conf > region_scores[region_name]:
                x1, y1, x2, y2 = map(int, box)
                
                # Special handling for bill_of_lading region
                if region_name == "bill_of_lading":
                    x1, y1 = 0, 0  # Extend to top-left corner
                    x2 = min(x2, w)  # Limit to image width
                    y2 = min(y2, h)  # Limit to image height
                
                # Convert coordinates to tensor for save_one_box
                box_tensor = torch.tensor([x1, y1, x2, y2])
                cropped_img = save_one_box(box_tensor, im=orig_img, save=False)
                cropped_pil = Image.fromarray(cropped_img)
                
                # Validate cropped region quality
                if validate_image_quality(cropped_pil, min_size=DetectionConfig.MIN_REGION_SIZE):
                    boxes[region_name] = convert_image_to_base64(cropped_pil)
                    region_scores[region_name] = conf
                    logger.info(f"Detected {region_name} with confidence: {conf:.2f}")
                else:
                    logger.warning(f"Rejected {region_name} due to poor quality")
    
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

def validate_and_clean_stamp_data(ocr_response: dict, region_name: str) -> dict:
    """
    Validate and clean stamp OCR responses to catch common errors
    Args:
        ocr_response: Raw OCR response from VLM
        region_name: Name of the region being processed
    Returns:
        Cleaned OCR response
    """
    if region_name != "stamp":
        return ocr_response
    
    cleaned = ocr_response.copy()
    
    # ===== VALIDATE total_received =====
    total_received = cleaned.get("total_received")

    # ===== PARSE received_qty patterns (208 cases, 9=4, 70 ctns) =====
    if total_received and isinstance(total_received, str):
        # Extract first number before text/symbols
        match = re.match(r'^(\d+)', total_received.strip())
        if match:
            total_received = int(match.group(1))
            cleaned["total_received"] = total_received
    
    # List of common false positive values
    # These are typically DD/DC/BDC reference numbers, not carton counts
    suspicious_values = [226, 3511, 2028, 2098, 1504, 1430, 1424, 412113, 287868]
    
    if isinstance(total_received, (int, str)):
        try:
            total_rec_int = int(total_received) if isinstance(total_received, str) else total_received
            
            # Check if it's a known false positive
            if total_rec_int in suspicious_values:
                logger.warning(f"⚠️ Suspicious total_received: {total_received} (likely DD/DC/BDC/time number)")
                cleaned["total_received"] = "null"
            
            # Check if it's a 4-digit number > 1000 (likely time or reference)
            elif len(str(total_rec_int)) == 4 and total_rec_int > 1000:
                logger.warning(f"⚠️ 4-digit number {total_received} likely NOT cartons (time/reference)")
                cleaned["total_received"] = "null"
            
            # Check if it's > 5 digits (too long for carton count)
            elif total_rec_int > 99999:
                logger.warning(f"⚠️ Number {total_received} too large for carton count")
                cleaned["total_received"] = "null"
                
        except (ValueError, TypeError):
            pass
    
    # ===== VALIDATE pod_date =====
    pod_date = cleaned.get("pod_date")
    
    # Check if date matches example dates from prompt (false positive)
    example_dates = ["10/23/24", "10-23-2024", "OCT 23 2024", "OCT 23, 2024", "OCT - 23, 2024", "OCT / 23, 2024", "10/22/24", "9/16/25"]
    if pod_date in example_dates:
        logger.warning(f"⚠️ pod_date matches prompt example: {pod_date} - setting to empty")
        cleaned["pod_date"] = "empty"
    
    # Validate date format - should not have impossible values
    if pod_date and pod_date not in ["null", "empty", None]:
        try:
            # Handle different date formats
            date_str = str(pod_date).replace("-", "/")
            parts = date_str.split("/")
            
            if len(parts) >= 2:
                first_num = int(parts[0])
                second_num = int(parts[1])
                
                # Check for impossible values
                if first_num > 31 or second_num > 31:
                    logger.warning(f"⚠️ Invalid date values: {pod_date}")
                    cleaned["pod_date"] = "null"
                # Check for month > 12 (if format is MM/DD)
                elif first_num > 12 and second_num <= 12:
                    # Likely DD/MM format, which is fine
                    pass
                elif first_num > 12 and second_num > 12:
                    # Both > 12, invalid
                    logger.warning(f"⚠️ Invalid date: {pod_date}")
                    cleaned["pod_date"] = "null"
        except (ValueError, IndexError, AttributeError):
            pass
    
    # ===== VALIDATE damage/short/over/refused =====
    for field in ["damage", "short", "over", "refused", "roc_damaged", "damaged_kept"]:
        value = cleaned.get(field)
        
        # If value is a single letter without number, it's just a label
        if value in ["D", "S", "O", "R", "d", "s", "o", "r"]:
            logger.warning(f"⚠️ {field} is just a letter, setting to empty")
            cleaned[field] = "empty"
        
        # Validate reasonable range
        if isinstance(value, (int, str)):
            try:
                val_int = int(value) if isinstance(value, str) and value not in ["empty", "null"] else value
                if isinstance(val_int, int):
                    if val_int < 0:
                        logger.warning(f"⚠️ {field} value {val_int} is negative")
                        cleaned[field] = "null"
                    elif val_int > 10000:
                        logger.warning(f"⚠️ {field} value {val_int} unreasonably high")
                        cleaned[field] = "null"
            except (ValueError, TypeError):
                pass
    
    # ===== VALIDATE seal_intact =====
    seal_intact = cleaned.get("seal_intact")
    
    # If seal_intact has a value but stamp doesn't exist, likely false positive
    if seal_intact in ["yes", "no"] and cleaned.get("stamp_exist") == "no":
        logger.warning("⚠️ seal_intact has value but no stamp exists, setting to null")
        cleaned["seal_intact"] = "null"
    
    # ===== VALIDATE stamp_exist consistency =====
    stamp_exist = cleaned.get("stamp_exist")
    
    # If stamp_exist is "no" but we have data, there's likely a notation area
    if stamp_exist == "no":
        has_data = any([
            cleaned.get("pod_date") not in ["null", "empty", None],
            cleaned.get("pod_sign") not in ["null", "empty", None],
            cleaned.get("total_received") not in ["null", "empty", None]
        ])
        if has_data:
            logger.info("✓ Notation area detected (stamp_exist='no' but data present)")
            # This is fine - notation area without physical stamp
    
    return cleaned


def log_extraction_summary(ocr_response: dict, region_name: str):
    """
    Log summary of what was extracted for debugging
    Args:
        ocr_response: OCR response data
        region_name: Name of the region
    """
    if region_name == "stamp":
        logger.info(f"""
╔═══════════════════════════════════════════════════════╗
║ 📋 STAMP EXTRACTION SUMMARY                           ║
╠═══════════════════════════════════════════════════════╣
║ total_received: {str(ocr_response.get('total_received', 'N/A')):<35} ║
║ pod_date:       {str(ocr_response.get('pod_date', 'N/A')):<35} ║
║ pod_sign:       {str(ocr_response.get('pod_sign', 'N/A')):<35} ║
║ damage:         {str(ocr_response.get('damage', 'N/A')):<35} ║
║ short:          {str(ocr_response.get('short', 'N/A')):<35} ║
║ over:           {str(ocr_response.get('over', 'N/A')):<35} ║
║ refused:        {str(ocr_response.get('refused', 'N/A')):<35} ║
║ stamp_exist:    {str(ocr_response.get('stamp_exist', 'N/A')):<35} ║
║ seal_intact:    {str(ocr_response.get('seal_intact', 'N/A')):<35} ║
║ notation_exist:    {str(ocr_response.get('notation_exist', 'N/A')):<35} ║
╚═══════════════════════════════════════════════════════╝
        """)

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
            logger.info(f"\n\n🤖 VLM RAW RESPONSE\n...")
            
            # Extract JSON
            pattern = r"```json(.*?)```"
            match = re.search(pattern, json_to_parse, re.DOTALL)
            if match:
                json_to_parse = match.group(1)
            json_content = json_to_parse.strip("```json").strip("```").replace("'null'", "null")
            logger.info(f"\n\n📝 EXTRACTED JSON\n...")
            
            # ✅ NEW CODE - ADD VALIDATION:
            parsed_data = json.loads(json_content)
            
            # Apply validation and cleaning
            validated_data = validate_and_clean_stamp_data(
                parsed_data, 
                metadata["region_name"]
            )
            
            # Log extraction summary for debugging
            log_extraction_summary(validated_data, metadata["region_name"])
            
            final_ocr_responses.append(
                models.OCRBatchResponse(
                    page_type=metadata["page_type"],
                    region_name=metadata["region_name"],
                    ocr_response=validated_data,  # ✅ Use validated data
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
    results: List[Results] = page_classifier.predict(processed_images, conf=DetectionConfig.PAGE_CONF)
    
    # Log detection statistics
    log_detection_stats(results, "Page Classification")
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

def merge_bol_data(data):
    """
    Merge multiple BOL entries for the same bill number
    Uses intelligent scoring to prioritize pages with better data quality
    Args:
        data: List of BOL data entries
    Returns:
        list: Merged BOL data with best quality fields
    """
    merged_data = []
    
    def calculate_data_quality_score(bol):
        """
        Calculate quality score for a BOL entry based on data completeness
        Higher score = better quality data
        """
        score = 0
        stamp = bol.get("stamp", {})
        
        # Critical fields (worth more points)
        if stamp.get("pod_date") not in ["null", "empty", "", None]:
            # Valid date format check
            pod_date = str(stamp["pod_date"])
            if "/" in pod_date and len(pod_date) >= 8:  # e.g., "06/24/2025"
                score += 50  # Full date is very valuable
            elif pod_date.isdigit() and len(pod_date) <= 2:  # e.g., "18"
                score += 5   # Partial date is low value
            else:
                score += 20  # Some date format
        
        if stamp.get("total_received") not in ["null", "empty", "", 0, None]:
            score += 40  # Total received is critical
        
        # Important fields
        if stamp.get("pod_sign") not in ["null", "empty", "", None]:
            score += 15
        
        if stamp.get("stamp_exist") == "yes":
            score += 10
        
        # Discrepancy fields (if any are filled, add points)
        for field in ["damage", "short", "over", "refused"]:
            if stamp.get(field) not in ["null", "empty", "", None]:
                score += 5
        
        # Other fields
        if stamp.get("seal_intact") not in ["null", "empty", "", None]:
            score += 5
        
        if stamp.get("notation_exist") not in ["null", "empty", "", None]:
            score += 5
        
        return score
    
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
        if len(bol_list) == 1:
            # Only one BOL page, use it directly
            merged_data.append(bol_list[0])
            logger.info(f"Single BOL page for bill {bill_no}, using directly")
            continue
        
        # Multiple pages - score them and use the best as base
        scored_bols = [
            (bol, calculate_data_quality_score(bol)) 
            for bol in bol_list
        ]
        
        # Sort by score (highest first)
        scored_bols.sort(key=lambda x: x[1], reverse=True)
        
        # Log the scores for debugging
        logger.info(f"\n🔍 BOL MERGE QUALITY SCORES for {bill_no}:")
        for idx, (bol, score) in enumerate(scored_bols):
            logger.info(f"  Page {idx}: Score={score}, pod_date={bol['stamp'].get('pod_date')}, "
                       f"total_received={bol['stamp'].get('total_received')}")
        
        # Start with the highest quality BOL as base
        best_bol = scored_bols[0][0]
        merged = copy.deepcopy(best_bol)
        logger.info(f"✅ Using page with score {scored_bols[0][1]} as base for {bill_no}")
        
        # Now merge in any better values from other pages
        for key in merged["stamp"].keys():
            values = [bol["stamp"][key] for bol in bol_list]
            
            # If all values are the same, keep it
            if all(value == values[0] for value in values):
                continue
            
            # Get non-null/empty values with their quality indicators
            non_null_values = []
            for bol in bol_list:
                value = bol["stamp"][key]
                if value not in ["null", "empty", "", None]:
                    non_null_values.append(value)
            
            if non_null_values:
                # For pod_date, prefer longer/more complete dates
                if key == "pod_date":
                    # Sort by length (longer = more complete)
                    sorted_dates = sorted(non_null_values, key=lambda x: len(str(x)), reverse=True)
                    merged["stamp"][key] = sorted_dates[0]
                    if len(non_null_values) > 1:
                        logger.info(f"  Merged {key}: chose '{sorted_dates[0]}' over {non_null_values[1:]}")
                
                # For numeric fields, prefer non-zero values
                elif key in ["total_received", "damage", "short", "over", "refused"]:
                    numeric_values = [v for v in non_null_values if v != 0]
                    if numeric_values:
                        merged["stamp"][key] = numeric_values[0]
                    else:
                        merged["stamp"][key] = non_null_values[0]
                
                # For other fields, keep the value from the highest-scored page
                # (already set from best_bol)
                else:
                    pass
            else:
                # All values are null/empty, use resolution logic
                merged["stamp"][key] = resolve_conflict(values)
        
        merged_data.append(merged)
        logger.info(f"✅ Final merged data for {bill_no}: pod_date={merged['stamp']['pod_date']}, "
                   f"total_received={merged['stamp']['total_received']}\n")

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
    results = stamp_detector.predict(base64topil(image), conf=DetectionConfig.STAMP_CONF)
    
    # Log detection statistics
    log_detection_stats(results, "Stamp Detection")
    
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

@traceable(run_type="chain", name="run-ocr", dangerously_allow_filesystem=True)
async def run_ocr(file_url_or_path: str, batch_size=4, port=None) -> List[models.OCRResponse]:
    # Initialize ML Models
   
    
    try:
        logger.info(f"\n\n🚀 STARTING OCR PROCESSING\nFile: {file_url_or_path}\n\n")
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
        logger.info(f"\n\n📄 DOCUMENT CLASSIFICATION START\n\n")


        classified_pages = classify_documents(
            [convert_image_to_base64(page) for page in pages]
        )
        logger.info(f"\n\n📄 DOCUMENT CLASSIFICATION COMPLETE\nPages classified: {len(classified_pages)}\nClassification results: {list(classified_pages.keys())}")

        
        ocr_output_pdf = defaultdict(list)
        all_requests = []
        for cls, cls_pages in classified_pages.items():
            if cls == "BOL":
                for page_idx, page in enumerate(cls_pages):
                    straighten_result =await straighten_img(page)
                    page =straighten_result["img"]
                    stamp_exist =await bol_stamp_exist(page)
                    bill_of_ladings_regions = bol_detect_regions(page).model_dump()
                    logger.info(f"\n\n🎯 BOL REGION DETECTION\nPage: {page_idx}\nRegions detected: {list(bill_of_ladings_regions.keys())}\nStamp exists: {stamp_exist}\n\n")
                    ocr_output_image = {}
                    # Enhanced fallback logic for missing BOL regions
                    if bill_of_ladings_regions:
                        for region_name, image_base64 in bill_of_ladings_regions.items():
                            region: BaseModel = region2model[region_name]
                            if not image_base64:
                                logger.warning(f"\n\n⚠️ MISSING BOL REGION IMAGE\nRegion: {region_name}\nUsing full page as fallback\n\n")
                                image_base64 = page
                            logger.info(f"\n\n📝 PROCESSING REGION\nRegion: {region_name}\nPrompt type: {region_name}\nStamp exists: {stamp_exist}\n\n")
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
                    else:
                        # If no regions detected at all, use heuristic fallback
                        logger.warning(f"\n\n⚠️ NO BOL REGIONS DETECTED\nUsing heuristic stamp crop as fallback\n\n")
                        from utils import heuristic_stamp_crop_base64
                        fallback_stamp = heuristic_stamp_crop_base64(page)
                        if fallback_stamp:
                            all_requests.append(
                                models.OCRBatch(
                                    **{
                                        "page_type": f"BOL_{page_idx}",
                                        "region_name": "stamp",
                                        "prompt": region2prompt["stamp"],
                                        "image": fallback_stamp,
                                        "stamp_exist": stamp_exist,
                                    }
                                )
                            )
            if cls == "receipt":
                receipt_page= await straighten_img(cls_pages[0])
                receipt_page = receipt_page["img"]
                receipt_regions = receipt_detect_regions(receipt_page).model_dump()
                logger.info(f"\n\n🎯 RECEIPT REGION DETECTION\nRegions detected: {list(receipt_regions.keys())}\n\n")
                ocr_output_image = {}
                if receipt_regions:
                    # Enhanced fallback logic for missing receipt regions
                    for region_name, image_base64 in receipt_regions.items():
                        if not image_base64:
                            logger.warning(f"\n\n⚠️ MISSING RECEIPT REGION IMAGE\nRegion: {region_name}\nUsing full page as fallback\n\n")
                            image_base64 = receipt_page
                        logger.info(f"\n\n📝 PROCESSING RECEIPT REGION\nRegion: {region_name}\nPrompt type: {region_name}\n\n")
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
                else:
                    # If no regions detected at all, use critical regions as fallback
                    logger.warning(f"\n\n⚠️ NO RECEIPT REGIONS DETECTED\nUsing critical regions as fallback\n\n")
                    critical_regions = ["pod_date", "pod_sign", "total_received"]
                    for region_name in critical_regions:
                        logger.info(f"\n\n📝 PROCESSING FALLBACK RECEIPT REGION\nRegion: {region_name}\nUsing full page\n\n")
                        prompt = region2prompt[region_name]
                        all_requests.append(
                            models.OCRBatch(
                                **{
                                    "page_type": "receipt",
                                    "region_name": region_name,
                                    "prompt": prompt,
                                    "image": receipt_page,
                                }
                            )
                        )
        logger.info(f"\n\n🔍 REGION DETECTION START\nTotal requests prepared: {len(all_requests)}\n\n")

        logger.info(f"\n\n🔍 REGION DETECTION COMPLETE \n\n")

        logger.info(f"\n\n🤖 VLM OCR PROCESSING START\nBatch size: {len(all_requests)}\n\n")
        batch_responses=await batch_ocr(all_requests) 
        logger.info(f"\n\n🤖 VLM OCR PROCESSING COMPLETE\nResponses received: {len(batch_responses)}\n\n")


        
            
        
        grouped_batch_responses: Dict[str, List[models.OCRBatchResponse]] = group_by_page_type(batch_responses)
        logger.info(f"\n\n⚙️ POST-PROCESSING START\nGrouped responses: {list(grouped_batch_responses.keys())}\n\n")

        for page_type, responses in grouped_batch_responses.items():
            ocr_output_image = {}
            #__________________________________________________________________________________________________________________________________________________________FIX-20
            if "BOL" in page_type:
                for response in responses:
                    logger.info(f"\n\n📊 OCR RESPONSE RECEIVED\nRegion: {response.region_name}\nPage type: {response.page_type}\nRaw response: {response.ocr_response}\n\n")
                    # Normalize stamp data before parsing to handle type conversion issues
                    if response.region_name == "stamp":
                        # Define expected types for stamp fields
                        stamp_field_types = {
                            "damage": int,
                            "short": int,
                            "refused": int,
                            "over": int,
                            "total_received": int,
                            "stamp_exist": str,
                            "notation_exist": str,
                            "pod_sign": str,
                            "pod_date": str,
                            "seal_intact": str
                        }
                        # Normalize stamp data to prevent type errors
                        response.ocr_response = normalize_ocr_data(response.ocr_response, stamp_field_types)
                        logger.info(f"\n\n✅ STAMP DATA NORMALIZED\nNormalized response: {response.ocr_response}\n\n")
                    
                    elif response.region_name == "customer_order_info":
                        customer_order_field_types = {
                            "total_order_quantity": int,
                        }
                        response.ocr_response = normalize_ocr_data(response.ocr_response, customer_order_field_types)
                        logger.info(f"\n\n✅ CUSTOMER ORDER DATA NORMALIZED\nNormalized response: {response.ocr_response}\n\n")    

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
                    # Normalize receipt data before processing
                    if response.region_name in ["total_received", "damage", "refused"]:
                        # If it's a list, convert each element safely
                        if isinstance(response.ocr_response.get(response.region_name), list):
                            items = response.ocr_response[response.region_name]
                            response.ocr_response[response.region_name] = [
                                safe_int_conversion(item, 0) for item in items
                            ]
                    
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

        logger.info(f"\n\n⚙️ POST-PROCESSING COMPLETE\nFinal POD fields generated: \n\n")
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
            POD_FIELDS["Issued Qty"] = safe_int_conversion(
                bol_data["customer_order_info"]["total_order_quantity"], 0
            )
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
                    if bol_data["stamp"]["stamp_exist"] == "yes" or bol_data["stamp"]["notation_exist"] == "yes":
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

                if (
                    (bol_data["stamp"].get("stamp_exist") != "yes" and bol_data["stamp"].get("notation_exist") != "yes")
                    and len(ocr_output_pdf.get("receipt", [])) == 0
                    and (bill_date_str is None or bill_date_str in ["null", "empty", ""])
                ):
                    
                    yesterday = datetime.now() - timedelta(days=1)
                    POD_FIELDS["POD Date"] = f"{yesterday.month:02d}/{yesterday.day:02d}/{yesterday.year % 100:02d}"
                    # Set other required fields
                    POD_FIELDS["B/L Number"] = bol_data["bill_of_lading"]["bill_no"]
                    POD_FIELDS["Issued Qty"] = safe_int_conversion(
                        bol_data["customer_order_info"]["total_order_quantity"], 0
                    )
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
                                pod_month = safe_int_conversion(pod_parts[0], None)
                        except:
                            pod_month = None

                    processed_bill_date = preprocess_date(bill_date_str)
                    extracted_year = None
                    month_bill = None
                    day_bill = None
                    
                    if processed_bill_date not in ["null", "empty", "Error Processing Date"]:
                        try:
                            parts = processed_bill_date.split('/')
                            if len(parts) >= 2:
                                # If bill date is missing month (first part is day), use POD month
                                first_part = safe_int_conversion(parts[0], 0)
                                if pod_month is not None and first_part > 12:
                                    month_bill = pod_month
                                    day_bill = first_part
                                else:
                                    month_bill = first_part
                                    day_bill = safe_int_conversion(parts[1], 0)
                                
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
                    pod_date = preprocess_date(bol_data["stamp"].get("pod_date"))
                    processed_bill_date = preprocess_date(bol_data["bill_of_lading"].get("date"))

                    print(f"Initial POD date after preprocessing: {pod_date}")
                    print(f"Processed bill date: {processed_bill_date}")

                    # Explicit checks for stamp existence and signature presence
                    if (
                        (
                            bol_data["stamp"].get("stamp_exist") in ["yes", True, "True"]
                            or bol_data["stamp"].get("notation_exist") in ["yes", True, "True"]
                        )
                        and bol_data["stamp"].get("pod_sign") in ["yes", "no", True, False, "True", "False"]
                    ):

                        # Prefer POD date if it's valid, otherwise bill date, otherwise yesterday
                        if pod_date not in ["null", "empty", "Error Processing Date"]:
                            POD_FIELDS["POD Date"] = pod_date
                            print(f"✅ Using POD date directly: {POD_FIELDS['POD Date']}")
                        elif processed_bill_date not in ["null", "empty", "Error Processing Date"]:
                            POD_FIELDS["POD Date"] = processed_bill_date
                            print(f"Using bill date: {POD_FIELDS['POD Date']}")
                        else:
                            POD_FIELDS["POD Date"] = pod_date
                            print(f"Using yesterday date: {POD_FIELDS['POD Date']}")
                    else:
                        # Invalid stamp - mark fields as null/invalid
                        POD_FIELDS["POD Date"] = "null"
                        POD_FIELDS["Signature Exists"] = "null"
                        POD_FIELDS["Stamp Exists"] = "null"
                        POD_FIELDS["Status"] = "invalid"
                        print("Invalid stamp - setting POD date to null")

                    # Continue with other fields (unchanged)
                    POD_FIELDS["B/L Number"] = bol_data["bill_of_lading"]["bill_no"]
                    POD_FIELDS["Issued Qty"] = safe_int_conversion(
                        bol_data["customer_order_info"]["total_order_quantity"], 0
                    )
                    POD_FIELDS["Stamp Exists"] = bol_data["stamp"]["stamp_exist"]
                    POD_FIELDS["Seal Intact"] = bol_data["stamp"]["seal_intact"]
                    POD_FIELDS["Signature Exists"] = bol_data["stamp"]["pod_sign"]

    
                    # FIX: Add signature fallback logic - if stamp pod_sign is null/empty, try signatures region
                    if (bol_data["stamp"]["pod_sign"] in ["null", "empty", "", None] and 
                        "signatures" in bol_data and bol_data["signatures"].get("receiver_signature")):
                        POD_FIELDS["Signature Exists"] = bol_data["signatures"]["receiver_signature"]
                        print(f"Using signatures region fallback: {POD_FIELDS['Signature Exists']}")
    
                    # Safely convert all stamp quantities to integers
                    stamp_total_received = safe_int_conversion(bol_data["stamp"]["total_received"], 0)
                    stamp_total_received_str = safe_str_conversion(bol_data["stamp"]["total_received"], "null")
                    customer_order_qty = safe_int_conversion(bol_data["customer_order_info"]["total_order_quantity"], 0)

                    if stamp_total_received_str not in ["null", "empty", "0"]:
                        if customer_order_qty == stamp_total_received:
                            # All quantities match - no discrepancies
                            POD_FIELDS["Refused Qty"] = 0
                            POD_FIELDS["Short Qty"] = 0
                            POD_FIELDS["Damage Qty"] = 0
                            POD_FIELDS["Over Qty"] = 0
                            POD_FIELDS["Received Qty"] = stamp_total_received
                        else:
                            # Has discrepancies - convert each quantity safely
                            POD_FIELDS["Refused Qty"] = safe_int_conversion(bol_data["stamp"]["refused"], 0)
                            POD_FIELDS["Short Qty"] = safe_int_conversion(bol_data["stamp"]["short"], 0)
                            POD_FIELDS["Damage Qty"] = safe_int_conversion(bol_data["stamp"]["damage"], 0)
                            POD_FIELDS["Over Qty"] = safe_int_conversion(bol_data["stamp"]["over"], 0)
                            POD_FIELDS["Received Qty"] = stamp_total_received
                    else:
                        # No total received - still convert quantities safely
                        POD_FIELDS["Refused Qty"] = safe_int_conversion(bol_data["stamp"]["refused"], 0)
                        POD_FIELDS["Short Qty"] = safe_int_conversion(bol_data["stamp"]["short"], 0)
                        POD_FIELDS["Damage Qty"] = safe_int_conversion(bol_data["stamp"]["damage"], 0)
                        POD_FIELDS["Over Qty"] = safe_int_conversion(bol_data["stamp"]["over"], 0)
                        POD_FIELDS["Received Qty"] = safe_str_conversion(bol_data["stamp"]["total_received"], "null")
                    #______________________________________________________________________________________________________________________________________________FIX-18
    
    
                    # Updated check with type handling
                    received_qty = str(POD_FIELDS['Received Qty']).lower()
                    issued_qty = str(POD_FIELDS['Issued Qty']).lower()

                    print(f"End of post_process: {datetime.now()}")

                    
                    # Check and handle invalid 'Received Qty'
                        #__________________________________________________________________________________________________________________________________________________________FIX-18
                    if received_qty in ["null", "empty", "", "n/a", "0"] and issued_qty not in ["null", "empty", "", "n/a", "0"]:
                        if bol_data["stamp"]["stamp_exist"] == "yes" or bol_data["stamp"]["notation_exist"] == "yes":
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
                                POD_FIELDS['Received Qty'] = "==null"
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