from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from typing import Optional, List, Dict, Any, Tuple
from langsmith import traceable
from dotenv import load_dotenv
import os
import asyncio
from ultralytics import YOLO
import concurrent.futures
import logging

import models
from models import (
    OCRRequest, 
    PODResponse,
    TemplateSyncRequest,
    TemplateSyncResponse,
    EnhancedOCRRequest,
    EnhancedOCRResponse,
    TemplateTestRequest,
    TemplateTestResponse,
    ClassificationDetails,
    SuggestedTemplate,
    MemoryStoreStats
)
import time
import numpy as np
from main import run_ocr

from template_engine import get_template_store, TemplateMatcher, RegionDetector, PromptLoader, PostProcessor

load_dotenv()

app = FastAPI(default_response_class=ORJSONResponse)

# Basic structured logging for API
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

# Template engine components
template_store = get_template_store()
template_matcher = TemplateMatcher()
region_detector = RegionDetector()
prompt_loader = PromptLoader()
post_processor = PostProcessor()

logger.info("Template engine components initialized")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== GLOBALS ==========
# Note: Ensure your YOLO model loading is thread-safe if using multiple workers.
# For Uvicorn with workers=1, this is fine. For >1, look into proper model management.
YOLO_MODEL = YOLO("Models/doc_classify_best.pt")
YOLO_QUEUE = asyncio.Queue()

# ========== GPU Inference Background Worker ==========
async def gpu_worker():
    while True:
        job = await YOLO_QUEUE.get()
        try:
            images = job["images"]
            conf = job.get("conf", 0.55)
            batch = job.get("batch", len(images))
            result = YOLO_MODEL.predict(images, conf=conf, batch=batch)
            job["future"].set_result(result)
        except Exception as e:
            job["future"].set_exception(e)
        finally:
            YOLO_QUEUE.task_done()
 
# Launch the GPU worker(s) at startup
@app.on_event("startup")
async def startup_event():
    # Start a few workers to process GPU tasks from the queue concurrently
    # The number of workers can be tuned based on your GPU's capability
    num_workers = int(os.environ.get("GPU_WORKERS", 4))
    for _ in range(num_workers):
        asyncio.create_task(gpu_worker())

# Shutdown handler to clean up resources
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down ThreadPoolExecutor...")
    CPU_POOL.shutdown(wait=True, cancel_futures=False)
    logger.info("ThreadPoolExecutor shut down complete")

# This section with commented-out code seems unused in your final endpoint.
# You can remove it for clarity if it's no longer needed.
# # ========== CLASSIFY via YOLO QUEUE ==========
# async def classify_with_gpu_queue(images, conf=0.55, batch=5):
#     ...
# # ========== OCR Endpoint (Upload + Classify + OCR) ==========
# @app.post("/ocr")
# async def handle_user_ocr(file: UploadFile = File(...)):
#     ...
# @app.on_event("startup")
# async def load_models():
#     ...

# ====================================================================
#  START: MODIFIED SECTION FOR CONCURRENT PROCESSING
# ====================================================================

CPU_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())

def _run_ocr_sync(path: str):
    """sync wrapper: execute the async pipeline in this worker thread"""
    return asyncio.run(run_ocr(path))

async def run_ocr_async(path: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(CPU_POOL, _run_ocr_sync, path)

async def process_single_ocr_request(ocr_request: OCRRequest) -> List[PODResponse]:
    """
    Handles the complete processing for a single OCR request.
    This function contains the logic from your original endpoint.
    """
    try:
        # 1. Run the core OCR and analysis logic for one file
        logger.info(f"\n\n🎯 STARTING SINGLE OCR REQUEST\nRequest ID: {ocr_request.id}\nFile path: {ocr_request.file_url_or_path}")
        result_items = await run_ocr_async(ocr_request.file_url_or_path)
        logger.info(f"\n\n✅ COMPLETED SINGLE OCR REQUEST\nRequest ID: {ocr_request.id}\nItems generated: {len(result_items)}\n\n")

        # 2. Map the results to the PODResponse model, adding the specific _id
        responses = []
        for item in result_items:
            logger.info(f"\n\n📋 FINAL POD FIELDS FOR REQUEST {ocr_request.id}\nB/L Number: {item.get('B/L Number')}\nStamp Exists: {item.get('Stamp Exists')}\nPOD Date: {item.get('POD Date')}\nReceived Qty: {item.get('Received Qty')}\nDamage Qty: {item.get('Damage Qty')}\nShort Qty: {item.get('Short Qty')}\nOver Qty: {item.get('Over Qty')}\nRefused Qty: {item.get('Refused Qty')}\nStatus: {item.get('Status')}\nNotation Exists: {item.get('Notation Exists')}\n\n")
            response_dict = {
                "B_L_Number": item.get("B/L Number"),
                "Stamp_Exists": item.get("Stamp Exists"),
                "Seal_Intact": item.get("Seal Intact"),
                "POD_Date": item.get("POD Date"),
                "Signature_Exists": item.get("Signature Exists"),
                "Issued_Qty": item.get("Issued Qty"),
                "Received_Qty": item.get("Received Qty"),
                "Damage_Qty": item.get("Damage Qty"),
                "Short_Qty": item.get("Short Qty"),
                "Over_Qty": item.get("Over Qty"),
                "Refused_Qty": item.get("Refused Qty"),
                "Customer_Order_Num": item.get("Customer Order Num"),
                "Notation_Exists": item.get("Notation Exists"),
                "Status": item.get("Status"),
                "_id": ocr_request.id  # Attach the ID from the original request
            }
            responses.append(PODResponse.model_validate(response_dict))
        
        return responses

    except Exception as e:
        # Log the error with context and re-raise it so asyncio.gather can catch it.
        # This prevents one failed file from crashing the entire batch.
        logger.exception(f"Error processing request id={ocr_request.id}: {e}")
        # Re-raising the exception is important for the gathering logic
        raise e


@traceable(run_type="chain", name="api_batch_ocr")

# ============================================================================
# 🆕 STEP 3: ADD NEW ENDPOINT - Template Sync
# ============================================================================

@app.post("/api/templates/sync", response_model=TemplateSyncResponse)
async def sync_template(request: TemplateSyncRequest):
    """
    Sync template with AI memory store
    Called by Backend when template status changes
    
    Actions:
    - 'add': Add new active template to memory
    - 'remove': Remove inactive/deprecated template from memory
    - 'update': Update existing template in memory
    """
    try:
        action = request.action.lower()
        
        if action == "add" or action == "update":
            if not request.template:
                return TemplateSyncResponse(
                    success=False,
                    message="Template data required for add/update action"
                )
            
            success = template_store.add_template(request.template)
            template_id = request.template.get("template_id")
            
            if success:
                stats = template_store.get_stats()
                return TemplateSyncResponse(
                    success=True,
                    message=f"Template {template_id} {'added' if action == 'add' else 'updated'} successfully",
                    template_id=template_id,
                    stats=stats
                )
            else:
                return TemplateSyncResponse(
                    success=False,
                    message=f"Failed to {action} template"
                )
        
        elif action == "remove":
            if not request.template_id:
                return TemplateSyncResponse(
                    success=False,
                    message="Template ID required for remove action"
                )
            
            success = template_store.remove_template(request.template_id)
            
            if success:
                stats = template_store.get_stats()
                return TemplateSyncResponse(
                    success=True,
                    message=f"Template {request.template_id} removed successfully",
                    template_id=request.template_id,
                    stats=stats
                )
            else:
                return TemplateSyncResponse(
                    success=False,
                    message=f"Template {request.template_id} not found in memory"
                )
        
        else:
            return TemplateSyncResponse(
                success=False,
                message=f"Unknown action: {action}. Use 'add', 'remove', or 'update'"
            )
    
    except Exception as e:
        logger.error(f"Error in template sync: {e}")
        return TemplateSyncResponse(
            success=False,
            message=f"Error: {str(e)}"
        )


# ============================================================================
# 🆕 STEP 4: ADD NEW ENDPOINT - Template Testing
# ============================================================================

@app.post("/api/templates/test", response_model=TemplateTestResponse)
async def test_template(request: TemplateTestRequest):
    """
    Test a template that is NOT active/NOT in memory
    Used by Frontend for testing templates before activation
    
    Frontend sends complete template JSON
    AI processes document with provided template (skips matching)
    """
    start_time = time.time()
    
    try:
        # Process document with provided template
        result = await process_document_with_template(
            file_url=request.file_url,
            template=request.template,
            user_id=request.user_id,
            skip_matching=True
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return TemplateTestResponse(
            success=True,
            extracted_data=result,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error in template testing: {e}")
        processing_time = int((time.time() - start_time) * 1000)
        
        return TemplateTestResponse(
            success=False,
            error=str(e),
            processing_time=processing_time
        )


# ============================================================================
# 🆕 STEP 5: ADD NEW ENDPOINT - Memory Store Stats
# ============================================================================

@app.get("/api/templates/stats", response_model=MemoryStoreStats)
async def get_memory_stats():
    """
    Get statistics about templates in memory
    Useful for debugging and monitoring
    """
    stats = template_store.get_stats()
    return MemoryStoreStats(**stats)


@app.post("/run-ocr")
async def analyze_images_concurrently(ocr_requests: List[EnhancedOCRRequest]):
    """
    MODIFIED: Enhanced OCR endpoint with template support
    
    Handles 3 cases:
    1. Normal flow: Match template, process if score >= 0.75
    2. Unregistered doc: template_id in request (skip matching)
    3. No match: Return suggested templates
    """
    results = []
    
    # Check GPU memory
    is_low_memory = check_gpu_memory()
    
    if is_low_memory:
        logger.warning("Low GPU memory detected, processing sequentially")
        for request in ocr_requests:
            try:
                result = await process_single_document_enhanced(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                results.append(create_error_response(str(e)))
    else:
        # Concurrent processing
        tasks = [process_single_document_enhanced(req) for req in ocr_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = [
            res if not isinstance(res, Exception) else create_error_response(str(res))
            for res in results
        ]
    
    return results

async def process_single_document_enhanced(request: EnhancedOCRRequest) -> EnhancedOCRResponse:
    """
    Process single document with template support
    
    Flow:
    1. If template_id provided → Skip matching, use that template
    2. Else → Match templates, use best match if score >= 0.75
    3. If no match → Return unregistered document with suggestions
    """
    start_time = time.time()
    
    try:
        # Download and process document
        from main import pdf_to_images
        images = pdf_to_images(request.file_url)
        
        if not images:
            return create_unregistered_response("Failed to process PDF", start_time)
        
        image = images[0]  # Process first page
        
        # Step 1: Primary classification with YOLO
        category, primary_confidence = classify_document_category(image)
        
        # Step 2: Template handling
        if request.template_id:
            # Case: Unregistered document with provided template_id
            logger.info(f"Processing with provided template: {request.template_id}")
            template = template_store.get_template(request.template_id)
            
            if not template:
                return create_unregistered_response(
                    f"Template {request.template_id} not found in memory",
                    start_time
                )
            
            template_confidence = 1.0  # Direct assignment
            suggested_templates = []
        else:
            # Case: Normal flow - Match templates
            templates = template_store.get_templates_by_category(category)
            
            if not templates:
                logger.warning(f"No templates found for category: {category}")
                return create_unregistered_response(
                    f"No templates available for category: {category}",
                    start_time,
                    category,
                    primary_confidence
                )
            
            # Match templates
            template, template_confidence, suggested_templates = template_matcher.match_templates(
                image=image,
                templates=templates
            )
            
            if not template:
                logger.info(f"No template matched (best score: {suggested_templates[0]['match_score'] if suggested_templates else 0})")
                return create_unregistered_response(
                    "No template matched with confidence >= 0.75",
                    start_time,
                    category,
                    primary_confidence,
                    suggested_templates
                )
        
        # Step 3: Process with matched template
        result = await process_document_with_template(
            file_url=request.file_url,
            template=template,
            user_id=request.user_id,
            image=image
        )
        
        # Add metadata
        processing_time = int((time.time() - start_time) * 1000)
        result.template_id = template.get("template_id")
        result.confidence = template_confidence
        result.processing_time = processing_time
        result.classification_details = ClassificationDetails(
            primary_model_prediction=category,
            primary_confidence=primary_confidence
        )
        result.suggested_templates = suggested_templates
        
        return result
    
    except Exception as e:
        logger.error(f"Error in enhanced processing: {e}")
        return create_error_response(str(e))


async def process_document_with_template(
    file_url: str,
    template: Dict,
    user_id: Optional[str] = None,
    image: Optional[np.ndarray] = None,
    skip_matching: bool = False
) -> EnhancedOCRResponse:
    """
    Process document using specific template
    Used for both matched templates and template testing
    """
    try:
        # Get image if not provided
        if image is None:
            from main import pdf_to_images
            images = pdf_to_images(file_url)
            if not images:
                raise Exception("Failed to process PDF")
            image = images[0]
        
        # Get category
        category = template.get("category", "BOL")
        
        # Step 1: Region detection using template config
        detected_regions = region_detector.detect_regions(image, template, category)
        
        if not detected_regions:
            logger.warning("No regions detected, using full image")
            detected_regions = {"full_document": image}
        
        # Step 2: Load prompts from template
        prompts = prompt_loader.build_batch_prompts(template, detected_regions)
        
        # Step 3: OCR extraction (use existing batch_ocr from main.py)
        from main import batch_ocr
        
        extracted_data = await batch_ocr(
            regions=detected_regions,
            prompts=prompts
        )
        
        # Step 4: Apply field mapping
        mapped_data = post_processor.apply_field_mapping(extracted_data, template)
        
        # Step 5: Apply post-processing rules
        final_data = post_processor.apply_rules(mapped_data, template)
        
        # Convert to response model
        response = EnhancedOCRResponse(**final_data)
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing with template: {e}")
        raise


def classify_document_category(image: np.ndarray) -> Tuple[str, float]:
    """
    Classify document using existing YOLO model
    Returns (category, confidence)
    """
    try:
        from main import classify_documents
        
        results = classify_documents([image])
        
        if results and len(results) > 0:
            category = results[0].get("category", "Others")
            confidence = results[0].get("confidence", 0.0)
            return category, confidence
        
        return "Others", 0.0
    
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return "Others", 0.0


def create_unregistered_response(
    message: str,
    start_time: float,
    category: str = "Others",
    primary_confidence: float = 0.0,
    suggested_templates: List[Dict] = None
) -> EnhancedOCRResponse:
    """
    Create response for unregistered documents
    All data fields are null/0, suggested templates included
    """
    processing_time = int((time.time() - start_time) * 1000)
    
    return EnhancedOCRResponse(
        B_L_Number="",
        Stamp_Exists="",
        Seal_Intact="null",
        POD_Date="",
        Signature_Exists="",
        Issued_Qty=0,
        Received_Qty=0,
        Damage_Qty="0",
        Short_Qty="0",
        Over_Qty="0",
        Refused_Qty="null",
        Customer_Order_Num="null",
        template_id=None,
        confidence=0.0,
        processing_time=processing_time,
        classification_details=ClassificationDetails(
            primary_model_prediction=category,
            primary_confidence=primary_confidence
        ),
        suggested_templates=suggested_templates or []
    )


def create_error_response(error_message: str) -> EnhancedOCRResponse:
    """Create error response"""
    return EnhancedOCRResponse(
        B_L_Number="",
        Stamp_Exists="",
        Seal_Intact="null",
        POD_Date="",
        Signature_Exists="",
        Issued_Qty=0,
        Received_Qty=0,
        Damage_Qty="0",
        Short_Qty="0",
        Over_Qty="0",
        Refused_Qty="null",
        Customer_Order_Num="null",
        template_id=None,
        confidence=0.0,
        processing_time=0,
        classification_details=None,
        suggested_templates=[]
    )

# ========== Run Server ==========
if __name__ == "__main__":
    import uvicorn
    # To leverage multiple CPU cores for non-async blocking tasks (like some parts of image processing),
    # you can use multiple workers. However, for a purely async application, a single worker is often sufficient.
    # The `uvicorn.run` with `workers=1` is generally recommended for async FastAPI apps.
    uvicorn.run("app:app", host="0.0.0.0", port=8080, workers=1)