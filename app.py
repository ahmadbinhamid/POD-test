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
from utils import convert_image_to_base64

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

def check_gpu_memory() -> bool:
    """
    Check if GPU memory is low
    Returns True if memory is low, False otherwise
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Get memory info for GPU 0
            mem_free = torch.cuda.mem_get_info(0)[0]
            mem_total = torch.cuda.mem_get_info(0)[1]
            usage_percent = ((mem_total - mem_free) / mem_total) * 100
            
            logger.info(f"GPU Memory Usage: {usage_percent:.1f}%")
            
            # Consider low if >85% used
            return usage_percent > 85
        else:
            logger.warning("CUDA not available, assuming sufficient memory")
            return False
    except Exception as e:
        logger.error(f"Error checking GPU memory: {e}")
        return False

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

# reprocess un-registered docs
@app.post("/api/ocr/reprocess", response_model=EnhancedOCRResponse)
async def reprocess_unregistered_document(request: EnhancedOCRRequest):
    """
    🔄 REPROCESS UNREGISTERED DOCUMENT WITH ASSIGNED TEMPLATE
    
    Flow:
    1. Document was previously unregistered (no match)
    2. Admin reviewed suggestions and assigned a template
    3. Frontend sends document + template_id
    4. AI skips matching, uses provided template directly
    
    Request Body:
    {
        "file_url": "https://example.com/document.pdf",
        "user_id": "user123",
        "template_id": "693a48d6527f2aeee281245a"  ← REQUIRED
    }
    """
    start_time = time.time()
    
    logger.info(f"""
        ╔═══════════════════════════════════════════════════════════╗
        ║ 🔄 REPROCESSING API - UNREGISTERED DOCUMENT               ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ File URL:     {request.file_url:<40} ║
        ║ User ID:      {request.user_id or 'N/A':<40} ║
        ║ Template ID:  {request.template_id or 'N/A':<40} ║
        ╚═══════════════════════════════════════════════════════════╝
            """)
    
    try:
        # ========== STEP 1: VALIDATE TEMPLATE_ID ==========
        if not request.template_id:
            logger.error("❌ Template ID missing in reprocess request")
            return create_error_response("Template ID is required for reprocessing")
        
        logger.info(f"✓ Step 1/6: Template ID validated: {request.template_id}")
        
        # ========== STEP 2: GET TEMPLATE FROM MEMORY ==========
        template = template_store.get_template(request.template_id)
        
        if not template:
            logger.error(f"❌ Template {request.template_id} not found in memory")
            logger.info(f"Available templates: {list(template_store._templates.keys())}")
            return create_unregistered_response(
                f"Template {request.template_id} not found in memory",
                start_time
            )
        
        logger.info(f"""
        ✓ Step 2/6: Template retrieved from memory
        Template Name: {template.get('template_name', 'Unknown')}
        Category: {template.get('category', 'Unknown')}
        Version: {template.get('version', 'Unknown')}
                """)
        
        # ========== STEP 3: DOWNLOAD AND PROCESS PDF ==========
        logger.info(f"→ Step 3/6: Downloading PDF from {request.file_url}")
        
        from main import pdf_to_images
        images = await pdf_to_images(request.file_url)
        
        if not images:
            logger.error("❌ Failed to process PDF")
            return create_unregistered_response("Failed to process PDF", start_time)
        
        image = images[0]  # Process first page
        logger.info(f"✓ Step 3/6: PDF processed - {len(images)} page(s), using page 1")
        
        # ========== STEP 4: PROCESS WITH ASSIGNED TEMPLATE ==========
        logger.info(f"→ Step 4/6: Processing with assigned template (SKIP MATCHING)")
        
        result = await process_document_with_template(
            file_url=request.file_url,
            template=template,
            user_id=request.user_id,
            image=image
        )
        
        logger.info(f"✓ Step 4/6: Document processed successfully")
        
        # ========== STEP 5: GET CLASSIFICATION FOR METADATA ==========
        logger.info(f"→ Step 5/6: Running classification for metadata")
        category, primary_confidence = classify_document_category(image)
        logger.info(f"✓ Step 5/6: Classification: {category} (confidence: {primary_confidence:.2f})")
        
        # ========== STEP 6: ADD METADATA ==========
        processing_time = int((time.time() - start_time) * 1000)
        result.template_id = template.get("template_id")
        result.confidence = 1.0  # Direct assignment = 100% confidence
        result.processing_time = processing_time
        result.classification_details = ClassificationDetails(
            primary_model_prediction=category,
            primary_confidence=primary_confidence
        )
        result.suggested_templates = []  # No suggestions needed for reprocessing
        
        logger.info(f"""
        ╔═══════════════════════════════════════════════════════════╗
        ║ ✅ REPROCESSING COMPLETE                                  ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ Template Used:    {template.get('template_name', 'Unknown'):<36} ║
        ║ Processing Time:  {processing_time:<36} ms ║
        ║ Confidence:       100% (direct assignment)               ║
        ╚═══════════════════════════════════════════════════════════╝
                """)
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Error in reprocessing: {e}")
        logger.exception("Full traceback:")
        return create_error_response(str(e))

async def process_single_document_enhanced(request: EnhancedOCRRequest) -> EnhancedOCRResponse:
    """
    🚀 PROCESS SINGLE DOCUMENT - ENHANCED WITH TEMPLATE SUPPORT
    
    Flow:
    1. Download & process PDF
    2. Primary classification (YOLO)
    3. Template matching OR use provided template_id
    4. Region detection
    5. Prompt loading
    6. OCR extraction
    7. Field mapping & post-processing
    
    Args:
        request: EnhancedOCRRequest with file_url, user_id, optional template_id
    
    Returns:
        EnhancedOCRResponse with extracted data + template info
    """
    start_time = time.time()
    
    logger.info(f"""
        ╔═══════════════════════════════════════════════════════════╗
        ║ 🚀 ENHANCED OCR PROCESSING START                          ║
        ╠═══════════════════════════════════════════════════════════╣
        ║ File URL:     {request.file_url[:50]:<40}... ║
        ║ User ID:      {request.user_id or 'N/A':<42} ║
        ║ Template ID:  {request.template_id or 'Auto-Match':<42} ║
        ║ Mode:         {'Manual (Unregistered)' if request.template_id else 'Auto-Match':<42} ║
        ╚═══════════════════════════════════════════════════════════╝
            """)
    
    try:
        # Download and process document
        from main import pdf_to_images
        images = await pdf_to_images(request.file_url)
        
        if not images:
            return create_unregistered_response("Failed to process PDF", start_time)
        
        image = images[0]  # Process first page
        logger.info(f"✓ Step 1/7: PDF downloaded and processed ({len(images)} pages)")
        
        # Step 1: Primary classification with YOLO
        category, primary_confidence = classify_document_category(image)
        logger.info(f"✓ Step 2/7: Primary classification: {category} (confidence: {primary_confidence:.2f})")
        
        # Step 2: Template handling
        if request.template_id:
            # Case: Unregistered document with provided template_id
            logger.info(f"✓ Step 3/7: Using provided template: {request.template_id} (Manual mode)")
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
            logger.info(f"→ Step 3/7: Matching templates for category: {category}")
            
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
            
            # Check if template exists BEFORE logging
            if not template:
                logger.info(f"No template matched (best score: {suggested_templates[0]['match_score'] if suggested_templates else 0})")
                return create_unregistered_response(
                    "No template matched with confidence >= threshold",
                    start_time,
                    category,
                    primary_confidence,
                    suggested_templates
                )
            
            # Safe to log now (template exists)
            logger.info(f"✓ Step 3/7: Template matched: {template.get('template_id')} (confidence: {template_confidence:.2f})")
        
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
        logger.exception("Full traceback:")
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
    
    FIXED: Correctly uses batch_ocr(ocr_batches: List[models.OCRBatch])
    """
    logger.info("\n" + "=" * 70)
    logger.info("🔧 PROCESS_WITH_TEMPLATE: Starting template-driven processing")
    logger.info("=" * 70)
    logger.info(f"  File URL: {file_url}")
    logger.info(f"  Template ID: {template.get('template_id')}")
    logger.info(f"  Template Name: {template.get('template_name')}")
    logger.info(f"  Skip Matching: {skip_matching}")
    
    try:
        # ========== STEP 1: GET IMAGE ==========
        if image is None:
            logger.info("\n  📥 STEP 1: Downloading PDF...")
            from main import pdf_to_images
            images = await pdf_to_images(file_url)
            if not images:
                raise Exception("Failed to process PDF")
            image = images[0]
            logger.info(f"  ✅ PDF processed: {len(images)} page(s)")
        else:
            logger.info("\n  ✅ STEP 1: Using provided image")
        
        # Get category
        category = template.get("category", "BOL")
        logger.info(f"  📁 Category: {category}")
        
        # ========== STEP 2: REGION DETECTION ==========
        logger.info("\n  🔍 STEP 2: Region Detection")
        detected_regions = region_detector.detect_regions(image, template, category)
        
        if not detected_regions:
            logger.warning("  ⚠️  No regions detected, using full image")
            detected_regions = {"full_document": image}
        
        logger.info(f"  ✅ Detected {len(detected_regions)} region(s):")
        for region_name in detected_regions.keys():
            logger.info(f"    - {region_name}")
        
        # ========== STEP 3: LOAD PROMPTS ==========
        logger.info("\n  📝 STEP 3: Prompt Loading")
        prompts = prompt_loader.build_batch_prompts(template, detected_regions)
        logger.info(f"  ✅ Loaded {len(prompts)} prompt(s):")
        for region_name in prompts.keys():
            logger.info(f"    - {region_name}")
        
        # ========== STEP 4: BUILD OCR BATCHES ==========
        logger.info("\n  📦 STEP 4: Building OCR Batches")
        
        from utils import convert_image_to_base64
        from main import batch_ocr
        import models
        
        ocr_batches = []
        
        for region_name, region_image in detected_regions.items():
            if region_name not in prompts:
                logger.warning(f"    ⚠️  No prompt for region: {region_name}, skipping")
                continue
            
            # Convert region image to base64
            if isinstance(region_image, np.ndarray):
                from PIL import Image
                region_pil = Image.fromarray(region_image)
                region_base64 = convert_image_to_base64(region_pil)
            else:
                region_base64 = convert_image_to_base64(region_image)
            
            # Get prompt for this region
            prompt = prompts[region_name]
            
            # Create OCRBatch object matching main.py signature
            ocr_batch = models.OCRBatch(
                page_type=category,  # e.g., "BOL"
                region_name=region_name,  # e.g., "stamp"
                prompt=prompt,
                image=region_base64,
                stamp_exist="unknown"  # Default value
            )
            
            ocr_batches.append(ocr_batch)
            logger.info(f"    ✓ Added batch for region: {region_name}")
        
        logger.info(f"  ✅ Built {len(ocr_batches)} OCR batch(es)")
        
        # ========== STEP 5: RUN OCR ==========
        logger.info("\n  🤖 STEP 5: Running OCR Extraction")
        
        if not ocr_batches:
            logger.error("  ❌ No OCR batches to process!")
            raise Exception("No valid regions/prompts for OCR processing")
        
        # Call batch_ocr with correct signature
        ocr_responses = await batch_ocr(ocr_batches)
        
        logger.info(f"  ✅ OCR complete: {len(ocr_responses)} response(s)")
        
        # ========== STEP 6: EXTRACT DATA FROM RESPONSES ==========
        logger.info("\n  📊 STEP 6: Extracting Data from OCR Responses")
        
        extracted_data = {}
        
        for ocr_response in ocr_responses:
            region_name = ocr_response.region_name
            ocr_result = ocr_response.ocr_response
            
            logger.info(f"    Processing response for: {region_name}")
            
            # Store with region prefix (e.g., "stamp.total_received")
            for field, value in ocr_result.items():
                key = f"{region_name}.{field}"
                extracted_data[key] = value
                logger.debug(f"      {key} = {value}")
            
            logger.info(f"      ✓ Extracted {len(ocr_result)} fields from {region_name}")
        
        logger.info(f"  ✅ Total extracted fields: {len(extracted_data)}")
        
        # Log extracted data for debugging
        logger.info("\n  🔍 EXTRACTED DATA SUMMARY:")
        for key, value in extracted_data.items():
            logger.info(f"    {key}: {value}")
        
        # ========== STEP 7: APPLY FIELD MAPPING ==========
        logger.info("\n  🗺️  STEP 7: Field Mapping")
        mapped_data = post_processor.apply_field_mapping(extracted_data, template)
        logger.info(f"  ✅ Mapped {len(mapped_data)} fields")
        
        # Log mapped data
        logger.info("\n  🔍 MAPPED DATA SUMMARY:")
        for key, value in mapped_data.items():
            logger.info(f"    {key}: {value}")
        
        # ========== STEP 8: APPLY POST-PROCESSING ==========
        logger.info("\n  🔧 STEP 8: Post-Processing Rules")
        final_data = post_processor.apply_rules(mapped_data, template)
        logger.info(f"  ✅ Post-processing complete")
        
        # ========== STEP 9: CREATE RESPONSE MODEL ==========
        logger.info("\n  📦 STEP 9: Creating Response Model")
        
        # Ensure all required fields exist with defaults
        required_fields = {
            "B_L_Number": "",
            "Stamp_Exists": "",
            "Seal_Intact": "null",
            "POD_Date": "",
            "Signature_Exists": "",
            "Issued_Qty": 0,
            "Received_Qty": 0,
            "Damage_Qty": "0",
            "Short_Qty": "0",
            "Over_Qty": "0",
            "Refused_Qty": "null",
            "Customer_Order_Num": "null",
            "Notation_Exists": "null",
            "Status": "valid"
        }
        
        # Merge final_data with defaults
        for field, default_value in required_fields.items():
            if field not in final_data or final_data[field] is None:
                final_data[field] = default_value
        
        logger.info(f"  ✅ Response model ready")
        
        # Log final extracted values
        logger.info("\n  📋 FINAL EXTRACTED VALUES:")
        logger.info(f"    B/L Number: {final_data.get('B_L_Number')}")
        logger.info(f"    Received Qty: {final_data.get('Received_Qty')}")
        logger.info(f"    POD Date: {final_data.get('POD_Date')}")
        logger.info(f"    Stamp Exists: {final_data.get('Stamp_Exists')}")
        logger.info(f"    Signature Exists: {final_data.get('Signature_Exists')}")
        logger.info(f"    Damage Qty: {final_data.get('Damage_Qty')}")
        logger.info(f"    Short Qty: {final_data.get('Short_Qty')}")
        logger.info(f"    Over Qty: {final_data.get('Over_Qty')}")
        
        response = EnhancedOCRResponse(**final_data)
        
        logger.info("=" * 70)
        logger.info("✅ PROCESS_WITH_TEMPLATE: Processing complete")
        logger.info("=" * 70 + "\n")
        
        return response
    
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"❌ Error processing with template: {e}")
        logger.exception("Full traceback:")
        logger.error("=" * 70 + "\n")
        raise


def classify_document_category(image) -> Tuple[str, float]:
    """
    Classify document using existing YOLO model
    Returns (category, confidence)
    
    FIXED: Handles Dict[str, List[str]] response correctly
    
    Args:
        image: Either PIL Image, numpy array, or base64 string
    
    Returns:
        Tuple of (category: str, confidence: float)
    """
    logger.info("\n" + "=" * 70)
    logger.info("📄 DOCUMENT CLASSIFICATION")
    logger.info("=" * 70)
    
    try:
        from main import classify_documents
        from PIL import Image
        import numpy as np
        
        # Ensure image is base64 string
        if not isinstance(image, str):
            # Convert PIL/numpy to base64
            logger.info("  Converting image to base64...")
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
                image = convert_image_to_base64(pil_image)
            elif isinstance(image, Image.Image):
                image = convert_image_to_base64(image)
        
        logger.info("  Calling classify_documents()...")
        
        # Call classify_documents (returns Dict[str, List[str]])
        # Format: {"BOL": ["img1"], "receipt": [], "others": []}
        results = classify_documents([image])
        
        logger.info(f"  Classification results: {list(results.keys())}")
        logger.info(f"  Results detail:")
        for category, images in results.items():
            logger.info(f"    - {category}: {len(images)} image(s)")
        
        # Find which category has images
        for category, images in results.items():
            if images and len(images) > 0:
                # Found a match!
                logger.info(f"  ✅ Document classified as: {category}")
                logger.info("=" * 70 + "\n")
                return category, 0.95  # High confidence since YOLO matched
        
        # No category matched
        logger.warning("  ⚠️  No classification match, defaulting to 'others'")
        logger.info("=" * 70 + "\n")
        return "others", 0.0
    
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"❌ Error in classification: {e}")
        logger.exception("Full traceback:")
        logger.error("=" * 70 + "\n")
        return "others", 0.0


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