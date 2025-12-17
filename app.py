# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import ORJSONResponse
# from typing import List
# from langsmith import traceable
# from dotenv import load_dotenv
# import os
# import asyncio
# from ultralytics import YOLO

# import models
# from models import OCRRequest, PODResponse
# from main import run_ocr  # We'll replace run_ocr_parallel

# load_dotenv()

# app = FastAPI(default_response_class=ORJSONResponse)

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ========== GLOBALS ==========
# YOLO_MODEL = YOLO("doc_classify_best.pt")
# YOLO_QUEUE = asyncio.Queue()
# def get_classifier():
#     return YOLO_MODEL
# # ========== GPU Inference Background Worker ==========
# async def gpu_worker():
#     while True:
#         job = await YOLO_QUEUE.get()
#         try:
#             images = job["images"]
#             conf = job.get("conf", 0.55)
#             batch = job.get("batch", len(images))
#             result = YOLO_MODEL.predict(images, conf=conf, batch=batch)
#             job["future"].set_result(result)
#         except Exception as e:
#             job["future"].set_exception(e)
#         YOLO_QUEUE.task_done()

# # Launch the GPU worker at startup
# @app.on_event("startup")
# async def startup_event():  
    
#     # asyncio.create_task(gpu_worker())
#     for _ in range(4):
        
#         asyncio.create_task(gpu_worker())

# # ========== CLASSIFY via YOLO QUEUE ==========
# async def classify_with_gpu_queue(images, conf=0.55, batch=5):
#     loop = asyncio.get_event_loop()
#     future = loop.create_future()
#     await YOLO_QUEUE.put({
#         "images": images,
#         "conf": conf,
#         "batch": batch,
#         "future": future
#     })
#     return await future

# # # ========== OCR Endpoint (Upload + Classify + OCR) ==========
# # @app.post("/ocr")
# # async def handle_user_ocr(file: UploadFile = File(...)):
# #     try:
# #         os.makedirs("./tmp", exist_ok=True)
# #         file_path = os.path.join("./tmp", file.filename)
# #         with open(file_path, "wb") as f:
# #             f.write(await file.read())

# #         # 🔁 Call your OCR logic with GPU batching here
# #         result = await run_ocr(file_path, classify_with_gpu_queue)  # <-- use new GPU batch logic
        
# #         return {"status": "success", "data": result}

# #     except Exception as e:
# #         return {"status": "error", "message": str(e)}


# # @app.on_event("startup")
# # async def load_models():
# #     print("⚙️  Loading models on startup...")

# #     app.state.bol_region_detector = YOLO("bol_regions_best.pt").to("cpu")
# #     app.state.page_classifier = YOLO("doc_classify_best.pt").to("cpu")
# #     app.state.receipt_region_detector = YOLO("receipts_regions_best.pt").to("cpu")
# #     app.state.stamp_detector = YOLO("stamp_existence.pt").to("cpu")

# #     # Page orientation model
# #     model = mobilenet_v3_small_page_orientation(
# #         pretrained=False, pretrained_backbone=False
# #     )
# #     params = torch.load("page_orientation.pt", map_location="cpu")
# #     model.load_state_dict(params)
# #     app.state.page_orient_predictor = page_orientation_predictor(arch=model, pretrained=False)

# #     print("✅ Models loaded.")
    
# # ========== URL-Based OCR ==========
# @traceable(run_type="chain", name="api")
# @app.post("/run-ocr", response_model=List[PODResponse])
# async def analyze_image(ocr_request: OCRRequest):
#     try:
#         result = await run_ocr(ocr_request.file_url_or_path)  # pass the inference method

#         responses = []
#         for item in result:
#             response_dict = {
#                 "B_L_Number": item.get("B/L Number"),
#                 "Stamp_Exists": item.get("Stamp Exists"),
#                 "Seal_Intact": item.get("Seal Intact"),
#                 "POD_Date": item.get("POD Date"),
#                 "Signature_Exists": item.get("Signature Exists"),
#                 "Issued_Qty": item.get("Issued Qty"),
#                 "Received_Qty": item.get("Received Qty"),
#                 "Damage_Qty": item.get("Damage Qty"),
#                 "Short_Qty": item.get("Short Qty"),
#                 "Over_Qty": item.get("Over Qty"),
#                 "Refused_Qty": item.get("Refused Qty"),
#                 "Customer_Order_Num": item.get("Customer Order Num"),
#                 "Status": item.get("Status"),
#                 "_id": ocr_request.id
#                 # "FILE_DATA": ocr_request.FILE_DATA
#             }
#             responses.append(PODResponse.model_validate(response_dict))

#         return responses

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # ========== Run Server ==========
# if __name__ == "__main__":
#     import uvicorn
#     # uvicorn.run("app:app", host="0.0.0.0", port=8080, workers=4)
#     uvicorn.run("app:app", host="0.0.0.0", port=8080, workers=1)




from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from typing import List
from langsmith import traceable
from dotenv import load_dotenv
import os
import asyncio
from ultralytics import YOLO

import concurrent.futures
import logging


import models
from models import OCRRequest, PODResponse
from main import run_ocr

load_dotenv()

app = FastAPI(default_response_class=ORJSONResponse)

# Basic structured logging for API
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

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
@app.post("/run-ocr", response_model=List[PODResponse])
async def analyze_images_concurrently(ocr_requests: List[OCRRequest]):
    """
    Accepts a list of OCR requests and processes them concurrently with memory-aware batching.
    """
    import torch

    logger.info(f"\n\n🚀 BATCH OCR PROCESSING START - {len(ocr_requests)} requests\n")

    # Memory-aware processing: check GPU memory before starting
    if torch.cuda.is_available():
        try:
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_allocated = torch.cuda.memory_allocated()
            memory_free = memory_total - memory_allocated
            memory_free_gb = memory_free / (1024**3)
            memory_utilization = (memory_allocated / memory_total) * 100

            logger.info(
                f"GPU Memory Status:\n"
                f"  Total: {memory_total / (1024**3):.2f}GB\n"
                f"  Allocated: {memory_allocated / (1024**3):.2f}GB\n"
                f"  Free: {memory_free_gb:.2f}GB\n"
                f"  Utilization: {memory_utilization:.1f}%"
            )

            # Adjust processing strategy based on available memory
            if memory_free_gb < 3:  # Less than 3GB free
                logger.warning(
                    f"Low GPU memory ({memory_free_gb:.2f}GB free). "
                    "Processing sequentially with memory cleanup."
                )

                # Process sequentially with aggressive memory cleanup
                final_responses = []
                failures = []

                for idx, req in enumerate(ocr_requests):
                    logger.info(f"Processing request {idx + 1}/{len(ocr_requests)}: {req.id}")

                    try:
                        result = await process_single_ocr_request(req)
                        final_responses.extend(result)
                    except Exception as e:
                        logger.error(f"Request {req.id} failed: {e}")
                        failures.append({"error": str(e)})

                    # Clear cache after each request
                    torch.cuda.empty_cache()
                    logger.info(f"Memory after cleanup: {torch.cuda.memory_allocated() / (1024**3):.2f}GB")

                if not final_responses and ocr_requests:
                    ids = [getattr(req, 'id', None) for req in ocr_requests]
                    logger.error(f"All batch items failed. ids={ids}; failures={failures}")
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "message": "All OCR requests in the batch failed to process.",
                            "ids": ids,
                            "failures": failures,
                        }
                    )

                return final_responses

        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}. Proceeding with normal processing.")

    # Normal concurrent processing (sufficient memory available)
    logger.info("Sufficient memory available. Processing concurrently.")
    tasks = [process_single_ocr_request(req) for req in ocr_requests]
    results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten the list of lists into a single list of responses
    final_responses = []
    failures = []
    for result in results_from_gather:
        if isinstance(result, Exception):
            failures.append({"error": str(result)})
            continue
        final_responses.extend(result)

    # If all tasks failed, it might be a server-side issue
    if not final_responses and ocr_requests:
        ids = [getattr(req, 'id', None) for req in ocr_requests]
        logger.error(f"All batch items failed. ids={ids}; failures={failures}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "All OCR requests in the batch failed to process.",
                "ids": ids,
                "failures": failures,
            }
        )

    logger.info(f"\n✅ BATCH OCR COMPLETE - Processed {len(final_responses)} successful responses\n")
    return final_responses

# ====================================================================
#  END: MODIFIED SECTION
# ====================================================================


# ========== Run Server ==========
if __name__ == "__main__":
    import uvicorn
    # To leverage multiple CPU cores for non-async blocking tasks (like some parts of image processing),
    # you can use multiple workers. However, for a purely async application, a single worker is often sufficient.
    # The `uvicorn.run` with `workers=1` is generally recommended for async FastAPI apps.
    uvicorn.run("app:app", host="0.0.0.0", port=8080, workers=1)