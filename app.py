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


import models
from models import OCRRequest, PODResponse
from main import run_ocr

load_dotenv()

app = FastAPI(default_response_class=ORJSONResponse)

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
        result_items = await run_ocr_async(ocr_request.file_url_or_path)

        # 2. Map the results to the PODResponse model, adding the specific _id
        responses = []
        for item in result_items:
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
                "Status": item.get("Status"),
                "_id": ocr_request.id  # Attach the ID from the original request
            }
            responses.append(PODResponse.model_validate(response_dict))
        
        return responses

    except Exception as e:
        # Log the error with context and re-raise it so asyncio.gather can catch it.
        # This prevents one failed file from crashing the entire batch.
        print(f"Error processing request with id='{ocr_request.id}': {e}")
        # Optionally, you can import traceback and print print_exc() for more detail
        import traceback
        traceback.print_exc()
        # Re-raising the exception is important for the gathering logic
        raise e


@traceable(run_type="chain", name="api_batch_ocr")
@app.post("/run-ocr", response_model=List[PODResponse])
async def analyze_images_concurrently(ocr_requests: List[OCRRequest]):
    """
    Accepts a list of OCR requests and processes them concurrently.
    """
    # Create a concurrent task for each request in the input list
    tasks = [process_single_ocr_request(req) for req in ocr_requests]

    # Run all tasks concurrently and gather their results.
    # return_exceptions=True ensures that if one task fails, the others continue.
    # The result for the failed task will be the exception object itself.
    results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten the list of lists into a single list of responses,
    # while filtering out any exceptions that occurred.
    final_responses = []
    for result in results_from_gather:
        if isinstance(result, Exception):
            # The error is already logged in the helper function.
            # We just skip adding it to the successful responses.
            continue
        # `result` is a List[PODResponse], so we use extend.
        final_responses.extend(result)

    # If all tasks failed, it might be a server-side issue.
    if not final_responses and ocr_requests:
         raise HTTPException(
             status_code=500,
             detail="All OCR requests in the batch failed to process. Check server logs for details."
         )

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