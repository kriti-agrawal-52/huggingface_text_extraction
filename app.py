from fastapi import FastAPI, UploadFile, File, HTTPException
import logging
import os
import asyncio
from config import MODEL_NAME, QUANTIZE_SAVE_PATH
from extract_text import call_model
from quantize import create_and_save_quantized_model

app = FastAPI()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_debug.log"),
        logging.StreamHandler()
    ]
)

# Asynchronous helper to ensure quantized model exists
async def ensure_quantized_model_exists():
    if not os.path.exists(QUANTIZE_SAVE_PATH):
        logging.info("Quantized model not found. Creating the quantized model.")
        save_path = await create_and_save_quantized_model(MODEL_NAME, QUANTIZE_SAVE_PATH)
        if not save_path or not os.path.exists(save_path):
            logging.error("Failed to create and save the quantized model.")
            raise HTTPException(status_code=500, detail="Failed to create the quantized model.")

# Asynchronous helper for text extraction
async def extract_text(file: UploadFile, model_path: str):
    if not file.filename.endswith((".png", ".jpg")):
        logging.error("Non-image file provided")
        raise HTTPException(status_code=400, detail="Only .png and .jpg files are supported")
    
    return await call_model(file, model_path)

@app.post('/extract_text_large_model')
async def large_model_extraction(file: UploadFile = File(...)):
    return {"extracted_text": await extract_text(file, MODEL_NAME)}

@app.post('/extract_text_quantized_model')
async def small_model_extraction(file: UploadFile = File(...)):
    await ensure_quantized_model_exists()
    return {"extracted_text": await extract_text(file, QUANTIZE_SAVE_PATH)}
