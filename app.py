from fastapi import FastAPI, UploadFile, File, HTTPException
import logging
from config import MODEL_NAME, QUANTIZE_SAVE_PATH
from extract_text import call_model

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

@app.post('/extract_text_large_model')
async def large_model_extraction(
    file: UploadFile = File(...)
):
    # Validate file extension
    if not file.filename.endswith((".png", ".jpg")):
        logging.error("Non-image file provided")
        raise HTTPException(status_code=400, detail="Only .png and .jpg files are supported")
    extracted_text = await call_model(file, MODEL_NAME)
    if extracted_text:
        return {"extracted_text": extracted_text}
    else:
        return {"detail": "Text extraction failed"}