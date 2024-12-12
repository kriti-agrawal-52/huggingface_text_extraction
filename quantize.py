import torch
from torch.quantization import quantize_dynamic
from transformers import AutoModel
from config import MODEL_NAME, QUANTIZE_SAVE_PATH
from pathlib import Path
import logging
import asyncio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantize_debug.log"),
        logging.StreamHandler()
    ]
)

async def load_pretrained_model(model_name: str):
    """
    Asynchronously loads a pre-trained model.
    """
    logging.info(f"Loading pre-trained model: {model_name}")
    try:
        return await asyncio.to_thread(
            AutoModel.from_pretrained,
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map="auto"
        )
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

async def quantize_model(model):
    """
    Asynchronously quantizes a Hugging Face model to torch.qint8.
    """
    logging.info("Starting quantization...")
    try:
        return await asyncio.to_thread(
            quantize_dynamic,
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    except Exception as e:
        logging.error(f"Quantization failed: {e}")
        raise RuntimeError(f"Quantization failed: {str(e)}")

async def save_quantized_model(quantized_model, save_path: str):
    """
    Asynchronously saves a quantized model's state dictionary.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        await asyncio.to_thread(torch.save, quantized_model.state_dict(), save_path)
        if save_path.exists() and save_path.stat().st_size > 0:
            logging.info(f"Quantized model saved successfully at {save_path}")
    except Exception as e:
        logging.error(f"Failed to save the quantized model: {e}")
        raise RuntimeError(f"Failed to save the quantized model: {str(e)}")

async def create_and_save_quantized_model(model_name: str, save_path: str):
    """
    Asynchronously orchestrates the process of loading, quantizing, and saving a model.
    """
    logging.info("Starting the quantization process...")
    try:
        model = await load_pretrained_model(model_name)
        
        for attempt in range(3):
            try:
                logging.info(f"Attempt {attempt + 1} to quantize the model...")
                quantized_model = await quantize_model(model)
                break
            except Exception as e:
                logging.error(f"Quantization attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise RuntimeError("Quantization failed after 3 attempts.")
                await asyncio.sleep(2)
        
        await save_quantized_model(quantized_model, save_path)
        logging.info(f"Quantized model process completed successfully. Saved at {save_path}")
        return save_path
    except Exception as e:
        logging.error(f"Quantization process failed: {e}")
        return None