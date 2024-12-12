import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
import transformers
from transformers import AutoModel
from config import MODEL_NAME, QUANTIZE_SAVE_PATH 
from pathlib import Path
import time

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    device_map="auto"  # Automatically distribute across available devices
)

def quantize_model(model):
    """
    Dynamically quantizes a Hugging Face model to torch.qint8.

    Parameters:
        model (torch.nn.Module): The pre-trained model to be quantized.

    Returns:
        torch.nn.Module: The quantized model.
    """
    try:
        # Dynamically quantize the model
        quantized_model = quantize_dynamic(
            model,  # The original model
            {torch.nn.Linear},  # Layers to quantize
            dtype=torch.qint8  # Fixed to qint8 for dynamic quantization
        )
        return quantized_model
    except Exception as e:
        raise RuntimeError(f"Quantization failed: {str(e)}")

def save_model(model):
    """
    Saves a quantized model's state dictionary to a specified path.
    Retries saving up to three times if it fails.

    Parameters:
        model (torch.nn.Module): The original model to quantize and save.

    Returns:
        str or None: The path to the saved model, or None if saving fails after retries.
    """
    save_path = Path(QUANTIZE_SAVE_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        quantized_model = quantize_model(model)  # Quantize the model
    except RuntimeError as e:
        print(f"Error during quantization: {e}")
        return None

    for attempt in range(3):  # Retry up to 3 times
        try:
            torch.save(quantized_model.state_dict(), save_path)
            if save_path.exists() and save_path.stat().st_size > 0:
                return str(save_path)  # Return the path if save is successful
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(2)  # Wait before retrying

    return None  # Return None if all attempts fail

if __name__ == "__main__":
    # Attempt to save the quantized model
    saved_path = save_model(model)
    print(saved_path)
