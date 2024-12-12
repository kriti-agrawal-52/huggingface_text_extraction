from transformers import AutoModel, AutoTokenizer
import tracemalloc
import time
import logging
from typing import Optional
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_metrics.log"),
        logging.StreamHandler()
    ]
)

# Logger for debugging (upper layer log file already exists)
debug_logger = logging.getLogger("api_debug")

async def initialize_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Asynchronously initializes and returns the tokenizer for the given model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        AutoTokenizer: The initialized tokenizer.
    """
    try:
        tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained,
            model_name,
            trust_remote_code=True
        )
        tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        return tokenizer
    except Exception as e:
        debug_logger.error(f"Error initializing tokenizer for model {model_name}: {e}")
        raise

async def initialize_model(model_name: str) -> AutoModel:
    """
    Asynchronously initializes and returns the model for the given model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        AutoModel: The initialized model.
    """
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
        debug_logger.error(f"Error initializing model {model_name}: {e}")
        raise

async def call_model(image_file: str, model_name: str) -> Optional[str]:
    """
    Asynchronously benchmarks the model's chat function for execution time and memory usage.

    Args:
        image_file (str): The input image file.
        model_name (str): The name of the model.

    Returns:
        Optional[str]: The result of the model's chat function or None in case of errors.
    """
    # Initialize tokenizer and model
    try:
        tokenizer = await initialize_tokenizer(model_name)
        model = await initialize_model(model_name)
    except Exception as e:
        debug_logger.error(f"Error during initialization of tokenizer or model for {model_name}: {e}")
        return None  # Handle the error and return None

    # Track memory and execution time
    tracemalloc.start()
    start_time = time.time()

    attempt = 0
    max_retries = 3
    res: Optional[str] = None

    while attempt < max_retries:
        try:
            # Attempt to execute the model's chat function
            res = await asyncio.to_thread(
                model.chat,
                tokenizer,
                image_file,
                ocr_type='format',
                render=True
            )
            break  # If successful, exit the retry loop
        except Exception as e:
            attempt += 1
            debug_logger.error(
                f"Attempt {attempt} failed for model.chat with error: {e}"
            )
            if attempt == max_retries:
                debug_logger.error(
                    f"Maximum retry limit reached for model.chat with {model_name} on file {image_file}."
                )
                return None  # Handle the error and return None

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Log metrics
    logging.info({
        "execution_time": end_time - start_time,
        "current_memory_mb": current / 1024 / 1024,
        "peak_memory_mb": peak / 1024 / 1024
    })

    # Return the result or None if errors occurred
    return res