from flask import Flask, Response, request
from flask_cors import CORS
import time
import threading
import queue
import asyncio
import struct
import torch
import gc
import logging
import os
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
from tokens_decoder import tokens_decoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- Global state ---
request_queue = queue.Queue()
model_lock = threading.RLock()
model_health_status = {
    "healthy": True, 
    "last_error": None, 
    "last_restart": None,
    "cuda_errors": 0,
    "restart_count": 0
}

# --- GPU Memory Management ---
def reset_gpu_memory():
    """Aggressively reset GPU memory"""
    try:
        # Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
        return True
    except Exception as e:
        logger.error(f"Failed to reset GPU memory: {e}")
        return False

# --- Create WAV Header ---
def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """Create a WAV header for streaming audio."""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
                         b'RIFF', 0,
                         b'WAVE',
                         b'fmt ', 16,
                         1,
                         channels,
                         sample_rate,
                         byte_rate,
                         block_align,
                         bits_per_sample,
                         b'data', 0)
    return header

# --- Model Manager Class ---
class LLMModelManager:
    def __init__(self, model_name="canopylabs/orpheus-tts-0.1-emo-instruct"):
        self.model_name = model_name
        self.model = None
        self.tokeniser = None
        self.loop = None
        self.engine_thread = None
        self.is_running = False
        self.start_token = torch.tensor([[128259]], dtype=torch.int64)
        self.end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
        self.sampling_params = SamplingParams(
            temperature=0.9, 
            top_p=0.6, 
            max_tokens=2000, 
            repetition_penalty=1.1, 
            stop_token_ids=[128258]
        )
        
        # Set environment variable to enable CUDA device-side assert tracking
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        self.initialize_model()

    def initialize_model(self):
        """Initialize the model and tokenizer."""
        with model_lock:
            try:
                # Make sure GPU memory is clear before initializing
                reset_gpu_memory()
                
                logger.info(f"Initializing model: {self.model_name}")
                model_health_status["last_restart"] = time.time()
                
                # Create a new event loop for the model
                if self.loop is None or self.loop.is_closed():
                    self.loop = asyncio.new_event_loop()
                
                # Initialize model
                engine_args = AsyncEngineArgs(
                    model=self.model_name, 
                    dtype=torch.float16,
                    gpu_memory_utilization=0.8
                )
                
                self.model = AsyncLLMEngine.from_engine_args(engine_args)
                self.tokeniser = AutoTokenizer.from_pretrained(self.model_name)
                
                # Start processing thread
                self.is_running = True
                self.engine_thread = threading.Thread(target=self._run_engine_loop, daemon=True)
                self.engine_thread.start()
                
                # Give the thread a moment to start
                time.sleep(1.0)
                
                model_health_status["healthy"] = True
                model_health_status["last_error"] = None
                model_health_status["restart_count"] += 1
                logger.info("Model initialization complete")
                return True
            except Exception as e:
                model_health_status["healthy"] = False
                model_health_status["last_error"] = str(e)
                logger.error(f"Model initialization failed: {e}")
                return False

    def shutdown(self):
        """Shutdown the model and event loop."""
        with model_lock:
            logger.info("Shutting down model...")
            
            # Signal thread to stop
            self.is_running = False
            
            # Stop the loop if it's running
            if self.loop and not self.loop.is_closed():
                try:
                    if self.loop.is_running():
                        self.loop.call_soon_threadsafe(self.loop.stop)
                except Exception as e:
                    logger.error(f"Error shutting down event loop: {e}")
            
            # Wait for thread to finish
            if self.engine_thread and self.engine_thread.is_alive():
                try:
                    self.engine_thread.join(timeout=5)
                except Exception as e:
                    logger.error(f"Error joining engine thread: {e}")
            
            # Explicitly clear model references
            self.model = None
            
            # Clear GPU memory
            reset_gpu_memory()
            
            logger.info("Model shutdown complete")

    def restart_model(self):
        """Restart the model with exponential backoff for CUDA errors."""
        logger.warning("Attempting to restart the model...")
        
        # Calculate backoff time based on CUDA error count
        backoff_time = min(2 ** model_health_status["cuda_errors"], 30)  # Max 30 seconds
        
        if "CUDA error" in str(model_health_status["last_error"]) or "cuDNN error" in str(model_health_status["last_error"]):
            model_health_status["cuda_errors"] += 1
            logger.warning(f"CUDA error detected. Backoff time: {backoff_time}s. Error count: {model_health_status['cuda_errors']}")
            time.sleep(backoff_time)
        
        # Shutdown existing model
        self.shutdown()
        
        # Initialize new model
        return self.initialize_model()

    def process_prompt(self, prompt):
        """Process a prompt by adding tokens and encoding."""
        prompt = prompt + " " + "<zac>"
        input_ids = self.tokeniser(prompt, return_tensors="pt").input_ids
        modified_input_ids = torch.cat([self.start_token, input_ids, self.end_tokens], dim=1)
        iids_string = self.tokeniser.decode(modified_input_ids[0].tolist())
        initial_tokens = len(self.tokeniser(iids_string, return_tensors="pt").input_ids[0])
        return iids_string, initial_tokens

    def _run_engine_loop(self):
        """Run the engine loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        
        async def process_queue():
            while self.is_running:
                try:
                    # Get the next request if any
                    if not request_queue.empty():
                        request_data = request_queue.get()
                        prompt, token_queue, request_id, attempt = request_data
                        
                        # Process this request
                        await self._process_request(prompt, token_queue, request_id, attempt)
                        request_queue.task_done()
                    else:
                        # No requests, sleep briefly
                        await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    # Handle cancellation cleanly
                    logger.info("Engine loop was cancelled")
                    break
                except Exception as e:
                    logger.error(f"Engine loop error: {e}")
                    model_health_status["healthy"] = False
                    model_health_status["last_error"] = str(e)
                    break
        
        try:
            self.loop.run_until_complete(process_queue())
        except Exception as e:
            logger.error(f"Engine thread crashed with error: {e}")
            model_health_status["healthy"] = False
            model_health_status["last_error"] = str(e)
        finally:
            if not self.loop.is_closed():
                self.loop.close()
            logger.info("Engine loop closed")

    async def _process_request(self, prompt, token_queue, request_id, attempt):
        """Process a single request and put tokens in the queue."""
        try:
            # Preprocess prompt
            prompt_string, initial_tokens = self.process_prompt(prompt)
            
            # Generate tokens with better error handling
            try:
                results_generator = self.model.generate(
                    prompt_string, 
                    self.sampling_params, 
                    request_id=request_id
                )
                
                previous_text = ""
                async for request_output in results_generator:
                    if not self.is_running:
                        # Stop if manager is shutting down
                        break
                        
                    text = request_output.outputs[0].text
                    new_text = text[len(previous_text):]
                    previous_text = text
                    
                    if new_text:
                        token_queue.put(new_text)
                
                # Request completed successfully
                token_queue.put(None)  # Signal completion
                
            except Exception as e:
                error_str = str(e)
                logger.error(f"Generation error: {error_str}")
                
                # Check for specific CUDA errors
                is_cuda_error = any(err in error_str for err in [
                    "CUDA error", "cuDNN error", "device-side assert", 
                    "CUBLAS", "CUDNN_STATUS", "out of memory"
                ])
                
                if is_cuda_error:
                    model_health_status["healthy"] = False
                    model_health_status["last_error"] = error_str
                    token_queue.put("CUDA_ERROR")
                elif "Background loop has errored" in error_str:
                    model_health_status["healthy"] = False
                    model_health_status["last_error"] = error_str
                    token_queue.put("RETRY")
                else:
                    # More common error, just retry
                    token_queue.put("RETRY" if attempt < 3 else "ERROR")
                    
        except Exception as e:
            logger.error(f"Error processing request {request_id} (attempt {attempt}): {e}")
            token_queue.put("ERROR")

# --- Watchdog thread ---
def watchdog_thread():
    """Monitor model health and restart if necessary."""
    restart_attempts = 0
    max_restarts = 10
    restart_window = 3600  # 1 hour window to count restarts
    last_restart_time = time.time()
    
    while True:
        try:
            # Check model health
            if not model_health_status["healthy"]:
                current_time = time.time()
                
                # Reset restart counter if we're outside the window
                if current_time - last_restart_time > restart_window:
                    restart_attempts = 0
                    last_restart_time = current_time
                
                # Check if we're not restarting too frequently
                if restart_attempts < max_restarts:
                    logger.warning(f"Watchdog detected unhealthy model. Attempting restart ({restart_attempts+1}/{max_restarts})...")
                    if model_manager.restart_model():
                        logger.info("Model restarted successfully")
                    else:
                        logger.error("Failed to restart model")
                    
                    restart_attempts += 1
                else:
                    logger.critical(f"Too many restart attempts ({restart_attempts}) within time window. Waiting 5 minutes...")
                    time.sleep(300)  # Wait longer before trying again
                    restart_attempts = 0  # Reset counter after the wait
            
            # Check every few seconds
            time.sleep(5)
        except Exception as e:
            logger.error(f"Watchdog error: {e}")
            time.sleep(10)  # Back off on errors

# --- Initialize model manager ---
model_manager = LLMModelManager()

# --- Start watchdog ---
threading.Thread(target=watchdog_thread, daemon=True).start()

# --- Flask endpoint for audio streaming ---
@app.route('/events', methods=['GET'])
def sse():
    prompt = request.args.get('prompt', 'No prompt provided')
    
    # Queue for this specific request
    token_queue = queue.Queue()
    
    # Generate a unique request ID
    request_id = f"{time.time()}-{hash(prompt) % 10000}"
    
    # Add request to queue with attempt number
    request_queue.put((prompt, token_queue, request_id, 1))
    
    def sse_event_stream(prompt):
        # First, yield the WAV header
        wav_header = create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1)
        yield wav_header
        
        # Collect tokens from the queue
        def raw_tokens():
            while True:
                token = token_queue.get()
                
                if token is None:
                    # Normal completion
                    break
                elif token in ["RETRY", "CUDA_ERROR", "ERROR"]:
                    # Error signals - need to restart model
                    logger.info(f"Error signal received: {token}")
                    break
                else:
                    # Normal token
                    yield token
        
        # Apply dummy processor to transform tokens into audio bytes
        for processed_token in tokens_decoder(raw_tokens()):
            logger.debug("Sending token")
            yield processed_token
        
        # If we got an error, wait for model to restart before retrying
        token = token_queue.get_nowait() if not token_queue.empty() else None
        if token in ["RETRY", "CUDA_ERROR"]:
            # Wait for model restart
            time.sleep(3)
            
            # Create a new queue for retry
            new_token_queue = queue.Queue()
            
            # Requeue the request
            request_queue.put((prompt, new_token_queue, f"{request_id}-retry", 2))
            
            # Process the retry in the same way
            def retry_tokens():
                while True:
                    retry_token = new_token_queue.get()
                    if retry_token is None or retry_token in ["RETRY", "CUDA_ERROR", "ERROR"]:
                        break
                    yield retry_token
            
            # Process retry tokens
            for processed_token in dummy_processor(retry_tokens()):
                logger.debug("Sending retry token")
                yield processed_token
    
    return Response(sse_event_stream(prompt), mimetype='audio/wav')

# --- Health check endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        try:
            gpu_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": f"{torch.cuda.memory_allocated(0)/1024**3:.2f} GB",
                "memory_reserved": f"{torch.cuda.memory_reserved(0)/1024**3:.2f} GB"
            }
        except Exception as e:
            gpu_info = {"error": str(e)}
    
    status = {
        "healthy": model_health_status["healthy"],
        "last_error": model_health_status["last_error"],
        "last_restart": model_health_status["last_restart"],
        "cuda_errors": model_health_status["cuda_errors"],
        "restart_count": model_health_status["restart_count"],
        "pending_requests": request_queue.qsize(),
        "gpu_info": gpu_info
    }
    return status

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8080, threaded=True)