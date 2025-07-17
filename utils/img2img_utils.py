import os
import streamlit as st
import torch
import logging
from PIL import Image
import gc

# Try to import diffusers with error handling
try:
    from diffusers import StableDiffusionImg2ImgPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import diffusers: {e}")
    st.error("Please check your requirements.txt for compatible versions of diffusers and huggingface_hub")
    DIFFUSERS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Model configuration - using a lightweight model
MODEL_ID = "runwayml/stable-diffusion-v1-5"

def get_huggingface_token():
    """
    Get Hugging Face token from Streamlit secrets or environment variables
    """
    try:
        return st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except:
        return os.getenv("HUGGINGFACEHUB_API_TOKEN")

def clear_gpu_memory():
    """
    Clear GPU memory to prevent OOM errors
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception as e:
        logger.warning(f"Error clearing GPU memory: {e}")

def get_device_config():
    """
    Get optimal device configuration for current environment
    """
    if torch.cuda.is_available():
        # Check available GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory / (1024**3)
        
        logger.info(f"GPU memory available: {gpu_memory_gb:.2f} GB")
        
        # Use different configurations based on available memory
        if gpu_memory_gb < 4:
            return {
                "device": "cpu",
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
                "enable_attention_slicing": True,
                "enable_sequential_cpu_offload": True
            }
        else:
            return {
                "device": "cuda",
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "enable_attention_slicing": True,
                "enable_model_cpu_offload": True
            }
    else:
        return {
            "device": "cpu",
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
            "enable_attention_slicing": True,
            "enable_sequential_cpu_offload": True
        }

@st.cache_resource
def setup_image_pipeline():
    """
    Set up the Stable Diffusion img2img pipeline with maximum memory efficiency
    """
    if not DIFFUSERS_AVAILABLE:
        st.error("Diffusers library not available. Please check your requirements.txt")
        return None
        
    try:
        logger.info("Setting up memory-efficient Stable Diffusion img2img pipeline...")
        
        # Clear any existing GPU memory
        clear_gpu_memory()
        
        # Get optimal device configuration
        device_config = get_device_config()
        
        # Get the Hugging Face token
        hf_token = get_huggingface_token()
        
        # Load the pipeline with maximum memory efficiency
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=device_config["torch_dtype"],
            use_safetensors=False,  # Force PyTorch format for compatibility
            token=hf_token,
            safety_checker=None,  # Disable safety checker to save memory
            requires_safety_checker=False,
            low_cpu_mem_usage=device_config["low_cpu_mem_usage"],
            device_map=None,  # Manual device management
            variant="fp16" if device_config["torch_dtype"] == torch.float16 else None,
            local_files_only=False,
            cache_dir=None,  # Use default cache to avoid issues
        )
        
        # Move to appropriate device
        pipe = pipe.to(device_config["device"])
        
        # Enable memory-saving features
        if device_config["enable_attention_slicing"]:
            try:
                pipe.enable_attention_slicing("auto")
                logger.info("Attention slicing enabled")
            except Exception as e:
                logger.warning(f"Could not enable attention slicing: {e}")
        
        if device_config["enable_model_cpu_offload"]:
            try:
                pipe.enable_model_cpu_offload()
                logger.info("Model CPU offload enabled")
            except Exception as e:
                logger.warning(f"Could not enable model CPU offload: {e}")
        
        if device_config["enable_sequential_cpu_offload"]:
            try:
                pipe.enable_sequential_cpu_offload()
                logger.info("Sequential CPU offload enabled")
            except Exception as e:
                logger.warning(f"Could not enable sequential CPU offload: {e}")
        
        # Enable memory efficient attention if available
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("XFormers memory efficient attention enabled")
        except Exception as e:
            logger.warning(f"XFormers not available: {e}")
        
        # Final memory cleanup
        clear_gpu_memory()
        
        logger.info("Memory-efficient Stable Diffusion img2img pipeline setup complete!")
        return pipe
        
    except Exception as e:
        logger.error(f"Error setting up img2img pipeline: {str(e)}")
        st.error(f"Failed to setup image generation: {str(e)}")
        clear_gpu_memory()
        return None

def optimize_image_size(image, max_size=512):
    """
    Optimize image size to reduce memory usage
    """
    try:
        # Get current dimensions
        width, height = image.size
        
        # Calculate optimal dimensions (must be divisible by 8 for Stable Diffusion)
        if width > height:
            new_width = min(width, max_size)
            new_height = int((height * new_width) / width)
        else:
            new_height = min(height, max_size)
            new_width = int((width * new_height) / height)
        
        # Ensure dimensions are divisible by 8
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        # Minimum size constraints
        new_width = max(new_width, 256)
        new_height = max(new_height, 256)
        
        if new_width != width or new_height != height:
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return image
    except Exception as e:
        logger.error(f"Error optimizing image size: {e}")
        return image

def generate_image_from_image(pipe, prompt, init_image, strength=0.75, guidance_scale=7.5, num_inference_steps=20):
    """
    Generate an image from an input image using the pipeline with memory optimization
    """
    try:
        if pipe is None:
            st.error("Pipeline not initialized")
            return None
        
        # Clear memory before generation
        clear_gpu_memory()
        
        # Ensure the input image is in the correct format
        if isinstance(init_image, str):
            init_image = Image.open(init_image).convert("RGB")
        elif not isinstance(init_image, Image.Image):
            init_image = Image.fromarray(init_image).convert("RGB")
        
        # Optimize image size to reduce memory usage
        init_image = optimize_image_size(init_image, max_size=512)
        
        # Use lower inference steps for memory efficiency
        num_inference_steps = min(num_inference_steps, 30)
        
        logger.info(f"Generating image with {num_inference_steps} steps")
        
        # Generate the image with memory management
        with torch.inference_mode():
            try:
                result = pipe(
                    prompt=prompt,
                    image=init_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    output_type="pil",
                    return_dict=True
                )
                
                generated_image = result.images[0]
                
                # Clear memory after generation
                clear_gpu_memory()
                
                return generated_image
                
            except torch.cuda.OutOfMemoryError:
                logger.error("GPU out of memory, trying with CPU offload")
                clear_gpu_memory()
                
                # Try with sequential CPU offload
                try:
                    pipe.enable_sequential_cpu_offload()
                    result = pipe(
                        prompt=prompt,
                        image=init_image,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=max(10, num_inference_steps // 2),  # Reduce steps further
                        output_type="pil",
                        return_dict=True
                    )
                    generated_image = result.images[0]
                    clear_gpu_memory()
                    return generated_image
                except Exception as e:
                    logger.error(f"Failed with CPU offload: {e}")
                    st.error("Generation failed due to memory constraints. Try with a smaller image or simpler prompt.")
                    return None
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        st.error(f"Failed to generate image: {str(e)}")
        clear_gpu_memory()
        return None

def process_uploaded_image(uploaded_file, max_size=512):
    """
    Process an uploaded image file with memory optimization
    """
    try:
        if uploaded_file is not None:
            # Load and convert image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Optimize size immediately
            image = optimize_image_size(image, max_size=max_size)
            
            logger.info(f"Processed uploaded image to size: {image.size}")
            return image
        return None
    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}")
        st.error(f"Failed to process uploaded image: {str(e)}")
        return None

def cleanup_pipeline():
    """
    Clean up pipeline resources
    """
    try:
        clear_gpu_memory()
        if 'pipe' in st.session_state:
            del st.session_state['pipe']
        logger.info("Pipeline cleanup completed")
    except Exception as e:
        logger.warning(f"Error during pipeline cleanup: {e}")

# Add cleanup on app exit
import atexit
atexit.register(cleanup_pipeline)
