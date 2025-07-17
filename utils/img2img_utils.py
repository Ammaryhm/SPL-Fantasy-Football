import os
import streamlit as st
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # Or whatever model you're using

def get_huggingface_token():
    """
    Get Hugging Face token from Streamlit secrets or environment variables
    """
    try:
        return st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except:
        return os.getenv("HUGGINGFACEHUB_API_TOKEN")

@st.cache_resource
def setup_image_pipeline():
    """
    Set up the Stable Diffusion img2img pipeline with proper error handling
    """
    try:
        logger.info("Setting up Stable Diffusion img2img pipeline...")
        
        # Get the Hugging Face token
        hf_token = get_huggingface_token()
        
        # Load the pipeline with explicit PyTorch format
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=False,  # Force PyTorch format
            token=hf_token,  # Updated parameter name
            safety_checker=None,  # Optional: disable safety checker for faster loading
            requires_safety_checker=False,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Move to appropriate device
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            # For CPU, use float32 for better stability
            pipe = pipe.to("cpu")
            pipe.torch_dtype = torch.float32
        
        # Enable memory efficient attention if available
        try:
            pipe.enable_attention_slicing()
        except:
            pass
            
        try:
            pipe.enable_model_cpu_offload()
        except:
            pass
        
        logger.info("Stable Diffusion img2img pipeline setup complete!")
        return pipe
        
    except Exception as e:
        logger.error(f"Error setting up img2img pipeline: {str(e)}")
        st.error(f"Failed to setup image generation: {str(e)}")
        return None

def generate_image_from_image(pipe, prompt, init_image, strength=0.75, guidance_scale=7.5, num_inference_steps=50):
    """
    Generate an image from an input image using the pipeline
    """
    try:
        if pipe is None:
            return None
            
        # Ensure the input image is in the correct format
        if isinstance(init_image, str):
            init_image = Image.open(init_image).convert("RGB")
        elif not isinstance(init_image, Image.Image):
            init_image = Image.fromarray(init_image).convert("RGB")
        
        # Resize image to a reasonable size (optional, for performance)
        init_image = init_image.resize((512, 512))
        
        # Generate the image
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
        
        return result.images[0]
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        st.error(f"Failed to generate image: {str(e)}")
        return None

def process_uploaded_image(uploaded_file):
    """
    Process an uploaded image file
    """
    try:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            return image
        return None
    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}")
        st.error(f"Failed to process uploaded image: {str(e)}")
        return None
