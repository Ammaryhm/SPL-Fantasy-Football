import os
import logging
import torch
import streamlit as st # Used for @st.cache_resource, st.error, st.warning

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler

logger = logging.getLogger(__name__)

 

MODEL_ID ="CompVis/stable-diffusion-v1-4"
#MODEL_ID ="runwayml/stable-diffusion-v1-5"

 

@st.cache_resource
def setup_image_pipeline():
    logger.info("Attempting to load image generation model (via st.cache_resource)...")

    try:
        # Determine device (MPS for Apple Silicon, else CPU)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Always use float16 and low_cpu_mem_usage to reduce RAM footprint
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

        pipe.to(device)
        logger.info("Stable Diffusion pipeline loaded successfully and cached.")
        return pipe

    except Exception as e:
        logger.error(f"Failed to load image generation pipeline: {e}")
        import traceback
        st.error("An error occurred during model loading. Please check the terminal for details.")
        traceback.print_exc()
        return None


 
