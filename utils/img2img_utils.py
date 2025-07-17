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
        # Determine device (CUDA if available, else CPU or MPS)
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")

        # Load with low‑memory settings and half precision
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            revision="fp16",               # load the fp16 weights if available
            torch_dtype=torch.float16,     # half‑precision everywhere
            use_safetensors=True,
            low_cpu_mem_usage=True,        # shards weights to reduce peak RAM
            use_auth_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

        # Reduce runtime memory further
        pipe.enable_attention_slicing()   # slice attention computation to save memory

        # If you have accelerate installed, you can offload parts to CPU
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            # if accelerate isn't available, just skip
            pass

        # Move model to the target device
        pipe.to(device)

        logger.info("Stable Diffusion pipeline loaded successfully and cached.")
        return pipe

    except Exception as e:
        logger.error(f"Failed to load image generation pipeline: {e}")
        import traceback
        st.error("An error occurred during model loading. Please check the terminal for details.")
        traceback.print_exc()
        return None
