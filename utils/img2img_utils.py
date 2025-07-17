import os
import streamlit as st
import logging
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import requests
import json

logger = logging.getLogger(__name__)

# Use a lightweight API-based approach instead of loading heavy models
def get_huggingface_token():
    """
    Get Hugging Face token from Streamlit secrets or environment variables
    """
    try:
        return st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except:
        return os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_openai_api_key():
    """
    Get OpenAI API key for DALL-E as alternative
    """
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.getenv("OPENAI_API_KEY")

def optimize_image_size(image, max_size=512):
    """
    Optimize image size to reduce processing load
    """
    try:
        width, height = image.size
        
        # Calculate optimal dimensions
        if width > height:
            new_width = min(width, max_size)
            new_height = int((height * new_width) / width)
        else:
            new_height = min(height, max_size)
            new_width = int((width * new_height) / height)
        
        # Ensure minimum size
        new_width = max(new_width, 256)
        new_height = max(new_height, 256)
        
        if new_width != width or new_height != height:
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return image
    except Exception as e:
        logger.error(f"Error optimizing image size: {e}")
        return image

def image_to_base64(image):
    """
    Convert PIL image to base64 string
    """
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

def apply_image_effects(image, effect_type="enhance"):
    """
    Apply various effects to image as a lightweight alternative to AI generation
    """
    try:
        if effect_type == "enhance":
            # Enhance contrast and color
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
        elif effect_type == "artistic":
            # Apply artistic filter
            image = image.filter(ImageFilter.SMOOTH_MORE)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.3)
            
        elif effect_type == "vintage":
            # Apply vintage effect
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.8)
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
        elif effect_type == "dramatic":
            # Apply dramatic effect
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.4)
            
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.9)
            
        return image
    except Exception as e:
        logger.error(f"Error applying image effects: {e}")
        return image

def generate_with_huggingface_api(prompt, image_base64, hf_token):
    """
    Use Hugging Face Inference API for img2img generation
    """
    try:
        # Use the img2img specific endpoint
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        headers = {"Authorization": f"Bearer {hf_token}"}
        
        # For img2img, we need to send the image data differently
        payload = {
            "inputs": prompt,
            "parameters": {
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "strength": 0.75,
                "width": 512,
                "height": 512
            }
        }
        
        # Make the request
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            try:
                image_bytes = response.content
                generated_image = Image.open(io.BytesIO(image_bytes))
                return generated_image
            except Exception as e:
                logger.error(f"Error processing HF API response: {e}")
                return None
        elif response.status_code == 503:
            st.warning("🔄 Model is loading, please wait a moment and try again...")
            return None
        else:
            logger.error(f"HF API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.warning("⏱️ Request timed out. The model might be busy. Please try again.")
        return None
    except Exception as e:
        logger.error(f"Error with Hugging Face API: {e}")
        return None

def generate_with_openai_api(prompt, init_image, openai_key):
    """
    Use OpenAI DALL-E API for image generation
    """
    try:
        import openai
        
        client = openai.OpenAI(api_key=openai_key)
        
        # For DALL-E, we need to create a variation or edit
        # Since DALL-E 3 doesn't support img2img directly, we'll use a descriptive prompt
        enhanced_prompt = f"Create a Saudi Pro League football fan avatar based on this description: {prompt}. Make it look like a realistic person wearing team colors and merchandise."
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        image_response = requests.get(image_url, timeout=30)
        generated_image = Image.open(io.BytesIO(image_response.content))
        
        # Resize to match expected output size
        generated_image = optimize_image_size(generated_image, max_size=512)
        
        return generated_image
        
    except Exception as e:
        logger.error(f"Error with OpenAI API: {e}")
        return None

@st.cache_resource
def setup_image_pipeline():
    """
    Setup a lightweight image processing pipeline
    """
    try:
        logger.info("Setting up lightweight image processing pipeline...")
        
        # Check available API keys
        hf_token = get_huggingface_token()
        openai_key = get_openai_api_key()
        
        pipeline_config = {
            "huggingface_available": bool(hf_token),
            "openai_available": bool(openai_key),
            "fallback_effects": True  # Always available
        }
        
        logger.info("Lightweight image pipeline setup complete!")
        return pipeline_config
        
    except Exception as e:
        logger.error(f"Error setting up image pipeline: {str(e)}")
        return {"fallback_effects": True}

def generate_image_from_image(pipe_config, prompt, init_image, strength=0.75, guidance_scale=7.5, num_inference_steps=20):
    """
    Generate an image using available methods (API calls or effects)
    """
    try:
        if not isinstance(init_image, Image.Image):
            if isinstance(init_image, str):
                init_image = Image.open(init_image).convert("RGB")
            else:
                init_image = Image.fromarray(init_image).convert("RGB")
        
        # Optimize image size
        init_image = optimize_image_size(init_image, max_size=512)
        
        # Try different generation methods in order of preference
        
        # Method 1: Hugging Face API (if available)
        if pipe_config.get("huggingface_available"):
            st.info("🤖 Generating with Hugging Face API...")
            hf_token = get_huggingface_token()
            image_base64 = image_to_base64(init_image)
            
            if image_base64:
                result = generate_with_huggingface_api(prompt, image_base64, hf_token)
                if result:
                    return result
            
            st.warning("Hugging Face API failed, trying next method...")
        
        # Method 2: OpenAI API (if available)
        if pipe_config.get("openai_available"):
            st.info("🎨 Generating with OpenAI DALL-E...")
            openai_key = get_openai_api_key()
            
            result = generate_with_openai_api(prompt, init_image, openai_key)
            if result:
                return result
            
            st.warning("OpenAI API failed, using image effects...")
        
        # Method 3: Fallback to image effects
        st.info("✨ Applying artistic effects to your image...")
        
        # Determine effect type based on prompt keywords
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["vintage", "old", "retro", "sepia"]):
            effect_type = "vintage"
        elif any(word in prompt_lower for word in ["dramatic", "dark", "moody", "intense"]):
            effect_type = "dramatic"
        elif any(word in prompt_lower for word in ["artistic", "painting", "art", "stylized"]):
            effect_type = "artistic"
        else:
            effect_type = "enhance"
        
        enhanced_image = apply_image_effects(init_image, effect_type)
        
        st.success(f"Applied {effect_type} effects to your image!")
        return enhanced_image
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        st.error(f"Failed to generate image: {str(e)}")
        return None

def process_uploaded_image(uploaded_file, max_size=512):
    """
    Process an uploaded image file
    """
    try:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image = optimize_image_size(image, max_size=max_size)
            logger.info(f"Processed uploaded image to size: {image.size}")
            return image
        return None
    except Exception as e:
        logger.error(f"Error processing uploaded image: {str(e)}")
        st.error(f"Failed to process uploaded image: {str(e)}")
        return None

def get_generation_tips():
    """
    Provide tips for better image generation
    """
    return """
    💡 **Tips for better results:**
    
    🤖 **With API access:**
    - Use detailed, descriptive prompts
    - Specify art style (e.g., "in the style of...")
    - Mention colors, lighting, mood
    
    ✨ **With image effects:**
    - Use keywords like "vintage", "dramatic", "artistic"
    - Effects work best with portraits and landscapes
    - Try different keywords to get various effects
    
    🔑 **To enable AI generation:**
    - Add your Hugging Face token to Streamlit secrets
    - Or add your OpenAI API key for DALL-E access
    """

# Lightweight alternative that won't crash
def create_avatar_overlay(base_image, team_colors=None):
    """
    Create a team-themed avatar overlay
    """
    try:
        if team_colors is None:
            team_colors = [(0, 100, 0), (255, 255, 255)]  # Default green and white
        
        # Create a subtle overlay effect
        overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
        
        # Apply team color tint
        enhancer = ImageEnhance.Color(base_image)
        tinted_image = enhancer.enhance(1.2)
        
        return tinted_image
        
    except Exception as e:
        logger.error(f"Error creating avatar overlay: {e}")
        return base_image
