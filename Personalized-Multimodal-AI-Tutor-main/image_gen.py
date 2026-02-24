import torch
from diffusers import StableDiffusionPipeline

def get_image_generator():
    """
    Initializes the Stable Diffusion pipeline.
    """
    # Use CPU friendly settings
    # For CPU, float32 is generally safer than float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Image Generation model on {device}...")
    
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # We load in float32 for CPU compatibility.
    # If on GPU, we could use torch.float16 for speed.
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype
    )
    
    pipe = pipe.to(device)
    
    # Enable attention slicing to save memory on limited hardware
    pipe.enable_attention_slicing()
    
    return pipe

def generate_diagram(pipe, prompt):
    """
    Generates an image from a prompt.
    """
    if not pipe:
        return None
    
    # Enhance prompt for educational diagrams
    full_prompt = f"educational diagram, clear, textbook style, {prompt}, white background, high quality"
    
    # Reduce steps for CPU speed (default is 50, 20-25 might be acceptable for draft)
    image = pipe(full_prompt, num_inference_steps=25).images[0]
    return image
