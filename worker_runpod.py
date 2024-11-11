import json
import os
import numpy as np
import torch
from diffusers import (AutoencoderKL, CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from transformers import T5EncoderModel, T5Tokenizer
from omegaconf import OmegaConf
from PIL import Image
from cogvideox.models.transformer3d import CogVideoXTransformer3DModel
from cogvideox.models.autoencoder_magvit import AutoencoderKLCogVideoX
from cogvideox.pipeline.pipeline_cogvideox import CogVideoX_Fun_Pipeline
from cogvideox.pipeline.pipeline_cogvideox_inpaint import CogVideoX_Fun_Pipeline_Inpaint
from cogvideox.utils.lora_utils import merge_lora, unmerge_lora
from cogvideox.utils.utils import get_image_to_video_latent, save_videos_grid, ASPECT_RATIO_512, get_closest_ratio, to_pil
from huggingface_hub import HfApi, HfFolder

# Low GPU memory mode
low_gpu_memory_mode = False

# Model loading section
model_id = "/content/model"
transformer = CogVideoXTransformer3DModel.from_pretrained_2d(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
).to(torch.bfloat16)

vae = AutoencoderKLCogVideoX.from_pretrained(
    model_id, subfolder="vae"
).to(torch.bfloat16)

text_encoder = T5EncoderModel.from_pretrained(
    model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
)

sampler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler,
    "PNDM": PNDMScheduler,
    "DDIM_Cog": CogVideoXDDIMScheduler,
    "DDIM_Origin": DDIMScheduler,
}
scheduler = sampler_dict["DPM++"].from_pretrained(model_id, subfolder="scheduler")

# Pipeline setup
if transformer.config.in_channels != vae.config.latent_channels:
    pipeline = CogVideoX_Fun_Pipeline_Inpaint.from_pretrained(
        model_id, vae=vae, text_encoder=text_encoder,
        transformer=transformer, scheduler=scheduler,
        torch_dtype=torch.bfloat16
    )
else:
    pipeline = CogVideoX_Fun_Pipeline.from_pretrained(
        model_id, vae=vae, text_encoder=text_encoder,
        transformer=transformer, scheduler=scheduler,
        torch_dtype=torch.bfloat16
    )

if low_gpu_memory_mode:
    pipeline.enable_sequential_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()

@torch.inference_mode()
def generate(input):
    values = input["input"]
    prompt = values["prompt"]
    negative_prompt = values.get("negative_prompt", "")
    guidance_scale = values.get("guidance_scale", 6.0)
    seed = values.get("seed", 42)
    num_inference_steps = values.get("num_inference_steps", 50)
    base_resolution = values.get("base_resolution", 512)
    
    video_length = values.get("video_length", 53)
    fps = values.get("fps", 10)
    lora_weight = values.get("lora_weight", 1.00)
    save_path = "samples"
    partial_video_length = values.get("partial_video_length", None)
    overlap_video_length = values.get("overlap_video_length", 4)
    validation_image_start = values.get("validation_image_start", "asset/1.png")
    validation_image_end = values.get("validation_image_end", None)
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    aspect_ratio_sample_size = {key : [x / 512 * base_resolution for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
    start_img = Image.open(validation_image_start)
    original_width, original_height = start_img[0].size if isinstance(start_img, list) else Image.open(start_img).size
    closest_size, closest_ratio = get_closest_ratio(original_height, original_width, ratios=aspect_ratio_sample_size)
    height, width = [int(x / 16) * 16 for x in closest_size]
    sample_size = [height, width]
    if partial_video_length is not None:
        # Handle ultra-long video generation if required
        # ... (existing logic for partial video generation)
    else:
        # Standard video generation
        video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, validation_image_end, video_length=video_length, sample_size=sample_size)
        
        with torch.no_grad():
            sample = pipeline(
                prompt=prompt,
                num_frames=video_length,
                negative_prompt=negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                video=input_video,
                mask_video=input_video_mask
            ).videos
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    video_path = os.path.join(save_path, f"{prefix}.mp4")
    save_videos_grid(sample, video_path, fps=fps)

    # Upload final video to Hugging Face repository
    #hf_api = HfApi()
    #repo_id = values.get("repo_id", "your-username/your-repo")  # Set your HF repo
    #hf_api.upload_file(
    #    path_or_fileobj=video_path,
    #    path_in_repo=f"{prefix}.mp4",
    #    repo_id=repo_id,
    #    repo_type="model"  # or "dataset" if using a dataset repo
    #)

    # Prepare output
    #result_url = f"https://huggingface.co/{repo_id}/blob/main/{prefix}.mp4"
    result_url = ""
    job_id = values.get("job_id", "default-job-id")  # For RunPod job tracking
    return {"jobId": job_id, "result": result_url, "status": "DONE"}

runpod.serverless.start({"handler": generate})
