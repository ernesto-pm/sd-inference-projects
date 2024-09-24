import torch
from sd_inference.face_restoration.base import RestoreFacePipeline
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

model_path = "/Users/ghost/code/stable-diffusion-webui/models/Stable-diffusion/dreamshaperXL_v21TurboDPMSDE.safetensors"
lora_path = "/Users/ghost/Downloads/barbara_lora_sdxl_v1-000008.safetensors"
pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16)
pipe.to('mps')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(lora_path, adapter_name='one')
pipe.set_adapters(['one'], adapter_weights=[1.0])


# Generate the original image
torch.manual_seed(0)
output = pipe(
    prompt='A realistic photo of ohwx woman wearing red lingerie',
    negative_prompt='noisy, bad quality',
    guidance_scale=2,
    num_inference_steps=8,
    width=1024,
    height=1024
)

output.images[0].save("./original.png")

face_restore_images = output.images
face_restore_components = pipe.components
face_restore_pipe = RestoreFacePipeline(**face_restore_components)
face_restore_pipe.to('mps')

result = face_restore_pipe(
    common={"prompt": "A realistic photo of ohwx woman", "negative_prompt": "noisy, bad quality", "num_inference_steps": 8, "guidance_scale":2, "target_size": (1024, 1024)},
    inpaint_only={"strength": 0.4},
    images=face_restore_images,
    mask_dilation=4,
    mask_blur=4,
    mask_padding=128,
    model_path='./models/face_detection/face_yolov8n.pt'
)

result.images[0].save('./fix.png')



