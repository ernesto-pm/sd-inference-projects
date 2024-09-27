import arguably
import torch
from pathlib import Path
from utils.json_utils import load_json_file
from pydantic import BaseModel, field_validator, ValidationInfo
from typing import List
from enum import Enum
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler

class DeviceTypeEnum(str, Enum):
    mps = "mps"
    cuda = "cuda"

class SamplerTypeEnum(str, Enum):
    dpmppsdekarras = "DPM++ SDE Karras"

class LoraLikenessConfig(BaseModel):
    loras_directory_path: Path  # dir
    results_path: Path  # dir
    sd_model_path: Path  # file
    face_compare_image_path: Path  # file
    trigger_word: str
    seed: int
    prompts: List[List[str]]
    device: DeviceTypeEnum
    guidance_scale: float
    inference_steps: int
    sampler: SamplerTypeEnum
    samples_width: int
    samples_height: int

    @field_validator('loras_directory_path', 'sd_model_path', 'face_compare_image_path', 'results_path')
    def validate_path_existence(cls, v: Path, info: ValidationInfo) -> Path:
        assert v.exists(), f"Path: {v} does not exist"
        return v

    @field_validator('loras_directory_path', 'results_path')
    def validate_is_directory(cls, v: Path, info: ValidationInfo) -> Path:
        assert v.is_dir(), f"Path: {v} must be a directory"
        return v

    @field_validator('sd_model_path', 'face_compare_image_path')
    def validate_is_file(cls, v: Path, info: ValidationInfo) -> Path:
        assert v.is_file(), f"Path: {v} must be a file"
        return v

    @field_validator('sd_model_path')
    def validate_safetensors_File(cls, v: Path, info: ValidationInfo) -> Path:
        assert v.suffix == ".safetensors", f"Path: {v} must be a .safetensors file"
        return v

    @field_validator('face_compare_image_path')
    def validate_image_file(cls, v: Path, info: ValidationInfo) -> Path:
        assert v.suffix == ".jpg" or v.suffix == ".png" or v.suffix == ".jpeg", f"Path: {v} must be an image file: .png, .jpeg, .jpg"

        return v



def get_lora_file_paths(lora_directory_path: Path) -> List[Path]:
    loras = []
    for file in lora_directory_path.glob("*.safetensors"):
        loras.append(file)

    return loras

@arguably.command
def test_lora_likeness(lora_likeness_config_path: str):
    config_path = Path(lora_likeness_config_path)
    if not config_path.exists():
        raise Exception("Error, the path to the config doesn't exist")
    config = LoraLikenessConfig.model_validate(load_json_file(config_path))

    # Get all the paths for the loras we want to try out
    loras = get_lora_file_paths(config.loras_directory_path)

    # Create the sdxl pipeline we shall use
    pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_single_file(str(config.sd_model_path), torch_dtype=torch.float16)
    pipe.to(config.device.value)
    if config.sampler == SamplerTypeEnum.dpmppsdekarras:
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)  # ToDo: add scheduler to config options
    generator = torch.Generator(device=config.device).manual_seed(config.seed)

    # We run the inference and save the images
    for lora_path in loras:
        # load the lora
        lora_filename = lora_path.stem
        pipe.load_lora_weights(str(lora_path), adapter_name=lora_filename)
        pipe.set_adapters([lora_filename], adapter_weights=[1.0])

        # we go through all prompts and save the results
        for prompt_index, p in enumerate(config.prompts):
            prompt = p[0].replace("{triggerword}", config.trigger_word)
            negative_prompt = p[1].replace("{triggerword}", config.trigger_word)
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.inference_steps,
                width=config.samples_width,
                height=config.samples_height,
                generator=generator
            )

            img_filename = config.results_path.joinpath(f"{lora_filename}_prompt_{prompt_index}.png")
            output.images[0].save(img_filename)

        # unload the lora
        pipe.unload_lora_weights()
        pipe.delete_adapters([lora_filename])


        # Debug, check if this works lol
        '''
        for prompt_index, p in enumerate(config.prompts):
            prompt = p[0].replace("{triggerword}", config.trigger_word)
            negative_prompt = p[1].replace("{triggerword}", config.trigger_word)
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.inference_steps,
                width=config.samples_width,
                height=config.samples_height,
                generator=generator
            )

            img_filename = config.results_path.joinpath(f"{lora_filename}_prompt_{prompt_index}_disabled.png")
            output.images[0].save(img_filename)
        '''

if __name__ == "__main__":
    arguably.run()
