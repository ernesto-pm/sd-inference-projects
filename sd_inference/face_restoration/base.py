from diffusers import (
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from functools import cached_property
from diffusers.utils import BaseOutput
from dataclasses import dataclass
from PIL import Image
from typing import Any, Callable, Iterable, List, Mapping, Optional
from sd_inference.face_restoration.utils import composite, mask_dilate, mask_gaussian_blur, bbox_padding
from sd_inference.face_restoration.yolo import yolo_detector
import inspect

DetectorType = Callable[[Image.Image], Optional[List[Image.Image]]]

@dataclass
class ADOutput(BaseOutput):
    images: list[Image.Image]
    init_images: list[Image.Image]

class RestoreFacePipeline(StableDiffusionXLPipeline):
    @cached_property
    def inpaint_pipeline(self):
        print("Loading StableDiffusionXLInpaintPipeline")
        return StableDiffusionXLInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=self.scheduler,
            feature_extractor=self.feature_extractor,
        )

    def __call__(  # noqa: C901
            self,
            common: Mapping[str, Any] | None = None,
            inpaint_only: Mapping[str, Any] | None = None,
            images: Image.Image | Iterable[Image.Image] | None = None,
            detectors: DetectorType | Iterable[DetectorType] | None = None,
            mask_dilation: int = 4,
            mask_blur: int = 4,
            mask_padding: int = 32,

            model_path: str = None
    ):
        if common is None:
            common = {}
        if inpaint_only is None:
            inpaint_only = {}
        if "strength" not in inpaint_only:
            inpaint_only = {**inpaint_only, "strength": 0.4}

        if detectors is None:
            detectors = [self.default_detector]
        elif not isinstance(detectors, Iterable):
            detectors = [detectors]

        txt2img_images = None
        if images is None:
            print("No Generated image found")
        else:
            txt2img_images = [images] if not isinstance(images, Iterable) else images
            print("Inpainting...")

        init_images = []
        final_images = []

        for i, init_image in enumerate(txt2img_images):
            init_images.append(init_image.copy())
            final_image = None

            for j, detector in enumerate(detectors):
                if model_path:
                    masks = detector(init_image, model_path=model_path)
                else:
                    masks = detector(init_image)

                if masks is None:
                    print(f"No object detected on {i+1} image with {j+1} detector")
                    continue

                for k, mask in enumerate(masks):
                    mask = mask.convert("L")
                    mask = mask_dilate(mask, mask_dilation)
                    bbox = mask.getbbox()
                    if bbox is None:
                        print(f"No object in {k+1} mask")
                        continue
                    mask = mask_gaussian_blur(mask, mask_blur)
                    mask.save('./mask.png') #uncomment to save and visualize the generated mask

                    bbox_padded = bbox_padding(bbox, init_image.size, mask_padding)
                    print("padded dim:",bbox_padded)
                    inpaint_output = self.process_inpainting(
                        common,
                        inpaint_only,
                        init_image,
                        mask,
                        bbox_padded,
                    )
                    inpaint_image = inpaint_output[0][0]
                    print("generated inpaint dim:",inpaint_image.size) ## remove
                    final_image = composite(
                        init_image,
                        mask,
                        inpaint_image,
                        bbox_padded,
                    )
                    init_image = final_image

            if final_image is not None:
                final_images.append(final_image)

        return ADOutput(images=final_images, init_images=init_images)

    @property
    def default_detector(self) -> Callable[..., list[Image.Image] | None]:
        return yolo_detector

    def _get_inpaint_args(
            self, common: Mapping[str, Any], inpaint_only: Mapping[str, Any]
    ):
        common = dict(common)
        sig = inspect.signature(self.inpaint_pipeline)
        if (
                "control_image" in sig.parameters
                and "control_image" not in common
                and "image" in common
        ):
            common["control_image"] = common.pop("image")
        return {
            **common,
            **inpaint_only,
            "num_images_per_prompt": 1,
            "output_type": "pil",
        }

    def process_inpainting(
            self,
            common: Mapping[str, Any],
            inpaint_only: Mapping[str, Any],
            init_image: Image.Image,
            mask: Image.Image,
            bbox_padded: tuple[int, int, int, int],
    ):
        crop_image = init_image.crop(bbox_padded)
        crop_mask = mask.crop(bbox_padded)
        inpaint_args = self._get_inpaint_args(common, inpaint_only)
        inpaint_args["image"] = crop_image
        inpaint_args["mask_image"] = crop_mask

        if "control_image" in inpaint_args:
            inpaint_args["control_image"] = inpaint_args["control_image"].resize(
                crop_image.size
            )

        print(f"Running inpainting with arguments: {inpaint_args}")

        return self.inpaint_pipeline(**inpaint_args)
