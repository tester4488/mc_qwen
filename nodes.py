from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

import node_helpers
import comfy.utils
import math

class McTextEncodeQwenImEditPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="McTextEncodeQwenImEditPlus",
            category="MC nodi/Qwen",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Image.Input("image4", optional=True),
                io.Int.Input("target_size", optional=True, default=896, min=128, max=2048, step=32),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None, image4=None, target_size=896) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3, image4]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe key details of the input image (including any objects, characters, poses, facial features, clothing, setting, textures and style), then explain how the user's text instruction should alter, modify or recreate the image. Generate a new image that meets the user's requirements, which can vary from a small change to a completely new image using inputs as a guide.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(target_size * target_size)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    
                    height = int(samples.shape[2] * scale_by / 32) * 32
                    width = int(samples.shape[3] * scale_by / 32) * 32
                    
                    s = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")                    
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return io.NodeOutput(conditioning)




class MCExtensions(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            McTextEncodeQwenImEditPlus,
        ]


async def comfy_entrypoint() -> MCExtensions:  # ComfyUI calls this to load your extension and its nodes.
    return MCExtensions()
