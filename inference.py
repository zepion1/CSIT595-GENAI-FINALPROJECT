import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


DEFAULT_PROMPT = (
    "You are a security camera assistant. In several sentences, describe "
    "what activity is happening in this video. Cover who is present and what "
    "they look like, what they are doing from start to finish of the clip, "
    "any notable objects (packages, vehicles, tools, animals), and anything "
    "that seems unusual or worth flagging. Do not invent details you cannot "
    "actually see in the video."
)


class QwenActivityDescriber:
    """Wraps Qwen2.5-VL for short-clip activity description (Ring-camera style)."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        nframes: int = 8,
        fps: float = 1.0,
        max_pixels: int = 360 * 640,
        max_new_tokens: int = 256,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_id} on {self.device}...")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.nframes = nframes
        self.fps = fps
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens

        print("Model loaded successfully.")

    def describe_activity(
        self,
        video_source: str,
        prompt_text: str = DEFAULT_PROMPT,
    ) -> str:
        """Run activity description on a video file path or URL.

        `video_source` may be a `file://...` URI, an http(s) URL, or a plain
        local path. Qwen's `process_vision_info` handles all three, decoded
        via decord (must be installed).
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_source,
                        "fps": self.fps,
                        "nframes": self.nframes,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        print("Running inference...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text


if __name__ == "__main__":
    describer = QwenActivityDescriber()

    test_video_path = "sample.mp4"

    try:
        result = describer.describe_activity(video_source=test_video_path)
        print("\n--- Activity Description ---")
        print(result)
        print("----------------------------")
    except Exception as e:
        print(f"Error during inference: {e}")
