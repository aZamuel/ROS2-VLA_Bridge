import logging
from typing import Any, Dict
import base64
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VLAWrapper")

class VLAWrapper:
    def __init__(self, model_name: str = "openvla"):
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.processor, self.dtype = self._load_model()

    def _load_model(self):
        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32

        # Try enabling FlashAttention if available and compatible
        attn_impl = None
        if use_cuda:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                logger.info("FlashAttention detected, will use flash_attention_2")
            except Exception as e:
                logger.info(f"FlashAttention not available: {e}")

        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device).eval()

        logger.info(f"OpenVLA loaded on {self.device} (dtype={dtype}, flash_attn={attn_impl is not None})")
        return model, processor, dtype

    def predict(self, image: Any, instruction: str) -> Dict[str, float]:
        # 1) Decode Base64 → NumPy(BGR) → PIL(RGB)
        try:
            image_bytes = base64.b64decode(image)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2.imdecode returned None")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
        except Exception as e:
            logger.exception(f"Failed to decode image: {e}")
            # Conservative no-op deltas if vision fails
            return {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0,
                    "delta_roll": 0.0, "delta_pitch": 0.0, "delta_yaw": 0.0,
                    "delta_gripper": 0.0}

        # 2) Prompt (per OpenVLA README format)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"  # 

        # 3) Tokenize & infer
        with torch.no_grad():
            inputs = self.processor(prompt, pil_img).to(self.device)

            use_cuda = torch.cuda.is_available()
            if use_cuda:
                # move everything to GPU WITHOUT changing dtype
                inputs = {k: (v.to("cuda:0") if hasattr(v, "to") else v) for k, v in inputs.items()}
                # cast ONLY image-like tensors to bf16
                for k in ("pixel_values", "pixel_values_fused", "vision_pixels", "image"):
                    if k in inputs and hasattr(inputs[k], "to"):
                        inputs[k] = inputs[k].to("cuda:0", dtype=torch.bfloat16)
            
            # unnorm_key matches BridgeData V2 name in README; adjust if your fine-tune differs. 
            action = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

        # Robust type normalize
        if isinstance(action, torch.Tensor):
            action = action.detach().float().cpu().numpy()
        else:
            action = np.asarray(action)
        vals = (action.ravel().tolist() + [0.0]*7)[:7]
        dx, dy, dz, droll, dpitch, dyaw, grip = vals
        return {"delta_x":float(dx),
                "delta_y":float(dy),
                "delta_z":float(dz),
                "delta_roll":float(droll),
                "delta_pitch":float(dpitch),
                "delta_yaw":float(dyaw),
                "delta_gripper":float(grip)}

if __name__ == "__main__":
    wrapper = VLAWrapper(model_name="openvla/openvla-7b")

    # Load a small local test image
    from PIL import Image
    import base64, cv2, numpy as np

    img = cv2.imread("test.jpg")  # replace with path to any RGB photo
    _, buf = cv2.imencode(".jpg", img)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    prompt = "Pick up the red cube"

    result = wrapper.predict(img_b64, prompt)
    print("Control Output:", result)