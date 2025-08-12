import logging
from typing import Any, Dict
import base64
import cv2
import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VLAWrapper")

class VLAWrapper:
    def __init__(self, model_name: str = "openvla"):
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = self._load_model()

    def _load_model(self):
        """Load OpenVLA weights and processor (no inference yet)."""
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            # attn_implementation="flash_attention_2",  # remove if flash-attn not installed
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        logger.info("OpenVLA model loaded successfully!")
        return model, processor

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
            # unnorm_key matches BridgeData V2 name in README; adjust if your fine-tune differs. 
            action = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

        # 4) Map action → deltas (ASSUMPTION)
        # Assumed order: [dx, dy, dz, droll, dpitch, dyaw, grip]
        # If your checkpoint uses a different schema, we’ll swap indices here.
        action = action.detach().float().cpu().numpy().ravel().tolist()
        pad = action + [0.0] * max(0, 7 - len(action))
        dx, dy, dz, droll, dpitch, dyaw, grip = pad[:7]

        return {
            "delta_x": float(dx),
            "delta_y": float(dy),
            "delta_z": float(dz),
            "delta_roll": float(droll),
            "delta_pitch": float(dpitch),
            "delta_yaw": float(dyaw),
            "delta_gripper": float(grip),
        }

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