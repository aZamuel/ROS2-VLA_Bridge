import logging
from typing import Any, Dict
import base64
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
import os
import json
from pathlib import Path

# Fixed local export of your fine-tuned OpenVLA (override via env if you like)
LOCAL_OPENVLA_DIR = os.environ.get(
    "OPENVLA_LOCAL_DIR",
    os.path.expanduser(
        "/home/srochlitzer/Desktop/vla-finetuning/checkpoints/openvla-7b+openvla_finetune_franka3+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--30000_chkpt"
    )
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VLAWrapper")

def _hf_dir_ok(p: str) -> bool:
    try:
        return (Path(p) / "config.json").exists()
    except Exception:
        return False

class VLAWrapper:
    def __init__(self, model_name: str = "openvla/openvla-7b"):
        self.model_name = model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.processor, self.dtype = self._load_model()
        self.unnorm_key_name = "bridge_orig"

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

        # [ADDED] choose local-vs-hub source when model_name == openvla/openvla-7b
        source = self.model_name
        if self.model_name == "openvla/openvla-7b" and _hf_dir_ok(LOCAL_OPENVLA_DIR):
            source = LOCAL_OPENVLA_DIR
            logger.info(f"Loading OpenVLA from local directory: {source}")
        elif self.model_name == "openvla/openvla-7b":
            logger.warning(
                f"Local fine-tune not found at {LOCAL_OPENVLA_DIR}; falling back to HF hub '{self.model_name}'."
            )

        # [CHANGED] load from `source` (could be local dir or HF repo id)
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            source,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device).eval()

        # [ADDED] optional dataset statistics for local runs (mirrors deploy.py behavior)
        try:
            if os.path.isdir(source):
                stats_path = Path(source) / "dataset_statistics.json"
                if stats_path.exists():
                    with open(stats_path, "r") as f:
                        model.norm_stats = json.load(f)
                    logger.info(f"Loaded dataset_statistics.json from {stats_path}")
                    self.unnorm_key_name = "openvla_finetune_franka3"
                else:
                    logger.warning("No dataset_statistics.json found; proceeding without custom unnorm stats.")
        except Exception as e:
            logger.warning(f"Failed to load dataset statistics: {e}")

        logger.info(
            f"OpenVLA loaded from '{source}' on {self.device} "
            f"(dtype={dtype}, flash_attn={attn_impl is not None})"
        )
        return model, processor, dtype

    def predict(self, bgr: np.ndarray, instruction: str) -> Dict[str, float]:
        # 1) Decode Base64 → NumPy(BGR) → PIL(RGB)
        try:
            pil_img = self._pil_from_bgr_openvla7b(bgr)
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
            action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key_name, do_sample=False)

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
    
    def _pil_from_bgr_openvla7b(self, bgr: np.ndarray) -> Image.Image:
        # BGR → RGB → 224×224 → PIL(RGB)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        return Image.fromarray(rgb).convert("RGB")

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
    print("Output:", result)