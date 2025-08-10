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
        # Decode Base64 image
        try:
            image_data = base64.b64decode(image)
            np_arr = np.frombuffer(image_data, np.uint8)
            decoded_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            logger.info("Image decoded successfully.")
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            decoded_image = None

        # Show image for testing
        #if decoded_image is not None:
        #    cv2.imshow("Decoded Image", decoded_image)
        #    cv2.waitKey(10)
        
        """
        Perform inference using the VLA model.

        Args:
            image: A visual input (e.g., numpy array, PIL image, etc.)
            instruction: A natural language instruction.

        Returns:
            A dictionary with pose deltas or joint commands.
        """
        logger.info(f"Received instruction: {instruction}")
        logger.info(f"Received image of type: {type(image)}")

        # Placeholder output
        control_output = {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.05,
            "delta_roll": 0.0,
            "delta_pitch": 0.0,
            "delta_yaw": 0.0,
            "delta_gripper": 0.5
        }

        logger.info(f"Returning dummy control output: {control_output}")
        return control_output

# __main__ (tiny smoke test)
if __name__ == "__main__":
    VLAWrapper(model_name="openvla")
    print("Loaded OpenVLA successfully.")