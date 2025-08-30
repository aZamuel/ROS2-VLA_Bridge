import base64, io # for debugging
import cv2, numpy as np
from flask import send_file
last_image = None  # latest decoded frame

from flask import Flask, request, jsonify
from vla_wrapper import VLAWrapper

app = Flask(__name__)
vla = VLAWrapper(model_name="openvla/openvla-7b")

@app.route("/predict", methods=["POST"])
def predict():
    instruction = request.json.get("prompt")
    joint_angles = request.json.get("joint_angles")
    image_jason = request.json.get("image")
    model = request.json.get("model")
    if not instruction:
        return jsonify({"error": "No instruction provided"}), 400
    if (model != "openvla/openvla-7b"):
        return jsonify({"error": "Only openvla/openvla-7b available"}), 401
    
    # decode once for debugging
    global last_image
    try:
        buf = np.frombuffer(base64.b64decode(image_jason), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            last_image = img
    except Exception:
        last_image = None

    result = vla.predict(image=image_jason, instruction=instruction)
    return jsonify(result)

@app.route("/debug/last_image.jpg", methods=["GET"])
def debug_last_image_jpg():
    if last_image is None:
        return jsonify({"error": "no image received yet"}), 404
    ok, enc = cv2.imencode(".jpg", last_image)
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    return send_file(io.BytesIO(enc.tobytes()), mimetype="image/jpeg")

@app.route("/debug/view", methods=["GET"])
def debug_view():
    return """
    <html><body>
      <img src="/debug/last_image.jpg" style="max-width:100%%"/>
      <script>setTimeout(()=>location.reload(), 500);</script>
    </body></html>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)