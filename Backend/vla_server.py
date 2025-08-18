from flask import Flask, request, jsonify
from vla_wrapper import VLAWrapper

app = Flask(__name__)
vla = VLAWrapper(model_name="openvla/openvla-7b")

@app.route("/predict", methods=["POST"])
def predict():
    instruction = request.json.get("prompt")
    joint_angles = request.json.get("joint_angles")
    image_jason = request.json.get("image")
    if not instruction:
        return jsonify({"error": "No instruction provided"}), 400

    result = vla.predict(image=image_jason, instruction=instruction)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)