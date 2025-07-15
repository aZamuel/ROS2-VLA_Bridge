from flask import Flask, request, jsonify
from vla_wrapper import VLAWrapper

app = Flask(__name__)
vla = VLAWrapper(model_name="pi0")

#@app.route("/predict", methods=["POST"])
#def predict():
#    instruction = request.json.get("instruction")
#    if not instruction:
#        return jsonify({"error": "No instruction provided"}), 400
#
#    result = vla.predict(image=None, instruction=instruction)
#    return jsonify(result)

@app.route('/api/vla_request', methods=['POST'])
def vla_request():
    data = request.get_json()
    print("Received request:")
    print(data)
    return jsonify({"status": "received"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)