from label_studio_ml.api import init_app
from model import YOLOv8LabelStudioAdapter
from flask import jsonify

app = init_app(model_class=YOLOv8LabelStudioAdapter)

@app.route('/health')
def health():
    return jsonify({"status": "UP"}), 200
