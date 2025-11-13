from label_studio_ml.api import init_app
from model import YOLOv8LabelStudioAdapter
from flask import jsonify
from flask_cors import CORS

# 初始化应用
app = init_app(model_class=YOLOv8LabelStudioAdapter)

# 启用 CORS
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/health')
def health():
    return jsonify({"status": "UP"}), 200

@app.route('/')
def index():
    return jsonify({"message": "YOLOv8 Backend Running"}), 200
