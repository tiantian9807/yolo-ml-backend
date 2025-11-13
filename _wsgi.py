from label_studio_ml.api import init_app
from model import YOLOv8LabelStudioAdapter
from flask import jsonify

# 初始化应用
app = init_app(model_class=YOLOv8LabelStudioAdapter)

# 添加健康检查端点
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "UP"}), 200

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "status": "running",
        "model": "YOLOv8",
        "version": "1.0"
    }), 200
