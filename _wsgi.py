from label_studio_ml.api import init_app
from model import YOLOv8LabelStudioAdapter
from flask import jsonify, Response

# 初始化应用
app = init_app(model_class=YOLOv8LabelStudioAdapter)

# 健康检查端点 - 返回纯文本
@app.route('/health')
def health():
    return Response("OK", status=200, mimetype='text/plain')

# 根路径
@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "model": "YOLOv8"
    })
