from label_studio_ml.api import init_app
from model import YOLOv8LabelStudioAdapter

# init_app 已经自动创建了 /health, /setup, /predict 等端点
# 不需要手动添加！
app = init_app(model_class=YOLOv8LabelStudioAdapter)

# 如果需要自定义路由，使用不同的路径名
# @app.route('/custom-health')
# def custom_health():
#     return jsonify({"status": "OK"}), 200
