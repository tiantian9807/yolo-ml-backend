from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import torch

# 禁用 weights_only 检查
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except:
    pass

from ultralytics import YOLO
import requests
from PIL import Image
import io
import traceback

app = Flask(__name__)
CORS(app)

# 全局变量
model = None
classes = {}
conf_threshold = 0.3

def init_model():
    """初始化 YOLO 模型"""
    global model, classes
    
    print("🔧 开始加载 YOLOv8 模型...")
    try:
        model = YOLO('best.pt')
        classes = model.names
        print(f"✅ 模型加载成功！")
        print(f"📋 检测类别: {list(classes.values())}")
        print(f"🎯 置信度阈值: {conf_threshold}")
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        traceback.print_exc()
        return False

# 启动时初始化模型
print("🚀 初始化应用...")
if init_model():
    print("✅ 应用启动成功")
else:
    print("⚠️ 模型加载失败，但应用继续运行")

@app.route('/', methods=['GET'])
def index():
    """根路径"""
    return jsonify({
        "name": "YOLOv8 ML Backend for Label Studio",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "classes": list(classes.values()) if classes else [],
        "endpoints": {
            "/health": "GET - 健康检查",
            "/setup": "POST/GET - 模型配置",
            "/predict": "POST - 模型预测",
            "/webhook": "POST - Label Studio 回调"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "UP",
        "model_loaded": model is not None
    }), 200

@app.route('/setup', methods=['POST', 'GET'])
def setup():
    """模型设置端点"""
    try:
        print("📡 收到 setup 请求")
        
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        response = {
            "model_version": "yolov8-1.0.0",
            "score_threshold": conf_threshold,
            "labels": list(classes.values())
        }
        
        print(f"✅ 返回配置: {response}")
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Setup 错误: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """预测端点"""
    try:
        print("\n🔮 收到预测请求")
        
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.json
        print(f"📦 请求数据: {data}")
        
        if not data or 'tasks' not in data:
            return jsonify({"error": "Missing 'tasks' in request"}), 400
        
        tasks = data['tasks']
        predictions = []
        
        for task in tasks:
            try:
                print(f"\n📥 处理任务 ID: {task.get('id')}")
                
                # 获取图片URL
                image_url = task['data'].get('image')
                if not image_url:
                    raise ValueError("任务中未包含图片地址")
                
                print(f"🖼️  图片URL: {image_url}")
                
                # 加载图片
                if image_url.startswith(('http://', 'https://')):
                    print("⬇️  下载网络图片...")
                    response = requests.get(image_url, timeout=30)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                else:
                    print("📁 加载本地图片...")
                    image = Image.open(image_url)
                
                img_width, img_height = image.size
                print(f"✅ 图片尺寸: {img_width}x{img_height}")
                
                # YOLOv8 推理
                print("🔍 正在预测...")
                results = model.predict(image, conf=conf_threshold, verbose=False)
                
                # 转换为 Label Studio 格式
                label_studio_results = []
                for result in results[0].boxes:
                    x1, y1, x2, y2 = result.xyxy[0].tolist()
                    class_id = int(result.cls[0])
                    confidence = float(result.conf[0])
                    class_name = classes.get(class_id, f"class_{class_id}")
                    
                    label_studio_results.append({
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "rectanglelabels": [class_name],
                            "x": (x1 / img_width) * 100,
                            "y": (y1 / img_height) * 100,
                            "width": ((x2 - x1) / img_width) * 100,
                            "height": ((y2 - y1) / img_height) * 100
                        },
                        "score": confidence
                    })
                
                print(f"📊 检测到 {len(label_studio_results)} 个目标")
                
                predictions.append({
                    "result": label_studio_results,
                    "score": sum(r["score"] for r in label_studio_results) / len(label_studio_results) if label_studio_results else 0,
                    "model_version": "yolov8-1.0.0"
                })
                
            except Exception as e:
                print(f"❌ 处理任务失败: {str(e)}")
                traceback.print_exc()
                predictions.append({"result": [], "score": 0})
        
        response = {"results": predictions}
        print(f"✅ 返回 {len(predictions)} 个预测结果")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ 预测错误: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook 端点"""
    try:
        print("📨 收到 webhook")
        data = request.json
        print(f"📦 Webhook 数据: {data}")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        print(f"❌ Webhook 错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=False)
