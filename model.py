import torch
import os

# 方法1：禁用安全检查（最简单）
# 注意：仅在信任模型来源时使用
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

# 或者方法2：正确添加安全全局变量（需要导入类）
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception as e:
    print(f"警告：无法添加安全全局变量: {e}")

from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
import requests
from PIL import Image
import io
from typing import List, Dict

class YOLOv8LabelStudioAdapter(LabelStudioMLBase):
    """YOLOv8模型与Label Studio的对接适配器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 检查模型文件是否存在
        if not os.path.exists('best.pt'):
            raise FileNotFoundError("best.pt 文件不存在，请确保模型文件已上传")
        
        try:
            # 加载 YOLOv8 模型（禁用安全检查）
            self.model = YOLO('best.pt')
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
        
        # 类别列表（必须与训练时的顺序一致）
        self.classes = ['qiwu', 'xuhao', 'tuzhu', 'muzang', 'zhengti']
        
        # 置信度阈值
        self.conf_threshold = 0.3
        
        print(f"📋 加载类别: {self.classes}")
        print(f"🎯 置信度阈值: {self.conf_threshold}")

    def predict(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        """
        处理Label Studio的标注任务，返回模型预测结果
        """
        print(f"📥 收到 {len(tasks)} 个预测任务")
        predictions = []
        
        for task in tasks:
            try:
                # 1. 获取图片URL
                image_url = task['data'].get('image')
                if not image_url:
                    raise ValueError("任务中未包含图片地址")

                print(f"🖼️ 处理图片: {image_url}")

                # 2. 加载图片
                if image_url.startswith(('http://', 'https://')):
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                else:
                    image = Image.open(image_url)

                # 3. YOLOv8 推理
                results = self.model.predict(
                    image,
                    conf=self.conf_threshold,
                    classes=None,
                    verbose=False  # 减少日志输出
                )

                # 4. 转换为 Label Studio 格式
                label_studio_results = []
                img_width, img_height = image.size
                
                for result in results[0].boxes:
                    x1, y1, x2, y2 = result.xyxy[0].tolist()
                    class_id = int(result.cls[0])
                    confidence = float(result.conf[0])

                    # 验证类别ID
                    if class_id < 0 or class_id >= len(self.classes):
                        label = f"unknown_class_{class_id}"
                        print(f"⚠️ 未知类别ID: {class_id}")
                    else:
                        label = self.classes[class_id]

                    # 转换为百分比坐标
                    label_studio_results.append({
                        "type": "rectanglelabels",
                        "value": {
                            "x": (x1 / img_width) * 100,
                            "y": (y1 / img_height) * 100,
                            "width": ((x2 - x1) / img_width) * 100,
                            "height": ((y2 - y1) / img_height) * 100,
                            "rectanglelabels": [label]  # 注意：这里改为 rectanglelabels
                        },
                        "score": confidence,
                        "from_name": "label",  # 添加必需字段
                        "to_name": "image"      # 添加必需字段
                    })

                print(f"✅ 检测到 {len(label_studio_results)} 个目标")

                # 5. 返回预测结果
                predictions.append({
                    "result": label_studio_results,
                    "score": sum(r['score'] for r in label_studio_results) / len(label_studio_results) if label_studio_results else 0.0
                })

            except Exception as e:
                print(f"❌ 处理任务 {task.get('id')} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                predictions.append({
                    "result": [],
                    "score": 0.0
                })

        print(f"📤 返回 {len(predictions)} 个预测结果")
        return predictions
