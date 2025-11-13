import torch
# 添加 Ultralytics 模型到安全全局变量
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
import requests
from PIL import Image
import io
from typing import List, Dict
import os

class YOLOv8LabelStudioAdapter(LabelStudioMLBase):
    """YOLOv8模型与Label Studio的对接适配器"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 检查模型文件是否存在
        if not os.path.exists('best.pt'):
            raise FileNotFoundError("best.pt 文件不存在，请确保模型文件已上传")
        
        # 加载 YOLOv8 模型
        self.model = YOLO('best.pt')
        
        # 类别列表（必须与训练时的顺序一致）
        self.classes = ['qiwu', 'xuhao', 'tuzhu', 'muzang', 'zhengti']
        
        # 置信度阈值
        self.conf_threshold = 0.3

    def predict(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        """
        处理Label Studio的标注任务，返回模型预测结果
        """
        predictions = []
        
        for task in tasks:
            try:
                # 1. 获取图片URL
                image_url = task['data']['image']
                if not image_url:
                    raise ValueError("任务中未包含图片地址")

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
                    classes=None
                )

                # 4. 转换为 Label Studio 格式
                label_studio_results = []
                for result in results[0].boxes:
                    x1, y1, x2, y2 = result.xyxy[0].tolist()
                    class_id = int(result.cls[0])
                    confidence = float(result.conf[0])

                    # 验证类别ID
                    if class_id < 0 or class_id >= len(self.classes):
                        label = f"unknown_class_{class_id}"
                    else:
                        label = self.classes[class_id]

                    # 转换为百分比坐标
                    img_width, img_height = image.size
                    label_studio_results.append({
                        "type": "rectanglelabels",
                        "value": {
                            "x": (x1 / img_width) * 100,
                            "y": (y1 / img_height) * 100,
                            "width": ((x2 - x1) / img_width) * 100,
                            "height": ((y2 - y1) / img_height) * 100,
                            "labels": [label]
                        },
                        "score": confidence
                    })

                # 5. 返回预测结果
                predictions.append({
                    "result": label_studio_results,
                    "task": task["id"],
                    "model_version": "yolov8-custom"
                })

            except Exception as e:
                print(f"处理任务 {task.get('id')} 时出错: {str(e)}")
                predictions.append({
                    "result": [],
                    "task": task["id"],
                    "error": str(e)
                })

        return predictions
