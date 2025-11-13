from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
import requests
from PIL import Image
import io
from typing import List, Dict  # 类型注解，增强代码可读性

class YOLOv8LabelStudioAdapter(LabelStudioMLBase):
    """YOLOv8模型与Label Studio的对接适配器"""
    def __init__(self, **kwargs):
        super().__init__(** kwargs)
        # 加载训练好的YOLOv8模型（确保best.pt与当前文件同目录）
        self.model = YOLO('best.pt')
        
        # **************************
        # 关键：替换为你的模型实际类别
        # 类别顺序必须与训练时的classes完全一致！
        # 例如：训练时classes为['cat', 'dog', 'bird']，则按此顺序填写
        self.classes = ['qiwu', 'xuhao', 'tuzhu', 'muzang', 'zhengti']  # <--- 必须修改这里！
        # **************************

        # 置信度阈值（过滤低置信度结果，可根据需要调整）
        self.conf_threshold = 0.3

    def predict(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        """
        处理Label Studio的标注任务，返回模型预测结果
        :param tasks: Label Studio发送的任务列表（包含图片数据）
        :return: 转换为Label Studio格式的预测结果
        """
        predictions = []
        
        for task in tasks:
            try:
                # 1. 从任务中获取图片地址
                image_url = task['data']['image']
                if not image_url:
                    raise ValueError("任务中未包含图片地址")

                # 2. 下载或加载图片
                if image_url.startswith(('http://', 'https://')):
                    # 处理网络图片
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()  # 抛出HTTP错误（如404、500）
                    image = Image.open(io.BytesIO(response.content))
                else:
                    # 处理本地图片（适用于Label Studio本地部署）
                    image = Image.open(image_url)

                # 3. 使用YOLOv8模型推理
                results = self.model.predict(
                    image,
                    conf=self.conf_threshold,  # 过滤低置信度结果
                    classes=None  # 不限制类别，使用模型所有类别
                )

                # 4. 转换结果为Label Studio支持的格式
                label_studio_results = []
                for result in results[0].boxes:  # 取第一张图片的检测结果
                    # 获取边界框坐标（x1, y1: 左上角；x2, y2: 右下角，单位：像素）
                    x1, y1, x2, y2 = result.xyxy[0].tolist()
                    # 获取类别ID和置信度
                    class_id = int(result.cls[0])
                    confidence = float(result.conf[0])

                    # 校验类别ID有效性
                    if class_id < 0 or class_id >= len(self.classes):
                        label = f"unknown_class_{class_id}"
                    else:
                        label = self.classes[class_id]

                    # 转换坐标为Label Studio的百分比格式（x, y为左上角百分比）
                    img_width, img_height = image.size
                    label_studio_results.append({
                        "type": "rectanglelabels",  # Label Studio矩形框标注类型
                        "value": {
                            "x": (x1 / img_width) * 100,  # 左上角x百分比
                            "y": (y1 / img_height) * 100,  # 左上角y百分比
                            "width": ((x2 - x1) / img_width) * 100,  # 宽度百分比
                            "height": ((y2 - y1) / img_height) * 100,  # 高度百分比
                            "labels": [label]  # 标注的类别标签
                        },
                        "score": confidence  # 置信度分数（0-1）
                    })

                # 5. 整理当前任务的预测结果
                predictions.append({
                    "result": label_studio_results,  # 标注结果列表
                    "task": task["id"],  # 对应任务ID
                    "model_version": "yolov8-custom"  # 模型版本标识
                })

            except Exception as e:
                # 捕获并记录错误，避免单个任务失败导致整个服务崩溃
                print(f"处理任务{task.get('id')}时出错: {str(e)}")
                # 即使出错，也返回空结果，确保Label Studio能正常接收响应
                predictions.append({
                    "result": [],
                    "task": task["id"],
                    "error": str(e)
                })

        return predictions