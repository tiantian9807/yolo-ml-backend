import os
import torch

# 方案1：完全禁用 weights_only 检查（最简单）
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

# 方案2：添加安全类（作为备份）
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception as e:
    print(f"⚠️ 添加安全类失败: {e}")

from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
import requests
from PIL import Image
import io
from typing import List, Dict, Optional

class YOLOv8LabelStudioAdapter(LabelStudioMLBase):
    """YOLOv8模型与Label Studio的对接适配器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("🔧 开始加载 YOLOv8 模型...")
        
        try:
            # 加载 YOLOv8 模型时禁用 weights_only 检查
            self.model = YOLO('best.pt')
            
            # 获取模型类别
            self.classes = self.model.names  # 这是一个字典 {0: 'class1', 1: 'class2', ...}
            print(f"✅ 模型加载成功！")
            print(f"📋 检测类别: {list(self.classes.values())}")
            
            # 置信度阈值
            self.conf_threshold = 0.3
            print(f"🎯 置信度阈值: {self.conf_threshold}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def setup(self):
        """
        Label Studio 调用此方法来验证模型配置
        返回模型的标签配置
        """
        print("📡 收到 setup 请求")
        
        # 返回模型支持的标签
        self.parsed_label_config = {
            "label": {
                "type": "rectanglelabels",
                "to_name": ["image"],
                "inputs": [
                    {"type": "image", "value": "image"}
                ],
                "labels": list(self.classes.values())  # 返回所有检测类别
            }
        }
        
        print(f"✅ Setup 完成，支持的标签: {list(self.classes.values())}")
        return self.parsed_label_config

    def predict(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        """
        处理Label Studio的标注任务，返回模型预测结果
        """
        predictions = []
        
        for task in tasks:
            try:
                print(f"\n📥 收到预测任务 ID: {task.get('id')}")
                
                # 1. 获取图片URL
                image_url = task['data'].get('image')
                if not image_url:
                    raise ValueError("任务中未包含图片地址")
                
                print(f"🖼️  原始图片URL: {image_url}")
                
                # 2. 加载图片（支持网络图片和本地图片）
                if image_url.startswith(('http://', 'https://')):
                    # 网络图片：直接下载
                    print("⬇️  下载网络图片...")
                    response = requests.get(image_url, timeout=30)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                else:
                    # 本地图片（Label Studio本地部署）
                    print("📁 加载本地图片...")
                    image = Image.open(image_url)
                
                img_width, img_height = image.size
                print(f"✅ 图片加载成功，尺寸: {img_width}x{img_height}")

                # 3. YOLOv8 推理
                print("🔍 正在进行模型预测...")
                results = self.model.predict(
                    image,
                    conf=self.conf_threshold,
                    verbose=False
                )

                # 4. 转换为 Label Studio 格式
                label_studio_results = []
                for result in results[0].boxes:
                    x1, y1, x2, y2 = result.xyxy[0].tolist()
                    class_id = int(result.cls[0])
                    confidence = float(result.conf[0])
                    
                    # 获取类别名称
                    class_name = self.classes.get(class_id, f"class_{class_id}")
                    
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

                print(f"📊 预测完成，检测到 {len(label_studio_results)} 个目标")
                
                predictions.append({
                    "result": label_studio_results,
                    "score": sum(r["score"] for r in label_studio_results) / len(label_studio_results) if label_studio_results else 0,
                    "model_version": "yolov8-public"
                })

            except Exception as e:
                print(f"❌ 处理任务 {task.get('id')} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                predictions.append({
                    "result": [],
                    "score": 0
                })

        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        """
        训练方法（可选）
        如果不需要在线训练，返回空字典即可
        """
        print("📚 收到训练请求（当前版本不支持在线训练）")
        return {}
