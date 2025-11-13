import os
import sys

# 必须在导入任何库之前设置
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print("=" * 50)
print("🚀 开始初始化 YOLO ML Backend")
print("=" * 50)

# 导入 torch 并配置
import torch
print(f"PyTorch 版本: {torch.__version__}")

# 修改 torch.load 的默认行为
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load
print("✅ 已禁用 PyTorch weights_only 检查")

from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
import requests
from PIL import Image
import io
from typing import List, Dict

print(f"Ultralytics 版本: {YOLO.__module__}")
print(f"当前工作目录: {os.getcwd()}")

# 检查模型文件
if os.path.exists('best.pt'):
    size_mb = os.path.getsize('best.pt') / (1024 * 1024)
    print(f"✅ best.pt 存在，大小: {size_mb:.2f} MB")
else:
    print("❌ best.pt 不存在！")
    raise FileNotFoundError("模型文件 best.pt 未找到")

class YOLOv8LabelStudioAdapter(LabelStudioMLBase):
    """YOLOv8模型与Label Studio的对接适配器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("\n🔧 开始加载 YOLOv8 模型...")
        
        try:
            # 强制加载模型
            self.model = YOLO('best.pt', task='detect')
            print("✅ 模型加载成功！")
            
            # 获取模型信息
            if hasattr(self.model, 'names'):
                print(f"   📋 模型类别数: {len(self.model.names)}")
                print(f"   📋 模型类别: {self.model.names}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 类别列表（与训练时一致）
        self.classes = ['qiwu', 'xuhao', 'tuzhu', 'muzang', 'zhengti']
        self.conf_threshold = 0.3
        
        print(f"🎯 使用类别: {self.classes}")
        print(f"🎯 置信度阈值: {self.conf_threshold}")
        print("=" * 50)

    def predict(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        """
        处理 Label Studio 的标注任务，返回模型预测结果
        """
        print(f"\n📥 收到 {len(tasks)} 个预测任务")
        predictions = []
        
        for idx, task in enumerate(tasks):
            try:
                # 1. 获取图片URL
                image_url = task['data'].get('image')
                if not image_url:
                    raise ValueError("任务中未包含图片地址")

                print(f"🖼️  [{idx+1}/{len(tasks)}] 处理图片: {image_url[:80]}...")

                # 2. 加载图片
                if image_url.startswith(('http://', 'https://')):
                    response = requests.get(image_url, timeout=15)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                else:
                    image = Image.open(image_url)

                img_width, img_height = image.size
                print(f"   图片尺寸: {img_width}x{img_height}")

                # 3. YOLOv8 推理
                results = self.model.predict(
                    image,
                    conf=self.conf_threshold,
                    verbose=False,
                    device='cpu'  # 强制使用 CPU
                )

                # 4. 转换为 Label Studio 格式
                label_studio_results = []
                
                if len(results) > 0 and hasattr(results[0], 'boxes'):
                    for result in results[0].boxes:
                        x1, y1, x2, y2 = result.xyxy[0].tolist()
                        class_id = int(result.cls[0])
                        confidence = float(result.conf[0])

                        # 验证类别ID
                        if 0 <= class_id < len(self.classes):
                            label = self.classes[class_id]
                        else:
                            label = f"unknown_class_{class_id}"
                            print(f"   ⚠️  未知类别ID: {class_id}")

                        # 转换为百分比坐标
                        label_studio_results.append({
                            "type": "rectanglelabels",
                            "value": {
                                "x": (x1 / img_width) * 100,
                                "y": (y1 / img_height) * 100,
                                "width": ((x2 - x1) / img_width) * 100,
                                "height": ((y2 - y1) / img_height) * 100,
                                "rectanglelabels": [label]
                            },
                            "score": confidence,
                            "from_name": "label",
                            "to_name": "image"
                        })

                print(f"   ✅ 检测到 {len(label_studio_results)} 个目标")

                # 5. 返回预测结果
                avg_score = sum(r['score'] for r in label_studio_results) / len(label_studio_results) if label_studio_results else 0.0
                predictions.append({
                    "result": label_studio_results,
                    "score": avg_score
                })

            except Exception as e:
                print(f"   ❌ 处理失败: {str(e)}")
                import traceback
                traceback.print_exc()
                predictions.append({
                    "result": [],
                    "score": 0.0
                })

        print(f"📤 返回 {len(predictions)} 个预测结果\n")
        return predictions

print("✅ model.py 加载完成")
