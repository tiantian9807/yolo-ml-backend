import os
import io
import requests
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO


class YOLOv8LabelStudioAdapter(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(** kwargs)
        print("\n🔧 开始加载 YOLOv8 模型...")

        # -------------------------- 配置参数（需根据实际情况修改） --------------------------
        # Label Studio 基础地址（例如：https://your-label-studio.render.com）
        self.LABEL_STUDIO_BASE_URL = "https://label-studio-latest-96wd.onrender.com"
        # YOLO模型路径（默认加载当前目录的best.pt，可替换为其他模型）
        self.MODEL_PATH = "best.pt"
        # 置信度阈值（过滤低置信度预测，0-1之间）
        self.CONF_THRESHOLD = 0.3
        # ---------------------------------------------------------------------------------

        try:
            # 加载YOLO模型
            self.model = YOLO(self.MODEL_PATH, task='detect')
            # 获取模型类别信息
            self.classes = self.model.names
            self.class_names = list(self.classes.values())
            print(f"✅ 模型加载成功！")
            print(f"   📋 模型类别数: {len(self.classes)}")
            print(f"   📋 模型类别: {self.classes}")
            print(f"🎯 使用类别: {self.class_names}")
            print(f"🎯 置信度阈值: {self.CONF_THRESHOLD}")
            print("==================================================")
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            raise  # 中断启动，确保模型加载成功

    def predict(self, tasks, **kwargs):
        """处理预测任务，返回Label Studio格式的结果"""
        try:
            # 1. 提取图片URL
            image_url = task["data"]["image"]  # Label Studio默认图片字段为"image"
            print(f"\n📥 收到预测任务，图片URL: {image_url}")

            # 2. 处理图片URL（确保为完整HTTP地址）
            if not image_url.startswith(('http://', 'https://')):
                # 移除开头多余斜杠，避免拼接后出现//
                image_url = image_url.lstrip('/')
                # 拼接完整URL
                image_url = f"{self.LABEL_STUDIO_BASE_URL}/{image_url}"
                print(f"🔄 转换后图片URL: {image_url}")

            # 3. 从环境变量获取Label Studio Token并构造认证头
            ls_token = os.getenv("LABEL_STUDIO_TOKEN")
            if not ls_token:
                raise ValueError("环境变量 LABEL_STUDIO_TOKEN 未设置，请在云平台配置")
            headers = {"Authorization": f"Token {ls_token}"}

            # 4. 下载图片（带认证）
            print(f"🖼️ 正在下载图片...")
            response = requests.get(image_url, headers=headers, timeout=15)
            response.raise_for_status()  # 检查HTTP错误（4xx/5xx）
            image = Image.open(io.BytesIO(response.content))
            img_width, img_height = image.size
            print(f"✅ 图片下载成功，尺寸: {img_width}x{img_height}")

            # 5. 用YOLO模型预测
            print("🔍 正在进行模型预测...")
            results = self.model.predict(
                image,
                conf=self.CONF_THRESHOLD,
                classes=None  # 预测所有类别，如需过滤可指定列表（如[0,1]）
            )

            # 6. 转换预测结果为Label Studio格式（矩形框）
            predictions = []
            for result in results:
                for box in result.boxes:
                    # YOLO输出格式：x1,y1,x2,y2（绝对坐标，左上角和右下角）
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # 类别ID和置信度
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.classes[cls_id]

                    # 构造Label Studio矩形框格式
                    predictions.append({
                        "from_name": "label",  # 需与Label Studio标签配置中的name一致
                        "to_name": "image",   # 需与Label Studio图片字段name一致
                        "type": "rectanglelabels",
                        "value": {
                            "rectanglelabels": [class_name],
                            "x": (x1 / img_width) * 100,  # 转换为相对宽度百分比
                            "y": (y1 / img_height) * 100, # 转换为相对高度百分比
                            "width": ((x2 - x1) / img_width) * 100,
                            "height": ((y2 - y1) / img_height) * 100
                        },
                        "score": confidence  # 置信度分数
                    })

            print(f"📊 预测完成，生成 {len(predictions)} 个目标框")
            return {"results": predictions}

        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            print(f"❌ {error_msg}")
            return {"results": [], "error": error_msg}


if __name__ == "__main__":
    # 本地测试用（运行脚本时启动服务）
    from label_studio_ml.server import run_server
    run_server(YOLOv8LabelStudioAdapter, port=8000)

