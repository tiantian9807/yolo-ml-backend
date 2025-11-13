from label_studio_ml.api import init_app
from model import YOLOv8LabelStudioAdapter

app = init_app(model_class=YOLOv8LabelStudioAdapter)
