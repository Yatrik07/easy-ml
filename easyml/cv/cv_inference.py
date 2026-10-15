import os
import torch
from torchvision import transforms
from PIL import Image
import base64
import io
import json
import traceback
from easyml.cv.cv_config import CV_InferenceBody
from easyml.config.constants import error_code_mappings
from easyml.utils.logger import write_log
from easyml.utils.database_operations import get_record_by_id
from easyml.cv.cv_model_builder import get_cv_model

class CVInferenceController:
    """
    Handles inference for trained PyTorch CV models from base64 string images.
    """
    def __init__(self, run_id: int, model_name: str, image_base64: str):
        self.inference_body = CV_InferenceBody(
            run_id=run_id,
            model_name=model_name,
            image_base64=image_base64
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def get_inference(self):
        write_log("entered cv get_inference", "info", os.path.basename(__file__))
        try:
            model, class_names, img_size = self.load_objects()
            model = model.to(self.device)
            model.eval()
            
            # Decode and process image
            image = self._preprocess_image(self.inference_body.image_base64, img_size)
            image = image.to(self.device)
            
            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_class_idx = torch.max(probabilities, 1)
                
            prediction = class_names[top_class_idx.item()]
            confidence = top_prob.item()
            
            return {
                "prediction": prediction,
                "probability": confidence,
                "confidence": confidence,
            }
            
        except Exception as e:
            if self.inference_body.status_code == 200:
                self.inference_body.error_message = "CVInferenceError"
                self.inference_body.status_code = error_code_mappings.get(self.inference_body.error_message, 500)
                self.inference_body.error_traceback = traceback.format_exc()
            raise e

    def load_objects(self):
        record = get_record_by_id(self.inference_body.run_id)
        folder_path = record["folder_path"]
        
        # Load Config
        config_path = os.path.join(folder_path, "config_json.json")
        with open(config_path, "r", encoding="utf-8") as f:
            conf = json.load(f)
            
        class_names = conf["classes"]
        num_classes = conf["num_classes"]
        img_size = conf["image_size"]
        custom_layers = conf.get("custom_layers", None)
        
        # Load Model architecture and weights (kept in memory while the weights file is unchanged)
        from easyml.core.model_cache import cached_load

        model_path = os.path.join(folder_path, f"{self.inference_body.model_name}.pth")

        def build():
            model = get_cv_model(self.inference_body.model_name, num_classes, custom_layers=custom_layers,
                                 image_size=img_size, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            return model.to(self.device).eval()

        model = cached_load("cv_model", model_path, build,
                            extra=(str(self.device), json.dumps([num_classes, img_size, custom_layers], default=str)))
        
        return model, class_names, img_size
        
    def _preprocess_image(self, b64_str: str, img_size: list):
        # Remove data URI prefix if present
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]

        img_bytes = base64.b64decode(b64_str)
        return self._preprocess_pil(Image.open(io.BytesIO(img_bytes)), img_size)

    @staticmethod
    def _preprocess_pil(img, img_size):
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img.convert("RGB")).unsqueeze(0)  # Add batch dimension

    def load_model(self):
        """Load (once) and return ``(model, class_names, image_size)`` ready for inference."""
        if not hasattr(self, "_loaded"):
            model, class_names, img_size = self.load_objects()
            model = model.to(self.device)
            model.eval()
            self._loaded = (model, class_names, img_size)
        return self._loaded

    def predict_images(self, paths, batch_size=32):
        """
        Classify image files.

        Returns:
            tuple: (labels, confidences) in the order of ``paths``.
        """
        model, class_names, img_size = self.load_model()

        labels, confidences = [], []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            tensors = []
            for path in chunk:
                with Image.open(path) as img:
                    tensors.append(self._preprocess_pil(img, img_size))
            batch = torch.cat(tensors).to(self.device)
            with torch.no_grad():
                probabilities = torch.nn.functional.softmax(model(batch), dim=1)
                top_prob, top_idx = torch.max(probabilities, 1)
            labels.extend(class_names[i] for i in top_idx.tolist())
            confidences.extend(float(p) for p in top_prob.tolist())
        return labels, confidences
