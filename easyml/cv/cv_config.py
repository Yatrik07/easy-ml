from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

AVAILABLE_CV_MODELS: List[str] = [
    "SimpleCNN",
    "ResNet18",
    "MobileNetV2",
    "CustomCNN",
]

class CV_RequestBody(BaseModel):
    """
    Schema for Computer Vision training requests.
    """
    task: str = "ImageClassification"
    name: str = "cv_run"
    dataset_path: str = "" # Path to directory with class subfolders
    
    # Preprocessing & augmentation
    image_size: tuple = (224, 224)
    batch_size: int = 32
    validation_split: float = 0.2
    augment: bool = True
    
    # Model & Training Config
    models: List[str] = Field(default_factory=list)
    epochs: int = 10
    learning_rate: float = 0.001
    
    # Internal state managed during pipeline execution
    status_code: int = 200
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    classes: List[str] = Field(default_factory=list)
    num_classes: int = 0
    run_id: Optional[int] = None
    path_to_save: str = ""
    custom_layers: Optional[List[Dict[str, Any]]] = None  # For CustomCNN layer spec
    use_gpu: bool = False  # GPU toggle
    
class CV_InferenceBody(BaseModel):
    """
    Schema for Computer Vision inference requests.
    """
    run_id: int
    model_name: str
    image_base64: str = ""
    
    status_code: int = 200
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    task: str = ""
    classes: List[str] = Field(default_factory=list)
