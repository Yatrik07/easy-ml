import os
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from easyml.utils.logger import write_log
from easyml.cv.cv_config import CV_RequestBody
from easyml.config.GLOBAL_VARS import error_code_mappings
import traceback

class MultiFolderImageDataset(Dataset):
    def __init__(self, folder_paths, transform=None):
        self.transform = transform
        self.samples = []
        self.classes = []
        
        for p in folder_paths:
            p = p.strip()
            if not os.path.exists(p):
                continue
            # use the folder name as the class name
            cls = os.path.basename(os.path.normpath(p))
            if cls not in self.classes:
                self.classes.append(cls)
            class_idx = self.classes.index(cls)
            
            for root, _, files in os.walk(p):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                        self.samples.append((os.path.join(root, file), class_idx))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            sample = Image.open(path).convert('RGB')
        except Exception:
            # Fallback for unreadable images to avoid crashing midway
            sample = Image.new('RGB', (224, 224))
        
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def _translate_path_for_docker(path: str) -> str:
    """
    Translate a Windows host path to a Docker container path.
    
    When running inside Docker, paths like 'D:\\PycharmProjects\\easy-ml\\test_folder\\X'
    need to be mapped to '/app/test_folder/X' (based on docker-compose volume mount).
    """
    path = path.strip()
    # Normalize backslashes to forward slashes
    normalized = path.replace('\\', '/')
    
    # Check if it's a Windows-style absolute path (e.g. D:/PycharmProjects/easy-ml/test_folder/...)
    # Look for the project's test_folder or artifacts in the path
    for marker in ['test_folder/', 'artifacts/']:
        idx = normalized.lower().find(marker.lower())
        if idx != -1:
            docker_path = '/app/' + normalized[idx:]
            if os.path.exists(docker_path):
                return docker_path
    
    # If the path already exists as-is (e.g. already a container path), return it
    if os.path.exists(path):
        return path
    
    # Last resort: try replacing common Windows prefixes
    if ':' in normalized:
        # Try stripping the drive letter prefix up to the project folder
        for marker in ['test_folder/', 'artifacts/']:
            idx = normalized.lower().find(marker.lower())
            if idx != -1:
                return '/app/' + normalized[idx:]
    
    return path  # Return original if nothing works

def cv_preprocessing_controller(request_body: CV_RequestBody) -> tuple:
    """
    Sets up PyTorch datasets and DataLoaders for Image Classification.
    Returns (train_loader, val_loader, class_names)
    """
    write_log("entered cv_preprocessing_controller", "info", os.path.basename(__file__))
    
    try:
        # Define Transforms
        if request_body.augment:
            train_transform = transforms.Compose([
                transforms.Resize(request_body.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(request_body.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        val_transform = transforms.Compose([
            transforms.Resize(request_body.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Translate paths for Docker environment
        raw_path = request_body.dataset_path
        if ',' in raw_path:
            paths = [_translate_path_for_docker(p) for p in raw_path.split(',')]
            write_log(f"Translated paths: {paths}", "info", os.path.basename(__file__))
            full_dataset = MultiFolderImageDataset(folder_paths=paths, transform=train_transform)
            class_names = full_dataset.classes
        else:
            translated_path = _translate_path_for_docker(raw_path)
            write_log(f"Translated path: {translated_path}", "info", os.path.basename(__file__))
            full_dataset = datasets.ImageFolder(root=translated_path, transform=train_transform)
            class_names = full_dataset.classes
            
        request_body.classes = class_names
        request_body.num_classes = len(class_names)
        
        write_log(f"Loaded dataset from {request_body.dataset_path}. Found {len(class_names)} classes, {len(full_dataset)} images.", "info", os.path.basename(__file__))
        
        # Validate dataset has enough samples
        if len(full_dataset) == 0:
            raise ValueError(
                f"No images found in the dataset path(s): {request_body.dataset_path}. "
                f"Make sure the paths are accessible inside the Docker container. "
                f"Use paths relative to the mounted volume, e.g. '/app/test_folder/your_folder'. "
                f"Supported formats: .png, .jpg, .jpeg, .bmp, .gif, .webp"
            )
        
        if len(class_names) < 2:
            raise ValueError(
                f"Only {len(class_names)} class(es) found. Image classification requires at least 2 classes. "
                f"Each folder path should represent one class."
            )
        
        # Train / Val Split
        val_size = int(len(full_dataset) * request_body.validation_split)
        train_size = len(full_dataset) - val_size
        
        # Ensure at least 1 sample in each split
        if train_size == 0 or val_size == 0:
            train_size = max(1, len(full_dataset) - 1)
            val_size = len(full_dataset) - train_size
            
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Force the val dataset to use val_transform (no augmentation)
        val_dataset.dataset.transform = val_transform
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=request_body.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=request_body.batch_size, shuffle=False, num_workers=0)
        
        write_log(f"Created DataLoaders: Train size={train_size}, Val size={val_size}", "info", os.path.basename(__file__))
        
        return train_loader, val_loader, class_names

    except Exception as e:
        if request_body.status_code == 200:
            request_body.error_message = "PreProcessingError"
            request_body.status_code = error_code_mappings.get(request_body.error_message, 500)
            request_body.error_traceback = traceback.format_exc()
            write_log(f"Error: {request_body.error_message} - {request_body.error_traceback}", "error", os.path.basename(__file__))
        raise e

