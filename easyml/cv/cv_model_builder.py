import torch
import torch.nn as nn
import torchvision.models as models
from easyml.cv.cv_config import CV_RequestBody
from easyml.config.GLOBAL_VARS import error_code_mappings
from easyml.utils.logger import write_log
import os
import traceback
import copy
import time
from easyml.utils.socket_emitter import emit_progress

# Global cancel event — set by /stop_training endpoint
import threading
training_cancel_event = threading.Event()

def get_cv_model(model_name: str, num_classes: int, custom_layers: list = None) -> nn.Module:
    """
    Factory function to get PyTorch image classification models.
    """
    if model_name == "SimpleCNN":
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        return model
        
    elif model_name == "ResNet18":
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
        
    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model

    elif model_name == "CustomCNN":
        if not custom_layers:
            raise ValueError("CustomCNN requires 'custom_layers' specification")
        from easyml.cv.custom_cnn import build_custom_cnn
        return build_custom_cnn(custom_layers, input_channels=3,
                                input_size=(224, 224), num_classes=num_classes)
        
    else:
        raise ValueError(f"Unknown CV model: {model_name}")


def train_cv_model(request_body: CV_RequestBody, model_name: str, train_loader, val_loader) -> tuple:
    """
    Trains a single PyTorch model and returns the best model weights and the best accuracy.
    """
    write_log(f"Starting training for {model_name}", "info", os.path.basename(__file__))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    write_log(f"Training on device: {device}", "info", os.path.basename(__file__))
    
    # Initialize Model, Loss function, Optimizer
    model = get_cv_model(model_name, request_body.num_classes,
                         custom_layers=getattr(request_body, 'custom_layers', None))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=request_body.learning_rate)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    num_epochs = request_body.epochs
    
    try:
        for epoch in range(num_epochs):
            # Check for cancellation at the start of each epoch
            if training_cancel_event.is_set():
                write_log("Training cancelled by user", "info", os.path.basename(__file__))
                emit_progress("⚠️ Training cancelled by user", pct=100, final=True,
                              result={"status": "cancelled", "message": "Training was cancelled by the user."})
                raise InterruptedError("Training cancelled by user")

            emit_progress(f"{model_name} — Epoch {epoch+1}/{num_epochs}")
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    dataloader = train_loader
                else:
                    model.eval()   # Set model to evaluate mode
                    dataloader = val_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                emit_progress(f"{model_name} — Epoch {epoch+1}/{num_epochs} [{phase}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        write_log(f'Best val Acc: {best_acc:4f} for {model_name}', "info", os.path.basename(__file__))
        
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, float(best_acc)
        
    except InterruptedError:
        raise  # Let cancellation propagate cleanly
    except Exception as e:
        if request_body.status_code == 200:
            request_body.error_message = "ModelBuildingError"
            request_body.status_code = error_code_mappings.get(request_body.error_message, 500)
            request_body.error_traceback = traceback.format_exc()
            write_log(f"Error: {request_body.error_message} - {request_body.error_traceback}", "error", os.path.basename(__file__))
        raise e
