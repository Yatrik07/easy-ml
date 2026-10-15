import torch
import torch.nn as nn
import torchvision.models as models
from easyml.cv.cv_config import CV_RequestBody
from easyml.config.constants import error_code_mappings
from easyml.utils.logger import write_log
import os
import traceback
import copy
from easyml.utils.socket_emitter import emit_progress
from easyml.utils.cancellation import check_cancelled, is_cancelled


def get_cv_model(model_name: str, num_classes: int, custom_layers: list = None,
                 image_size=(224, 224), pretrained: bool = True) -> nn.Module:
    """
    Factory function to get PyTorch image classification models.

    ``pretrained`` only matters for the transfer-learning backbones; inference passes False
    because the saved state dict overwrites every weight anyway.
    """
    height, width = int(image_size[0]), int(image_size[1])

    if model_name == "SimpleCNN":
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (height // 4) * (width // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        return model
        
    elif model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
        
    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model

    elif model_name == "CustomCNN":
        if not custom_layers:
            raise ValueError("CustomCNN requires 'custom_layers' specification")
        from easyml.cv.custom_cnn import build_custom_cnn
        return build_custom_cnn(custom_layers, input_channels=3,
                                input_size=(height, width), num_classes=num_classes)
        
    else:
        raise ValueError(f"Unknown CV model: {model_name}")

def train_cv_model(request_body: CV_RequestBody, model_name: str, train_loader, val_loader, base_pct=20, pct_range=70) -> tuple:
    """
    Trains a single PyTorch model with Early Stopping and ReduceLROnPlateau.
    Emits structured epoch_data for live frontend charting.
    """
    write_log(f"Starting training for {model_name}", "info", os.path.basename(__file__))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    write_log(f"Training on device: {device}", "info", os.path.basename(__file__))
    
    # Initialize Model, Loss function, Optimizer
    model = get_cv_model(model_name, request_body.num_classes,
                         custom_layers=getattr(request_body, 'custom_layers', None),
                         image_size=request_body.image_size)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=request_body.learning_rate)
    
    # LR Scheduler: reduce learning rate when val_loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        patience=getattr(request_body, 'reduce_lr_patience', 3),
        factor=getattr(request_body, 'reduce_lr_factor', 0.5),
    )
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_patience = getattr(request_body, 'early_stopping_patience', 5)
    
    num_epochs = request_body.epochs
    
    try:
        for epoch in range(num_epochs):
            # Check for cancellation at the start of each epoch; the job runner reports it to the UI.
            if is_cancelled():
                write_log("Training cancelled by user", "info", os.path.basename(__file__))
                raise InterruptedError("Training cancelled by user")

            epoch_pct = base_pct + int(((epoch + 1) / num_epochs) * pct_range)
            
            epoch_metrics = {"epoch": epoch + 1, "total_epochs": num_epochs, "model_name": model_name}

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    dataloader = train_loader
                else:
                    model.eval()
                    dataloader = val_loader

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloader:
                    check_cancelled()
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = float(running_corrects.double() / len(dataloader.dataset))

                if phase == 'train':
                    epoch_metrics["train_loss"] = round(epoch_loss, 4)
                    epoch_metrics["train_acc"] = round(epoch_acc, 4)
                else:
                    epoch_metrics["val_loss"] = round(epoch_loss, 4)
                    epoch_metrics["val_acc"] = round(epoch_acc, 4)
                    
                    # ReduceLROnPlateau step (based on validation loss)
                    lr_before = optimizer.param_groups[0]['lr']
                    scheduler.step(epoch_loss)
                    if optimizer.param_groups[0]['lr'] < lr_before:
                        write_log(f"{model_name}: reducing learning rate to {optimizer.param_groups[0]['lr']:.6f}",
                                  "info", os.path.basename(__file__))
                    
                    # Early stopping check
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    
                    # Deep copy the model if validation accuracy improves
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            epoch_metrics["learning_rate"] = current_lr
            epoch_metrics["epochs_without_improvement"] = epochs_without_improvement

            # Emit structured progress with epoch_data for live charting
            emit_progress(
                f"{model_name} — Epoch {epoch+1}/{num_epochs} | "
                f"Train: {epoch_metrics['train_loss']:.4f} / {epoch_metrics['train_acc']:.4f} | "
                f"Val: {epoch_metrics['val_loss']:.4f} / {epoch_metrics['val_acc']:.4f} | "
                f"LR: {current_lr:.6f}",
                pct=epoch_pct,
                epoch_data=epoch_metrics
            )

            # Early stopping trigger
            if epochs_without_improvement >= early_stopping_patience:
                write_log(f"Early stopping triggered at epoch {epoch+1} for {model_name}", "info", os.path.basename(__file__))
                emit_progress(
                    f"⏹️ Early stopping: no improvement for {early_stopping_patience} epochs",
                    pct=epoch_pct,
                    epoch_data={"early_stopped": True, "stopped_epoch": epoch + 1, "model_name": model_name}
                )
                break

        write_log(f'Best val Acc: {best_acc:.4f} for {model_name}', "info", os.path.basename(__file__))
        
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


def predict_loader(model: nn.Module, loader, with_proba: bool = False) -> tuple:
    """
    Run ``model`` over a DataLoader and return (true class indices, predicted class indices),
    plus the softmax probabilities as a third item when ``with_proba`` is set.
    """
    device = next(model.parameters()).device
    model.eval()
    y_true, y_pred, probabilities = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            check_cancelled()
            outputs = model(inputs.to(device))
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
            if with_proba:
                probabilities.extend(torch.softmax(outputs, dim=1).cpu().tolist())
    if with_proba:
        return y_true, y_pred, probabilities
    return y_true, y_pred


def classification_metrics(y_true, y_pred, class_names) -> dict:
    """Per-class metrics in the same shape tabular and NLP runs store in metrics_json."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    true_names = [class_names[i] for i in y_true]
    pred_names = [class_names[i] for i in y_pred]
    if not true_names:
        return {}
    return {
        "accuracy": accuracy_score(true_names, pred_names),
        "f1_weighted": f1_score(true_names, pred_names, labels=class_names, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(true_names, pred_names, labels=class_names),
        "confusion_matrix_order": list(class_names),
        "classification_report": classification_report(true_names, pred_names, labels=class_names,
                                                       output_dict=True, zero_division=0),
    }
