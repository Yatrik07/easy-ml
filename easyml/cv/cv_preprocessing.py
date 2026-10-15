import os
import traceback
from collections import namedtuple

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from easyml.config.constants import error_code_mappings
from easyml.cv.cv_config import CV_RequestBody
from easyml.utils.logger import write_log

CVDataLoaders = namedtuple("CVDataLoaders", ["train", "val", "test", "train_eval", "class_names"])

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SPLIT_SEED = 42


class ImageSampleDataset(Dataset):
    """A list of (image path, class index) pairs with its own transform."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with Image.open(path) as img:
            sample = img.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def _is_readable_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _image_files(folder):
    for root, _, files in os.walk(folder):
        for file in sorted(files):
            if file.lower().endswith(IMAGE_EXTENSIONS):
                yield os.path.join(root, file)


def _translate_path_for_docker(path: str) -> str:
    """
    Map a host path to the docker-compose mount (``./test_folder`` -> ``/app/test_folder``)
    when the path does not exist as given. Outside Docker paths are returned unchanged.
    """
    path = path.strip()
    if os.path.exists(path) or not os.path.exists("/.dockerenv"):
        return path

    normalized = path.replace('\\', '/')
    for marker in ['test_folder/', 'artifacts/']:
        idx = normalized.lower().find(marker.lower())
        if idx != -1:
            return '/app/' + normalized[idx:]
    return path


def collect_image_samples(dataset_path: str):
    """
    Resolve the dataset definition into (samples, class_names, skipped_files).

    * One folder: each sub-folder is a class (torchvision ``ImageFolder`` layout).
    * Comma-separated folders: each folder is one class, named after the folder.

    Unreadable or corrupt images are skipped instead of failing the run.
    """
    paths = [_translate_path_for_docker(p) for p in dataset_path.split(',') if p.strip()]
    missing = [p for p in paths if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(f"Dataset folder(s) not found: {', '.join(missing)}")

    if len(paths) == 1:
        root = paths[0]
        class_dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        class_folders = [(name, os.path.join(root, name)) for name in class_dirs]
    else:
        class_folders = [(os.path.basename(os.path.normpath(p)), p) for p in paths]

    class_names, samples, skipped = [], [], []
    for class_name, folder in class_folders:
        if class_name not in class_names:
            class_names.append(class_name)
        class_idx = class_names.index(class_name)
        for file_path in _image_files(folder):
            if _is_readable_image(file_path):
                samples.append((file_path, class_idx))
            else:
                skipped.append(file_path)

    return samples, class_names, skipped


def build_transforms(image_size, augment):
    base = [transforms.Resize(tuple(image_size))]
    augmentation = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)] if augment else []
    tail = [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(base + augmentation + tail), transforms.Compose(base + tail)


def _split(samples, fraction):
    """Stratified split of (path, label) pairs, falling back to a random split for tiny classes."""
    labels = [label for _, label in samples]
    counts = {label: labels.count(label) for label in set(labels)}
    n_held_out = max(1, int(round(len(samples) * fraction)))
    if len(samples) - n_held_out < 1:
        raise ValueError(f"Not enough images ({len(samples)}) to make training, validation and test splits.")
    stratify = labels if min(counts.values()) >= 2 and len(counts) <= n_held_out <= len(samples) - len(counts) else None
    return train_test_split(samples, test_size=n_held_out, random_state=SPLIT_SEED, stratify=stratify)


def cv_preprocessing_controller(request_body: CV_RequestBody) -> "CVDataLoaders":
    """
    Sets up PyTorch datasets and DataLoaders for Image Classification: training, validation
    (early stopping / best epoch) and a held-out test split for the reported metrics.
    """
    write_log("entered cv_preprocessing_controller", "info", os.path.basename(__file__))

    try:
        samples, class_names, skipped = collect_image_samples(request_body.dataset_path)
        if skipped:
            from easyml.utils.socket_emitter import emit_progress
            write_log(f"Skipped {len(skipped)} unreadable image(s), e.g. {skipped[:3]}", "warning", os.path.basename(__file__))
            emit_progress(f"⚠️ Skipped {len(skipped)} unreadable image file(s)", pct=17)

        request_body.classes = class_names
        request_body.num_classes = len(class_names)

        write_log(f"Loaded dataset from {request_body.dataset_path}. Found {len(class_names)} classes, {len(samples)} images.",
                  "info", os.path.basename(__file__))

        if not samples:
            raise ValueError(
                f"No readable images found in: {request_body.dataset_path}. "
                f"Supported formats: {', '.join(IMAGE_EXTENSIONS)}"
            )

        if len(class_names) < 2:
            raise ValueError(
                f"Only {len(class_names)} class(es) found. Image classification requires at least 2 classes: "
                f"either one folder with a sub-folder per class, or several folders separated by commas."
            )

        test_fraction = min(max(request_body.test_split, 0.05), 0.4)
        val_fraction = min(max(request_body.validation_split, 0.05), 0.5)
        # The held-out test split is never used for early stopping or picking the best epoch,
        # so its metrics are an honest estimate (validation accuracy is optimistically biased).
        train_val_samples, test_samples = _split(samples, test_fraction)
        train_samples, val_samples = _split(train_val_samples, val_fraction)

        train_transform, eval_transform = build_transforms(request_body.image_size, request_body.augment)

        def loader(subset, transform, shuffle=False):
            return DataLoader(ImageSampleDataset(subset, transform=transform), batch_size=request_body.batch_size,
                              shuffle=shuffle, num_workers=0)

        loaders = CVDataLoaders(
            train=loader(train_samples, train_transform, shuffle=True),
            val=loader(val_samples, eval_transform),
            test=loader(test_samples, eval_transform),
            # Train-set metrics are measured without augmentation, like the test set.
            train_eval=loader(train_samples, eval_transform),
            class_names=class_names,
        )

        write_log(f"Created DataLoaders: Train size={len(train_samples)}, Val size={len(val_samples)}, "
                  f"Test size={len(test_samples)}", "info", os.path.basename(__file__))

        return loaders

    except Exception as e:
        if request_body.status_code == 200:
            request_body.error_message = "PreProcessingError"
            request_body.status_code = error_code_mappings.get(request_body.error_message, 500)
            request_body.error_traceback = traceback.format_exc()
            write_log(f"Error: {request_body.error_message} - {request_body.error_traceback}", "error", os.path.basename(__file__))
        raise e
