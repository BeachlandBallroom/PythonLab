# dogs/__init__.py
from .dog_image import DogImage, ColorDogImage, GrayscaleDogImage
from .dog_image_processor import DogImageProcessor
from .logging_config import setup_logger_from_config, setup_logger

__all__ = [
    "DogImage",
    "ColorDogImage",
    "GrayscaleDogImage",
    "DogImageProcessor",
    "setup_logger_from_config",
    "setup_logger",
]