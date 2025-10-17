# dogs/__init__.py
from .dog_image import DogImage, ColorDogImage, GrayscaleDogImage
from .dog_image_processor import DogImageProcessor

__all__ = ['DogImage', 'ColorDogImage', 'GrayscaleDogImage', 'DogImageProcessor']