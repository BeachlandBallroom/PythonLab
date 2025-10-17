# dogs/dog_image.py
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict
from implementation import ImageProcessing

class DogImage(ABC):
    def __init__(self, image_data: np.ndarray, image_url: str, breed: str):
        self._image_data = image_data
        self._image_url = image_url
        self._breed = breed
        self._processed_images = {}
    
    @property
    def image_data(self) -> np.ndarray:
        return self._image_data
    
    @property
    def image_url(self) -> str:
        return self._image_url
    
    @property
    def breed(self) -> str:
        return self._breed
    
    @property
    def processed_images(self) -> Dict[str, np.ndarray]:
        return self._processed_images
    
    @abstractmethod
    def process_image(self, method: str, **kwargs) -> np.ndarray:
        pass
    
    def __add__(self, other):
        if not isinstance(other, DogImage):
            raise TypeError("Можно складывать только объекты DogImage")
        
        if self.image_data.shape != other.image_data.shape:
            raise ValueError("Изображения должны иметь одинаковый размер")
        
        result_data = cv2.add(self.image_data, other.image_data)
        return type(self)(result_data, f"combined_{self.breed}", "combined")
    
    def __sub__(self, other):
        if not isinstance(other, DogImage):
            raise TypeError("Можно вычитать только объекты DogImage")
        
        if self.image_data.shape != other.image_data.shape:
            raise ValueError("Изображения должны иметь одинаковый размер")
        
        result_data = cv2.subtract(self.image_data, other.image_data)
        return type(self)(result_data, f"subtracted_{self.breed}", "subtracted")
    
    def __str__(self) -> str:
        return f"DogImage(breed={self.breed}, shape={self.image_data.shape}, url={self.image_url})"

class ColorDogImage(DogImage):   
    def __init__(self, image_data: np.ndarray, image_url: str, breed: str):
        super().__init__(image_data, image_url, breed)
        self._processor = ImageProcessing()
    
    def process_image(self, method: str, **kwargs) -> np.ndarray:
        if method == "edges_custom":
            result = self._processor.edge_detection(self.image_data)
        elif method == "edges_library":
            result = self._processor.edge_detection2(self.image_data)
        elif method == "grayscale_custom":
            result = self._processor._rgb_to_grayscale(self.image_data)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method == "grayscale_library":
            result = self._processor._rgb_to_grayscale2(self.image_data)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f"Неизвестный метод обработки: {method}")
        
        self._processed_images[method] = result
        return result

class GrayscaleDogImage(DogImage):
    def __init__(self, image_data: np.ndarray, image_url: str, breed: str):
        processor = ImageProcessing()
        gray_data = processor._rgb_to_grayscale(image_data)
        gray_data_3ch = cv2.cvtColor(gray_data, cv2.COLOR_GRAY2BGR)
        super().__init__(gray_data_3ch, image_url, breed)
        self._processor = processor
    
    def process_image(self, method: str, **kwargs) -> np.ndarray:
        if method == "edges_custom":
            result = self._processor.edge_detection(self.image_data)
        elif method == "edges_library":
            result = self._processor.edge_detection2(self.image_data)
        else:
            raise ValueError(f"Метод {method} не поддерживается для ЧБ изображений")
        
        self._processed_images[method] = result
        return result