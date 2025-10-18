# dogs/dog_image_processor.py
import os
import time
import requests
import numpy as np
import cv2
from typing import List
from dotenv import load_dotenv
from .dog_image import ColorDogImage, GrayscaleDogImage

load_dotenv()

def timer_decorator(func):
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        print(f"Метод {func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    return wrapper

class DogImageProcessor:
    def __init__(self, api_url: str = os.getenv("DOG_URL")):
        self._api_url = api_url
        self._api_key = os.getenv("DOG_API_KEY")
        self._images: List = []
    
    @property
    def images(self) -> List:
        return self._images
    
    @timer_decorator
    def download_images(self, limit: int = 1) -> None:
        print(f"Начинаем загрузку {limit} изображений собак...")
        
        params = {
            'limit': limit,
            'has_breeds': 1
        }
        
        headers = {
            'x-api-key': self._api_key
        } if self._api_key else {}
        
        try:
            response = requests.get(self._api_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            for item in data:
                image_url = item['url']
                breed_info = item.get('breeds', [{}])[0]
                breed_name = breed_info.get('name', 'unknown')
                
                img_response = requests.get(image_url)
                img_array = np.frombuffer(img_response.content, np.uint8)
                image_data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if image_data is not None:
                    import random
                    if random.choice([True, False]):
                        dog_image = ColorDogImage(image_data, image_url, breed_name)
                    else:
                        dog_image = GrayscaleDogImage(image_data, image_url, breed_name)
                    
                    self._images.append(dog_image)
                    print(f"Загружено изображение породы: {breed_name}")
                else:
                    print(f"Ошибка декодирования изображения: {image_url}")
                    
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к Dog API: {e}")
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")
    
    @timer_decorator
    def process_all_images(self) -> None:
        print("Начинаем обработку изображений собак")
        
        for i, image in enumerate(self._images):
            print(f"Обработка изображения {i+1}: {image.breed}")
            
            image.process_image("edges_custom")
            image.process_image("edges_library")
            
            if isinstance(image, ColorDogImage):
                image.process_image("grayscale_library")
        
        self._perform_arithmetic_operations()

    @timer_decorator
    def _perform_arithmetic_operations(self) -> None:
        print("Выполнение арифметических операций над изображениями...")
        
        if len(self._images) < 2:
            print("Недостаточно изображений для арифметических операций (нужно минимум 2)")
            return
        
        for i in range(len(self._images) - 1):
            try:
                added_image = self._images[i] + self._images[i + 1]
                added_image._breed = f"added_{self._images[i].breed}_and_{self._images[i + 1].breed}"
                self._images.append(added_image)
                print(f"Создано сложенное изображение: {added_image.breed}")
            except Exception as e:
                print(f"Ошибка при сложении изображений {i} и {i+1}: {e}")
        
        for i in range(len(self._images) - 1):
            try:
                original_images = [img for img in self._images if not img.breed.startswith(('added_', 'subtracted_'))]
                
                if i < len(original_images) - 1:
                    subtracted_image = original_images[i] - original_images[i + 1]
                    subtracted_image._breed = f"subtracted_{original_images[i].breed}_from_{original_images[i + 1].breed}"
                    self._images.append(subtracted_image)
                    print(f"Создано вычтенное изображение: {subtracted_image.breed}")
            except Exception as e:
                print(f"Ошибка при вычитании изображений {i} и {i+1}: {e}")
    
    @timer_decorator
    def save_all_images(self, base_dir: str = "dog_images") -> None:
        print(f"Сохранение изображений собак в директорию: {base_dir}")
        
        os.makedirs(base_dir, exist_ok=True)
        
        for i, image in enumerate(self._images):
            breed_safe = "".join(c if c.isalnum() else "_" for c in image.breed)
            
            if not image.breed.startswith(('added_', 'subtracted_', 'combined_')):
                original_path = os.path.join(base_dir, f"{i+1:02d}_{breed_safe}_original.png")
                cv2.imwrite(original_path, image.image_data)
                print(f"Сохранено оригинальное изображение: {original_path}")
            
            for method, processed_image in image.processed_images.items():
                method_safe = method.replace('_', '')
                processed_path = os.path.join(
                    base_dir, 
                    f"{i+1:02d}_{breed_safe}_{method_safe}.png"
                )
                cv2.imwrite(processed_path, processed_image)
                print(f"Сохранено обработанное изображение: {processed_path}")
            
            if image.breed.startswith(('added_', 'subtracted_', 'combined_')):
                arithmetic_path = os.path.join(base_dir, f"{i+1:02d}_{breed_safe}_arithmetic.png")
                cv2.imwrite(arithmetic_path, image.image_data)
                print(f"Сохранено арифметическое изображение: {arithmetic_path}")