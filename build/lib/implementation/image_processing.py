"""
Модуль image_processing.py

Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

Содержит класс ImageProcessing, предоставляющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Кэнни)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (метод пока не реализован)

Модуль предназначен для учебных целей (лабораторная работа по курсу "Технологии программирования на Python").
"""

import cv2

import interfaces

import numpy as np
import time
from functools import wraps


class ImageProcessing(interfaces.IImageProcessing):
    """
    Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

    Предоставляет методы для обработки изображений, включая свёртку, преобразование
    в оттенки серого, гамма-коррекцию, а также обнаружение границ, углов и окружностей.

    Методы:
        _convolution(image, kernel): Выполняет свёртку изображения с ядром.
        _rgb_to_grayscale(image): Преобразует RGB-изображение в оттенки серого.
        _gamma_correction(image, gamma): Применяет гамма-коррекцию.
        edge_detection(image): Обнаруживает границы (Canny).
        corner_detection(image): Обнаруживает углы (Harris).
        circle_detection(image): Обнаруживает окружности (HoughCircles).
    """


    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Использует функцию cv2.filter2D для применения ядра к изображению.

        Args:
            image (np.ndarray): Входное изображение (может быть цветным или чёрно-белым).
            kernel (np.ndarray): Ядро свёртки (матрица).

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        img = image.astype(np.float64)
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="reflect")
        output = np.zeros_like(img, dtype=np.float64)

        H, W = img.shape[:2]
        for y in range(H):
            for x in range(W):
                for c in range(img.shape[2]):
                    region = padded[y:y+kh, x:x+kw, c]
                    output[y, x, c] = np.sum(region * kernel)
        
        return output

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Использует функцию cv2.cvtColor для преобразования цветного изображения
        в чёрно-белое.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.
        """
        return (0.299 * image[:, :, 0] +
                0.587 * image[:, :, 1] +
                0.114 * image[:, :, 2]).astype(np.uint8)

    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Коррекция осуществляется с помощью таблицы преобразования значений пикселей.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гамма-коррекции (>0).

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        inv_gamma = 1.0 / gamma
        norm = image / 255.0
        corrected = np.power(norm, inv_gamma) * 255.0
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def _gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Гауссово окно (2D)."""
        ax = np.linspace(-(size//2), size//2, size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
        return kernel / kernel.sum()


    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Использует оператор Кэнни (cv2.Canny) для выделения границ.
        Предварительно изображение преобразуется в оттенки серого.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """
        start_time = time.time()
        gray = self._rgb_to_grayscale(image)

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        gx = self._convolution(gray[:, :, None], sobel_x)[:, :, 0]
        gy = self._convolution(gray[:, :, None], sobel_y)[:, :, 0]

        magnitude = np.sqrt(gx.astype(float)**2 + gy.astype(float)**2)
        edges = (magnitude > 120).astype(np.uint8) * 255  # порог
        end_time = time.time()
        print(end_time-start_time)
        return edges

    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении.

        Использует алгоритм Харриса (cv2.cornerHarris) для поиска углов.
        Углы выделяются красным цветом на копии исходного изображения.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными углами (красные точки).
        """
        gray = self._rgb_to_grayscale(image).astype(np.float64)
    
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float64)
        sobel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=np.float64)

        Ix = self._convolution(gray[:, :, None], sobel_x)[:, :, 0].astype(float)
        Iy = self._convolution(gray[:, :, None], sobel_y)[:, :, 0].astype(float)

        Ixx, Iyy, Ixy = Ix*Ix, Iy*Iy, Ix*Iy

        g = self._gaussian_kernel(5, sigma=1.0)
        Sxx = self._convolution(Ixx[:, :, None], g)[:, :, 0]
        Syy = self._convolution(Iyy[:, :, None], g)[:, :, 0]
        Sxy = self._convolution(Ixy[:, :, None], g)[:, :, 0]

        det = Sxx*Syy - Sxy**2
        trace = Sxx + Syy
        R = det - 0.04*(trace**2)

        R_max = np.zeros_like(R)
        H, W = R.shape
        for y in range(1, H-1):
            for x in range(1, W-1):
                local_patch = R[y-1:y+2, x-1:x+2]
                if R[y, x] == local_patch.max() and R[y, x] > (0.1 * R.max()):
                    R_max[y, x] = 1

        result = image.copy()
        ys, xs = np.where(R_max == 1)
        for y, x in zip(ys, xs):
            if 0 <= y < result.shape[0] and 0 <= x < result.shape[1]:
                result[y, x] = [0, 0, 255]  # RGB (красный)

        return result

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении.

        Использует преобразование Хафа (cv2.HoughCircles) для поиска окружностей.
        Найденные окружности выделяются зелёным цветом, центры — красным.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными окружностями.
        """
        gray = self._rgb_to_grayscale(image)
        edges = self.edge_detection(image)
    
        h, w = gray.shape
        min_r, max_r = 15, 60   # диапазон радиусов 
        accumulator = np.zeros((h, w, max_r), dtype=np.uint64)
    
        edge_points = np.argwhere(edges > 0)
    
        for y, x in edge_points:
            for r in range(min_r, max_r):
                for theta in range(0, 360, 10):  # шаг угла 10°
                    a = int(x - r * np.cos(np.deg2rad(theta)))
                    b = int(y - r * np.sin(np.deg2rad(theta)))
                    if 0 <= a < w and 0 <= b < h:
                        accumulator[b, a, r] += 1
    
        result = image.copy()
        threshold = 0.5 * accumulator.max()
    
        circles = np.argwhere(accumulator > threshold)
    
        for y, x, r in circles:
            
            for angle in range(0, 360, 1):
                cx = int(x + r * np.cos(np.deg2rad(angle)))
                cy = int(y + r * np.sin(np.deg2rad(angle)))
                if 0 <= cx < w and 0 <= cy < h:
                    result[cy, cx] = [0, 255, 0]
            if 0 <= x < w and 0 <= y < h:
                result[y, x] = [255, 0, 0]
    
        return result
    
    def _convolution2(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return cv2.filter2D(image, -1, kernel)

    def _rgb_to_grayscale2(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def _gamma_correction2(self, image: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255
                          for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def edge_detection2(self, image: np.ndarray) -> np.ndarray:
        gray = self._rgb_to_grayscale(image)
        edges = cv2.Canny(gray, 100, 200)
        return edges

    def corner_detection2(self, image: np.ndarray) -> np.ndarray:
        gray = self._rgb_to_grayscale(image)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        result = image.copy()
        result[dst > 0.01 * dst.max()] = [255, 0, 0]
        return result
        
