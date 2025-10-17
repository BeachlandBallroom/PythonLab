
import argparse
import os
import numpy as np
import cv2
from implementation import ImageProcessing

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обработка изображения с помощью методов ImageProcessing (OpenCV).",
    )
    parser.add_argument(
        "method",
        choices=[
            "edges",
            "corners",
            "circles",
            "rgb",
            "gamma",
            "convolution",
            "edges2",
            "corners2",
            "rgb2",
            "gamma2",
            "convolution2",
        ],
        help="Метод обработки: edges, corners, circles, rgb, gamma, convolution",
    )
    parser.add_argument(
        "input",
        help="Путь к входному изображению",
    )
    parser.add_argument(
        "-o", "--output",
        help="Путь для сохранения результата (по умолчанию: <input>_result.png)",
    )

    args = parser.parse_args()

    # Загрузка изображения
    image = cv2.imread(args.input)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {args.input}")
        return

    processor = ImageProcessing()

    # Выбор метода
    if args.method == "edges":
        result = processor.edge_detection(image)
    elif args.method == "corners":
        result = processor.corner_detection(image)
    elif args.method == "circles":
        result = processor.circle_detection(image)
    elif args.method == "rgb":
        result = processor._rgb_to_grayscale(image)
    elif args.method == "gamma":
        gamma = float(input("Введите значение γ (>0): "))
        result = processor._gamma_correction(image, gamma)
    elif args.method == "convolution":
        size = int(input("Введите размер ядра (например 3): "))
        print("Введите ядро построчно, через пробел:")
        kernel = []
        for _ in range(size):
            row = list(map(float, input().split()))
            if len(row) != size:
                print("Ошибка: каждая строка должна содержать ровно", size, "чисел")
                return
            kernel.append(row)
        kernel = np.array(kernel)
        result = processor._convolution2(image, kernel)
    elif args.method == "edges2":
        result = processor.edge_detection2(image)
    elif args.method == "corners2":
        result = processor.corner_detection2(image)
    elif args.method == "rgb2":
        result = processor._rgb_to_grayscale2(image)
    elif args.method == "gamma2":
        gamma = float(input("Введите значение γ (>0): "))
        result = processor._gamma_correction2(image, gamma)
    elif args.method == "convolution2":
        size = int(input("Введите размер ядра (например 3): "))
        print("Введите ядро построчно, через пробел:")
        kernel = []
        for _ in range(size):
            row = list(map(float, input().split()))
            if len(row) != size:
                print("Ошибка: каждая строка должна содержать ровно", size, "чисел")
                return
            kernel.append(row)
        kernel = np.array(kernel)
        result = processor._convolution2(image, kernel)
    else:
        print("Ошибка: неизвестный метод")
        return

    # Определение пути для сохранения
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_result.png"

    # Сохранение результата
    cv2.imwrite(output_path, result)
    print(f"Результат сохранён в {output_path}")

if __name__ == "__main__":
    main()