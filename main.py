import argparse
import os
import sys

def run_lab1():
    from main1 import main as lab1_main
    lab1_main()

def run_lab2(limit: int = 1):
    from dogs.dog_image_processor import DogImageProcessor
    
    print("Обработка изображений собак через Dog API\n")
    
    # Создаем процессор
    processor = DogImageProcessor()
    
    # Выполняем все этапы
    processor.download_images(limit)
    processor.process_all_images()
    processor.save_all_images()

    print(f"Обработано {len(processor.images)} изображений собак")

def main():
    parser = argparse.ArgumentParser(
        description="Лабораторные работы по обработке изображений",
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Парсер для lab1
    lab1_parser = subparsers.add_parser('lab1', help='Лабораторная работа №1 - обработка одного изображения')
    lab1_parser.add_argument(
        "method",
        choices=[
            "edges", "corners", "circles", "rgb", "gamma", "convolution",
            "edges2", "corners2", "rgb2", "gamma2", "convolution2",
        ],
        help="Метод обработки",
    )
    lab1_parser.add_argument(
        "input",
        help="Путь к входному изображению",
    )
    lab1_parser.add_argument(
        "-o", "--output",
        help="Путь для сохранения результата",
    )
    
    # Парсер для lab2
    lab2_parser = subparsers.add_parser('lab2', help='Лабораторная работа №2 - обработка изображений собак через Dog API')
    lab2_parser.add_argument(
        "limit",
        nargs="?",
        type=int,
        default=1,
        help="Количество изображений для загрузки (по умолчанию: 1)",
    )
    
    args = parser.parse_args()
    
    if args.command == 'lab1':
        # Сохраняем аргументы для lab1 и запускаем
        sys.argv = [sys.argv[0], args.method, args.input]
        if args.output:
            sys.argv.extend(['-o', args.output])
        run_lab1()
    
    elif args.command == 'lab2':
        run_lab2(args.limit)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()