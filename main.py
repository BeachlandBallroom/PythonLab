# main.py
import argparse
import os
import sys
import asyncio
import time
from logging_config import setup_logger

def run_lab1():
    from main1 import main as lab1_main
    lab1_main()

async def run_lab2_async(limit: int = 1):
    from dogs.dog_image_processor_async import DogImageProcessorAsync
    
    print("=== АСИНХРОННАЯ ОБРАБОТКА ИЗОБРАЖЕНИЙ СОБАК ===")
    
    start_time = time.time()
    
    # Создаем асинхронный процессор
    processor = DogImageProcessorAsync()
    
    # Выполняем все этапы асинхронно
    await processor.download_images(limit)
    await processor.process_all_images()
    await processor.save_all_images()

    end_time = time.time()
    
    print(f"\n=== ОБРАБОТКА ЗАВЕРШЕНА ===")
    print(f"Обработано {len(processor.images)} изображений собак")
    print(f"Общее время выполнения: {end_time - start_time:.2f} секунд")

def run_lab2_sync(limit: int = 1):
    from dogs.dogs_lib.dog_image_processor import DogImageProcessor
    
    print("Обработка изображений собак через Dog API\n")
    
    start_time = time.time()
    
    # Создаем синхронный процессор (старая версия)
    processor = DogImageProcessor()
    
    # Выполняем все этапы синхронно
    processor.download_images(limit)
    processor.process_all_images()
    processor.save_all_images()

    end_time = time.time()
    
    print(f"Обработано {len(processor.images)} изображений собак")
    print(f"Общее время выполнения (синхронно): {end_time - start_time:.2f} секунд")

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
    
    # Парсер для lab2 (синхронная версия)
    lab2_parser = subparsers.add_parser('lab2', help='Лабораторная работа №2 - обработка изображений собак через Dog API (синхронная версия)')
    lab2_parser.add_argument(
        "limit",
        nargs="?",
        type=int,
        default=1,
        help="Количество изображений для загрузки (по умолчанию: 1)",
    )
    
    # Парсер для lab2-async (асинхронная версия)
    lab2_async_parser = subparsers.add_parser('lab2-async', help='Лабораторная работа №2 - асинхронная обработка изображений собак')
    lab2_async_parser.add_argument(
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
        run_lab2_sync(args.limit)
    
    elif args.command == 'lab2-async':
        asyncio.run(run_lab2_async(args.limit))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()