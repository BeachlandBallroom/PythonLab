"""
Точка входа для запуска библиотеки как модуля
python -m dogs --limit 3
"""
import sys
import asyncio
import argparse

from . import DogImageProcessor, setup_logger


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    setup_logger()
    processor = DogImageProcessor()
    await processor.run_async(limit=args.limit)


if __name__ == "__main__":
    asyncio.run(main())