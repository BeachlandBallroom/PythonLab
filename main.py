from weather.weather_analyzer import WeatherAnalyzer

def display_menu():
    """Меню выбора"""
    print("\n" + "=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №3: АНАЛИЗ ДАННЫХ О ПОГОДЕ")
    print("=" * 60)
    print("1 - Задание 1: Локации с экстремальными температурами")
    print("2 - Задание 2: Штаты с разбросом температур")
    print("3 - Задание 3: Самый ветренный штат")
    print("4 - Доп. задание: Корреляция ветра и осадков (Parquet)")
    print("5 - Сравнение скорости CSV vs Parquet")
    print("6 - Показать доступные штаты и города")
    print("7 - Выполнить все задания")
    print("0 - Выход")
    print("=" * 60)

def main():
    """Основная функция с интерактивным меню"""
    
    analyzer = WeatherAnalyzer('weather.csv')
    
    while True:
        display_menu()
        choice = input("Выберите задание (0-7): ").strip()
        
        if choice == '0':
            print("Выход из программы.")
            break
            
        elif choice == '1':
            print("\nЗапуск задания 1...")
            try:
                result = analyzer.task1_extreme_temperatures()
                if result is not None:
                    print("✓ Задание 1 выполнено успешно")
                else:
                    print("✗ Задание 1 не выполнено")
            except Exception as e:
                print(f"❌ Ошибка в задании 1: {e}")
                
        elif choice == '2':
            print("\nЗапуск задания 2...")
            try:
                result = analyzer.task2_temperature_variability()
                if result is not None:
                    print("✓ Задание 2 выполнено успешно")
                else:
                    print("✗ Задание 2 не выполнено")
            except Exception as e:
                print(f"❌ Ошибка в задании 2: {e}")
                
        elif choice == '3':
            print("\nЗапуск задания 3...")
            try:
                result = analyzer.task3_windiest_state()
                if result is not None:
                    print("✓ Задание 3 выполнено успешно")
                else:
                    print("✗ Задание 3 не выполнено")
            except Exception as e:
                print(f"❌ Ошибка в задании 3: {e}")
                
        elif choice == '4':
            print("\nЗапуск дополнительного задания...")
            try:
                result = analyzer.task4_wind_precipitation_correlation()
                if result is not None:
                    print("✓ Дополнительное задание выполнено успешно")
                    print(f"✓ Использован Parquet файл для анализа")
                else:
                    print("✗ Дополнительное задание не выполнено")
            except Exception as e:
                print(f"❌ Ошибка в дополнительном задании: {e}")
                
        elif choice == '5':
            print("\nСравнение скорости чтения...")
            try:
                analyzer.convert_to_parquet()
                analyzer.compare_read_speed()
                print("✓ Сравнение скорости выполнено")
            except Exception as e:
                print(f"❌ Ошибка при сравнении скорости: {e}")
                
        elif choice == '6':
            print("\nДоступные данные:")
            states = analyzer.get_available_states()
            cities = analyzer.get_available_cities()
            print(f"Штаты ({len(states)}): {states}")
            print(f"Города ({len(cities)}): {cities}")
            
        elif choice == '7':
            print("\nЗапуск всех заданий...")
            try:
                print("\n--- Задание 1 ---")
                result1 = analyzer.task1_extreme_temperatures()
                
                print("\n--- Задание 2 ---")
                result2 = analyzer.task2_temperature_variability()
                
                print("\n--- Задание 3 ---")
                result3 = analyzer.task3_windiest_state()
                
                print("\n--- Дополнительное задание ---")
                result4 = analyzer.task4_wind_precipitation_correlation()
                
                print("\n--- Сравнение скорости ---")
                analyzer.compare_read_speed()
                
                print("\n" + "=" * 60)
                print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ!")
                print("=" * 60)
                
            except Exception as e:
                print(f"❌ Ошибка при выполнении всех заданий: {e}")
                
        else:
            print("Неверный выбор. Попробуйте снова.")
        
        input("\nНажмите Enter для продолжения...")

if __name__ == "__main__":
    main()