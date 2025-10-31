from weather.weather_analyzer import WeatherAnalyzer

def display_menu():
    """Отображение меню выбора"""
    print("\n" + "=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №3: АНАЛИЗ ДАННЫХ О ПОГОДЕ")
    print("=" * 60)
    print("1 - Задание 1: Агрегация средней температуры по городам")
    print("2 - Задание 2: Дисперсия и доверительный интервал для осадков")
    print("3 - Задание 3: Изменение температуры и скользящее среднее")
    print("4 - Дополнительное задание: Анализ с Parquet")
    print("5 - Показать доступные штаты и города")
    print("6 - Выполнить все задания")
    print("0 - Выход")
    print("=" * 60)

def main():
    """Основная функция с интерактивным меню"""
    
    # Создаем экземпляр анализатора
    analyzer = WeatherAnalyzer('weather.csv')
    
    while True:
        display_menu()
        choice = input("Выберите задание (0-6): ").strip()
        
        if choice == '0':
            print("Выход из программы.")
            break
            
        elif choice == '1':
            print("\nЗапуск задания 1...")
            try:
                result = analyzer.task1_aggregation()
                if result is not None:
                    print("✓ Задание 1 выполнено успешно")
                else:
                    print("✗ Задание 1 не выполнено")
            except Exception as e:
                print(f"❌ Ошибка в задании 1: {e}")
                
        elif choice == '2':
            print("\nЗапуск задания 2...")
            try:
                # Показываем доступные штаты
                states = analyzer.get_available_states()
                print(f"Доступные штаты: {states}")
                
                state_name = input("Введите название штата: ").strip()
                result = analyzer.task2_confidence_interval(state_name)
                if result is not None:
                    print("✓ Задание 2 выполнено успешно")
                else:
                    print("✗ Задание 2 не выполнено")
            except Exception as e:
                print(f"❌ Ошибка в задании 2: {e}")
                
        elif choice == '3':
            print("\nЗапуск задания 3...")
            try:
                # Показываем доступные города
                cities = analyzer.get_available_cities()
                print(f"Доступные города: {cities}")
                
                city_name = input("Введите название города: ").strip()
                result = analyzer.task3_moving_average(city_name)
                if result is not None:
                    print("✓ Задание 3 выполнено успешно")
                else:
                    print("✗ Задание 3 не выполнено")
            except Exception as e:
                print(f"❌ Ошибка в задании 3: {e}")
                
        elif choice == '4':
            print("\nЗапуск дополнительного задания...")
            try:
                analyzer.convert_to_parquet()
                analyzer.compare_read_speed()
                result = analyzer.parquet_additional_analysis()
                if result is not None:
                    print("✓ Дополнительное задание выполнено успешно")
                else:
                    print("✗ Дополнительное задание не выполнено")
            except Exception as e:
                print(f"❌ Ошибка в дополнительном задании: {e}")
                
        elif choice == '5':
            print("\nДоступные данные:")
            states = analyzer.get_available_states()
            cities = analyzer.get_available_cities()
            print(f"Штаты ({len(states)}): {states}")
            print(f"Города ({len(cities)}): {cities}")
            
        elif choice == '6':
            print("\nЗапуск всех заданий...")
            try:
                # Задание 1
                print("\n--- Задание 1 ---")
                result1 = analyzer.task1_aggregation()
                
                # Задание 2 с выбором штата
                print("\n--- Задание 2 ---")
                states = analyzer.get_available_states()
                print(f"Доступные штаты: {states}")
                state_name = input("Введите название штата для задания 2: ").strip()
                result2 = analyzer.task2_confidence_interval(state_name)
                
                # Задание 3 с выбором города
                print("\n--- Задание 3 ---")
                cities = analyzer.get_available_cities()
                print(f"Доступные города: {cities}")
                city_name = input("Введите название города для задания 3: ").strip()
                result3 = analyzer.task3_moving_average(city_name)
                
                # Дополнительное задание
                print("\n--- Дополнительное задание ---")
                analyzer.convert_to_parquet()
                analyzer.compare_read_speed()
                result4 = analyzer.parquet_additional_analysis()
                
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