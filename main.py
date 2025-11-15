from weather.weather_analyzer import WeatherAnalyzer

def display_menu():
    """Меню выбора"""
    print("Анализ погоды")
    print("1 - Задание 1: Локации с экстремальными температурами")
    print("2 - Задание 2: Штаты с разбросом температур")
    print("3 - Задание 3: Самый ветренный штат")
    print("4 - Задание 4: Корреляция ветра и осадков")
    print("5 - Сравнение скорости CSV vs Parquet")
    print("0 - Выход")

def main():
    
    analyzer = WeatherAnalyzer('weather.csv')
    
    while True:
        display_menu()
        choice = input("Выберите задание (0-5): ").strip()
        
        if choice == '0':
            print("Выход")
            break
            
        elif choice == '1':
            try:
                result = analyzer.task1()
            except Exception as e:
                print(f"{e}")
                
        elif choice == '2':
            try:
                result = analyzer.task2()
            except Exception as e:
                print(f"{e}")
                
        elif choice == '3':
            try:
                result = analyzer.task3()
            except Exception as e:
                print(f"{e}")
                
        elif choice == '4':
            try:
                result = analyzer.task4()
            except Exception as e:
                print(f"{e}")
                
        elif choice == '5':
            try:
                analyzer.convert_to_parquet()
                analyzer.compare_read_speed()
            except Exception as e:
                print(f"{e}")

        else:
            print("Неверный выбор. Попробуйте снова.")
        
        input("\nНажмите Enter для продолжения...")

if __name__ == "__main__":
    main()