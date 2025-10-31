import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
import time
from scipy import stats
import os

class WeatherAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.parquet_file = 'weather.parquet'
        
    def inspect_data(self):
        """Детальный анализ структуры данных"""
        print("Детальный анализ данных:")
        
        # Читаем все данные для анализа
        all_data = pd.concat([chunk for chunk in self.csv_reader()])
        
        print(f"Всего строк: {len(all_data)}")
        print(f"Всего штатов: {all_data['Station.State'].nunique()}")
        print(f"Штаты: {sorted(all_data['Station.State'].unique().tolist())}")
        print(f"Города: {sorted(all_data['Station.City'].unique().tolist())}")
        print(f"Даты: {sorted(all_data['Date.Full'].unique().tolist())}")
        print(f"Диапазон дат: {all_data['Date.Full'].min()} - {all_data['Date.Full'].max()}")
        
        return all_data
        
    def csv_reader(self, chunksize=1000):
        """Генератор для чтения CSV файла по частям"""
        for chunk in pd.read_csv(self.csv_file, chunksize=chunksize):
            yield chunk
    
    def data_extractor(self, data_stream, columns):
        """Генератор для извлечения нужных столбцов"""
        for chunk in data_stream:
            # Проверяем, что столбцы существуют
            available_columns = [col for col in columns if col in chunk.columns]
            if available_columns:
                yield chunk[available_columns]
    
    def state_aggregator(self, data_stream):
        """Генератор для агрегации данных по штатам"""
        for chunk in data_stream:
            if 'Station.State' in chunk.columns and 'Data.Temperature.Avg Temp' in chunk.columns:
                aggregated = chunk.groupby('Station.State').agg({
                    'Data.Temperature.Avg Temp': 'mean'
                }).reset_index()
                yield aggregated
    
    def city_aggregator(self, data_stream):
        """Генератор для агрегации данных по городам"""
        for chunk in data_stream:
            if 'Station.City' in chunk.columns and 'Data.Temperature.Avg Temp' in chunk.columns:
                aggregated = chunk.groupby('Station.City').agg({
                    'Data.Temperature.Avg Temp': 'mean',
                    'Station.State': 'first'
                }).reset_index()
                yield aggregated

    # Задание 1: Агрегация данных - средняя температура
    def task1_aggregation(self):
        """Агрегация средней температуры по городам"""
        print("Задание 1: Агрегация средней температуры по городам")
        
        pipeline = self.csv_reader()
        pipeline = self.data_extractor(pipeline, ['Station.City', 'Station.State', 'Data.Temperature.Avg Temp'])
        pipeline = self.city_aggregator(pipeline)
        
        # Собираем все данные
        all_aggregated = []
        for aggregated in pipeline:
            all_aggregated.append(aggregated)
        
        if all_aggregated:
            final_result = pd.concat(all_aggregated)
            # Группируем по городам
            city_temp = final_result.groupby(['Station.City', 'Station.State'])['Data.Temperature.Avg Temp'].mean().sort_values(ascending=False)
            
            # Визуализация
            plt.figure(figsize=(15, 8))
            city_temp.head(20).plot(kind='bar', color='skyblue')
            plt.title('Средняя температура по городам (Топ-20)')
            plt.xlabel('Город, Штат')
            plt.ylabel('Средняя температура (°F)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            
            print("Топ-10 городов по средней температуре:")
            for (city, state), temp in city_temp.head(10).items():
                print(f"{city}, {state}: {temp:.1f}°F")
            
            return city_temp
        return None

    # Задание 2: Дисперсия и доверительный интервал для осадков по конкретному штату
    def task2_confidence_interval(self, state_name=None):
        """Дисперсия и доверительный интервал для осадков по городам выбранного штата"""
        print("\nЗадание 2: Дисперсия и доверительный интервал для осадков")
        
        # Получаем список штатов
        all_data = pd.concat([chunk for chunk in self.csv_reader()])
        available_states = sorted(all_data['Station.State'].unique().tolist())
        
        if state_name is None:
            print("Доступные штаты:", available_states)
            state_name = input("Введите название штата: ").strip()
        
        if state_name not in available_states:
            print(f"Штат '{state_name}' не найден. Доступные штаты: {available_states}")
            return None
        
        pipeline = self.csv_reader()
        pipeline = self.data_extractor(pipeline, ['Station.City', 'Station.State', 'Data.Precipitation'])
        
        # Собираем все данные
        all_data = []
        for chunk in pipeline:
            all_data.append(chunk)
        
        if all_data:
            full_data = pd.concat(all_data)
            # Фильтруем по выбранному штату
            state_data = full_data[full_data['Station.State'] == state_name]
            
            if state_data.empty:
                print(f"Нет данных для штата {state_name}")
                return None
            
            # Группируем по городам выбранного штата
            city_stats = state_data.groupby('Station.City').agg({
                'Data.Precipitation': ['mean', 'std', 'count']
            }).round(4)
            
            # Упрощаем мультииндекс
            city_stats.columns = ['Mean_Precipitation', 'Std_Precipitation', 'Count']
            city_stats = city_stats.reset_index()
            
            # Расчет доверительных интервалов
            confidence = 0.95
            results = []
            for _, row in city_stats.iterrows():
                if row['Count'] > 1 and not pd.isna(row['Std_Precipitation']) and row['Std_Precipitation'] > 0:
                    mean = row['Mean_Precipitation']
                    std = row['Std_Precipitation']
                    n = row['Count']
                    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
                    margin_error = t_value * (std / np.sqrt(n))
                    
                    results.append({
                        'City': row['Station.City'],
                        'Mean': mean,
                        'Std': std,
                        'Margin_Error': margin_error,
                        'CI_Lower': max(0, mean - margin_error),
                        'CI_Upper': mean + margin_error,
                        'Count': n
                    })
            
            if results:
                results_df = pd.DataFrame(results).sort_values('Mean', ascending=False)
                
                # Визуализация
                plt.figure(figsize=(12, 6))
                x_pos = np.arange(len(results_df))
                
                plt.bar(x_pos, results_df['Mean'], yerr=results_df['Margin_Error'], 
                        capsize=5, alpha=0.7, color='lightcoral', label='Среднее ± погрешность')
                
                plt.xlabel('Город')
                plt.ylabel('Среднее количество осадков')
                plt.title(f'Доверительные интервалы для осадков по городам штата {state_name}')
                plt.xticks(x_pos, results_df['City'], rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                plt.show()
                
                print(f"Города штата {state_name} с доверительными интервалами для осадков:")
                for _, row in results_df.iterrows():
                    print(f"{row['City']}: {row['Mean']:.3f} ± {row['Margin_Error']:.3f}")
                
                return results_df
            else:
                print("Недостаточно данных для расчета доверительных интервалов")
                return None
        return None

    # Задание 3: Изменение температуры во времени для конкретного города
    def task3_moving_average(self, city_name=None):
        """Изменение температуры во времени со скользящим средним для выбранного города"""
        print("\nЗадание 3: Изменение температуры во времени и скользящее среднее")
        
        # Получаем список городов
        all_data = pd.concat([chunk for chunk in self.csv_reader()])
        available_cities = sorted(all_data['Station.City'].unique().tolist())
        
        if city_name is None:
            print("Доступные города:", available_cities)
            city_name = input("Введите название города: ").strip()
        
        if city_name not in available_cities:
            print(f"Город '{city_name}' не найден. Доступные города: {available_cities}")
            return None
        
        pipeline = self.csv_reader()
        pipeline = self.data_extractor(pipeline, ['Date.Full', 'Data.Temperature.Avg Temp', 'Station.City', 'Station.State'])
        
        # Собираем все данные
        all_data = []
        for chunk in pipeline:
            all_data.append(chunk)
        
        if all_data:
            full_data = pd.concat(all_data)
            
            # Фильтруем по выбранному городу
            city_data = full_data[full_data['Station.City'] == city_name].copy()
            
            if city_data.empty:
                print(f"Нет данных для города {city_name}")
                return None
            
            city_data['Date.Full'] = pd.to_datetime(city_data['Date.Full'])
            city_data = city_data.sort_values('Date.Full')
            city_data['Moving_Avg'] = city_data['Data.Temperature.Avg Temp'].rolling(window=3, min_periods=1).mean()
            
            # Визуализация
            plt.figure(figsize=(12, 6))
            plt.plot(city_data['Date.Full'], city_data['Data.Temperature.Avg Temp'], 
                    alpha=0.5, label='Ежедневная температура', color='lightblue', marker='o', markersize=6)
            plt.plot(city_data['Date.Full'], city_data['Moving_Avg'], 
                    linewidth=2, label='Скользящее среднее (3 дня)', color='red')
            
            state = city_data['Station.State'].iloc[0]
            plt.title(f'Изменение температуры в {city_name}, {state}')
            plt.xlabel('Дата')
            plt.ylabel('Температура (°F)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            print(f"\nДетальный анализ для {city_name}:")
            print(city_data[['Date.Full', 'Data.Temperature.Avg Temp', 'Moving_Avg']].round(1))
            
            return city_data
        return None

    # Дополнительное задание: работа с Parquet
    def convert_to_parquet(self):
        """Конвертация CSV в Parquet"""
        if not os.path.exists(self.parquet_file):
            print("Конвертация CSV в Parquet...")
            all_data = pd.concat([chunk for chunk in self.csv_reader(chunksize=5000)])
            table = pa.Table.from_pandas(all_data)
            pq.write_table(table, self.parquet_file)
            print("Конвертация завершена!")
    
    def compare_read_speed(self):
        """Сравнение скорости чтения CSV и Parquet"""
        print("\nСравнение скорости чтения:")
        
        # Чтение CSV
        start_time = time.time()
        csv_data = pd.concat([chunk for chunk in self.csv_reader()])
        csv_time = time.time() - start_time
        print(f"CSV чтение: {csv_time:.4f} секунд")
        
        # Чтение Parquet
        if os.path.exists(self.parquet_file):
            start_time = time.time()
            parquet_data = pq.read_table(self.parquet_file).to_pandas()
            parquet_time = time.time() - start_time
            print(f"Parquet чтение: {parquet_time:.4f} секунд")
            
            if parquet_time > 0:
                speed_ratio = csv_time / parquet_time
                print(f"Parquet быстрее в {speed_ratio:.2f} раз")
            else:
                print("Parquet чтение слишком быстрое для точного сравнения")
        else:
            print("Parquet файл не найден")
        
        return csv_time, parquet_time if os.path.exists(self.parquet_file) else 0
    
    def parquet_additional_analysis(self):
        """Дополнительный анализ с использованием Parquet - агрегированные данные"""
        print("\nДополнительный анализ с Parquet:")
        
        if not os.path.exists(self.parquet_file):
            print("Parquet файл не найден, пропускаем анализ")
            return None
            
        # Чтение только нужных столбцов
        columns_to_read = ['Station.State', 'Station.City', 'Data.Temperature.Avg Temp', 'Data.Precipitation']
        table = pq.read_table(self.parquet_file, columns=columns_to_read)
        data = table.to_pandas()
        
        # Агрегируем данные по городам (средние значения)
        aggregated_data = data.groupby(['Station.City', 'Station.State']).agg({
            'Data.Temperature.Avg Temp': 'mean',
            'Data.Precipitation': 'mean'
        }).reset_index()
        
        print(f"Агрегированные данные по {len(aggregated_data)} городам")
        
        # Анализ: температура vs осадки по штатам
        plt.figure(figsize=(14, 8))
        
        # Получаем уникальные штаты
        states = aggregated_data['Station.State'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(states)))
        color_map = {state: color for state, color in zip(states, colors)}
        
        # Создаем scatter plot для каждого штата
        for state in states:
            state_data = aggregated_data[aggregated_data['Station.State'] == state]
            plt.scatter(state_data['Data.Temperature.Avg Temp'], 
                       state_data['Data.Precipitation'], 
                       alpha=0.7, 
                       c=[color_map[state]], 
                       label=state, 
                       s=80,  # Размер точек
                       edgecolors='black',  # Черная обводка
                       linewidth=0.5)
            
            # Добавляем подписи для некоторых точек (крупнейшие города)
            if len(state_data) > 0:
                # Подписываем город с максимальными осадками
                max_precip_city = state_data.loc[state_data['Data.Precipitation'].idxmax()]
                plt.annotate(max_precip_city['Station.City'], 
                            (max_precip_city['Data.Temperature.Avg Temp'], max_precip_city['Data.Precipitation']),
                            xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        plt.xlabel('Средняя температура (°F)', fontsize=12)
        plt.ylabel('Средние осадки', fontsize=12)
        plt.title('Зависимость между средней температурой и осадками по городам', fontsize=14, pad=20)
        
        # Размещаем легенду
        n_states = len(states)
        n_cols = 2 if n_states > 8 else 1
        
        plt.legend(bbox_to_anchor=(1.05, 1), 
                   loc='upper left', 
                   borderaxespad=0.,
                   fontsize=10,
                   ncol=n_cols,
                   title='Штаты')
        
        plt.grid(True, alpha=0.3)
        
        # Добавляем линию тренда
        z = np.polyfit(aggregated_data['Data.Temperature.Avg Temp'], aggregated_data['Data.Precipitation'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(aggregated_data['Data.Temperature.Avg Temp'].min(), 
                             aggregated_data['Data.Temperature.Avg Temp'].max(), 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2, label='Линия тренда')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=n_cols, title='Штаты')
        plt.tight_layout()
        plt.show()
        
        # Статистический анализ
        correlation = aggregated_data['Data.Temperature.Avg Temp'].corr(aggregated_data['Data.Precipitation'])
        print(f"Корреляция между средней температурой и осадками: {correlation:.3f}")
        
        # Анализ выбросов
        print("\nГорода с самыми высокими осадками:")
        top_precip = aggregated_data.nlargest(5, 'Data.Precipitation')[['Station.City', 'Station.State', 'Data.Precipitation']]
        print(top_precip.round(3))
        
        print("\nГорода с самой высокой температурой:")
        top_temp = aggregated_data.nlargest(5, 'Data.Temperature.Avg Temp')[['Station.City', 'Station.State', 'Data.Temperature.Avg Temp']]
        print(top_temp.round(1))
        
        return aggregated_data

    def get_available_states(self):
        """Получить список доступных штатов"""
        all_data = pd.concat([chunk for chunk in self.csv_reader()])
        return sorted(all_data['Station.State'].unique().tolist())
    
    def get_available_cities(self):
        """Получить список доступных городов"""
        all_data = pd.concat([chunk for chunk in self.csv_reader()])
        return sorted(all_data['Station.City'].unique().tolist())