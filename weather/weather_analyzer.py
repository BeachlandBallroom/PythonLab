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
        print(f"Штаты: {all_data['Station.State'].unique().tolist()}")
        print(f"Города: {all_data['Station.City'].unique().tolist()}")
        print(f"Даты: {all_data['Date.Full'].unique().tolist()}")
        print(f"Диапазон дат: {all_data['Date.Full'].min()} - {all_data['Date.Full'].max()}")
        
        # Статистика по штатам
        state_stats = all_data.groupby('Station.State').agg({
            'Data.Temperature.Avg Temp': ['count', 'mean', 'min', 'max'],
            'Data.Precipitation': ['mean', 'sum']
        }).round(2)
        
        print("\nСтатистика по штатам:")
        print(state_stats)
        
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
        """Агрегация средней температуры по городам (так как данных по штатам мало)"""
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

    # Задание 2: Дисперсия и доверительный интервал для осадков
    def task2_confidence_interval(self):
        """Дисперсия и доверительный интервал для осадков по городам"""
        print("\nЗадание 2: Дисперсия и доверительный интервал для осадков")
        
        pipeline = self.csv_reader()
        pipeline = self.data_extractor(pipeline, ['Station.City', 'Station.State', 'Data.Precipitation'])
        
        # Собираем все данные
        all_data = []
        for chunk in pipeline:
            all_data.append(chunk)
        
        if all_data:
            full_data = pd.concat(all_data)
            
            # Группируем по городам
            city_stats = full_data.groupby(['Station.City', 'Station.State']).agg({
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
                        'State': row['Station.State'],
                        'Mean': mean,
                        'Std': std,
                        'Margin_Error': margin_error,
                        'CI_Lower': max(0, mean - margin_error),  # Осадки не могут быть отрицательными
                        'CI_Upper': mean + margin_error,
                        'Count': n
                    })
            
            if results:
                results_df = pd.DataFrame(results).sort_values('Mean', ascending=False)
                
                # Визуализация
                plt.figure(figsize=(15, 8))
                x_pos = np.arange(len(results_df))
                
                plt.bar(x_pos, results_df['Mean'], yerr=results_df['Margin_Error'], 
                        capsize=5, alpha=0.7, color='lightcoral', label='Среднее ± погрешность')
                
                plt.xlabel('Город')
                plt.ylabel('Среднее количество осадков')
                plt.title('Доверительные интервалы для осадков по городам')
                plt.xticks(x_pos, [f"{row['City']}\n({row['State']})" for _, row in results_df.iterrows()], rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                plt.show()
                
                print("Города с доверительными интервалами для осадков:")
                for _, row in results_df.iterrows():
                    print(f"{row['City']}, {row['State']}: {row['Mean']:.3f} ± {row['Margin_Error']:.3f}")
                
                return results_df
            else:
                print("Недостаточно данных для расчета доверительных интервалов")
                return None
        return None

    # Задание 3: Изменение температуры во времени и скользящее среднее
    def task3_moving_average(self):
        """Изменение температуры во времени со скользящим средним"""
        print("\nЗадание 3: Изменение температуры во времени и скользящее среднее")
        
        pipeline = self.csv_reader()
        pipeline = self.data_extractor(pipeline, ['Date.Full', 'Data.Temperature.Avg Temp', 'Station.City', 'Station.State'])
        
        # Собираем все данные
        all_data = []
        for chunk in pipeline:
            all_data.append(chunk)
        
        if all_data:
            full_data = pd.concat(all_data)
            
            # Анализируем несколько городов для сравнения
            cities_to_analyze = full_data['Station.City'].value_counts().head(3).index.tolist()
            
            plt.figure(figsize=(15, 10))
            
            for i, city in enumerate(cities_to_analyze, 1):
                city_data = full_data[full_data['Station.City'] == city].copy()
                city_data['Date.Full'] = pd.to_datetime(city_data['Date.Full'])
                city_data = city_data.sort_values('Date.Full')
                city_data['Moving_Avg'] = city_data['Data.Temperature.Avg Temp'].rolling(window=3, min_periods=1).mean()
                
                plt.subplot(2, 2, i)
                plt.plot(city_data['Date.Full'], city_data['Data.Temperature.Avg Temp'], 
                        alpha=0.5, label='Ежедневная температура', color='lightblue', marker='o', markersize=4)
                plt.plot(city_data['Date.Full'], city_data['Moving_Avg'], 
                        linewidth=2, label='Скользящее среднее (3 дня)', color='red')
                
                state = city_data['Station.State'].iloc[0]
                plt.title(f'{city}, {state}')
                plt.xlabel('Дата')
                plt.ylabel('Температура (°F)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            # Детальный анализ для одного города
            main_city = cities_to_analyze[0]
            main_city_data = full_data[full_data['Station.City'] == main_city].copy()
            main_city_data['Date.Full'] = pd.to_datetime(main_city_data['Date.Full'])
            main_city_data = main_city_data.sort_values('Date.Full')
            main_city_data['Moving_Avg'] = main_city_data['Data.Temperature.Avg Temp'].rolling(window=3, min_periods=1).mean()
            
            print(f"\nДетальный анализ для {main_city}:")
            print(main_city_data[['Date.Full', 'Data.Temperature.Avg Temp', 'Moving_Avg']].round(1))
            
            return main_city_data
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
        """Дополнительный анализ с использованием Parquet"""
        print("\nДополнительный анализ с Parquet:")
        
        if not os.path.exists(self.parquet_file):
            print("Parquet файл не найден, пропускаем анализ")
            return None
            
        # Чтение только нужных столбцов
        columns_to_read = ['Station.State', 'Station.City', 'Data.Temperature.Avg Temp', 'Data.Precipitation']
        table = pq.read_table(self.parquet_file, columns=columns_to_read)
        data = table.to_pandas()
        
        # Анализ: температура vs осадки по штатам
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(data['Station.State'].unique())))
        color_map = {state: color for state, color in zip(data['Station.State'].unique(), colors)}
        
        for state in data['Station.State'].unique():
            state_data = data[data['Station.State'] == state]
            plt.scatter(state_data['Data.Temperature.Avg Temp'], state_data['Data.Precipitation'], 
                       alpha=0.6, c=[color_map[state]], label=state, s=50)
        
        plt.colorbar(label='Температура (°F)')
        plt.xlabel('Средняя температура (°F)')
        plt.ylabel('Осадки')
        plt.title('Зависимость между температурой и осадками по штатам')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Корреляция
        correlation = data['Data.Temperature.Avg Temp'].corr(data['Data.Precipitation'])
        print(f"Общая корреляция между температурой и осадками: {correlation:.3f}")
        
        return data

    def run_all_analysis(self):
        """Запуск полного анализа"""
        print("Начало анализа данных о погоде...")
        print("="*60)
        
        # Сначала анализируем структуру данных
        all_data = self.inspect_data()
        
        # Дополнительное задание: работа с Parquet
        print("\n" + "="*60)
        self.convert_to_parquet()
        self.compare_read_speed()
        
        # Основные задания
        print("\n" + "="*60)
        result1 = self.task1_aggregation()
        
        print("\n" + "="*60)
        result2 = self.task2_confidence_interval()
        
        print("\n" + "="*60)
        result3 = self.task3_moving_average()
        
        # Дополнительный анализ с Parquet
        print("\n" + "="*60)
        parquet_result = self.parquet_additional_analysis()
        
        print("\n" + "="*60)
        print("Анализ завершен!")
        return {
            'aggregation': result1,
            'confidence_interval': result2,
            'moving_average': result3,
            'parquet_analysis': parquet_result
        }