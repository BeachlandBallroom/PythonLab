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
        
    def csv_reader(self, chunksize=1000):
        """Генератор для чтения CSV файла по частям"""
        for chunk in pd.read_csv(self.csv_file, chunksize=chunksize):
            yield chunk
    
    def data_extractor(self, data_stream, columns):
        """Генератор для извлечения нужных столбцов"""
        for chunk in data_stream:
            available_columns = [col for col in columns if col in chunk.columns]
            if available_columns:
                yield chunk[available_columns]

    def task1_extreme_temperatures(self):
        """3 локации с самой высокой и 3 с самой низкой среднегодовой температурой"""
        print("Задание 1: Локации с экстремальными температурами")
        
        pipeline = self.csv_reader()
        pipeline = self.data_extractor(pipeline, ['Station.City', 'Station.State', 'Data.Temperature.Avg Temp'])
        
        all_data = []
        for chunk in pipeline:
            all_data.append(chunk)
        
        if all_data:
            full_data = pd.concat(all_data)
            
            city_temps = full_data.groupby(['Station.City', 'Station.State'])['Data.Temperature.Avg Temp'].agg(['mean', 'count']).reset_index()
            city_temps = city_temps[city_temps['count'] >= 3]
            
            highest_temps = city_temps.nlargest(3, 'mean')
            lowest_temps = city_temps.nsmallest(3, 'mean')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            bars1 = ax1.bar(range(len(highest_temps)), highest_temps['mean'], color='red', alpha=0.7)
            ax1.set_title('3 локации с самой высокой температурой')
            ax1.set_xlabel('Локация')
            ax1.set_ylabel('Средняя температура (°F)')
            ax1.set_xticks(range(len(highest_temps)))
            ax1.set_xticklabels([f"{row['Station.City']}\n{row['Station.State']}" for _, row in highest_temps.iterrows()], rotation=45)
            
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}°F', ha='center', va='bottom')
            
            bars2 = ax2.bar(range(len(lowest_temps)), lowest_temps['mean'], color='blue', alpha=0.7)
            ax2.set_title('3 локации с самой низкой температурой')
            ax2.set_xlabel('Локация')
            ax2.set_ylabel('Средняя температура (°F)')
            ax2.set_xticks(range(len(lowest_temps)))
            ax2.set_xticklabels([f"{row['Station.City']}\n{row['Station.State']}" for _, row in lowest_temps.iterrows()], rotation=45)
            
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}°F', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
            print("\n3 локации с самой высокой температурой:")
            for _, row in highest_temps.iterrows():
                print(f"{row['Station.City']}, {row['Station.State']}: {row['mean']:.1f}°F")
            
            print("\n3 локации с самой низкой температурой:")
            for _, row in lowest_temps.iterrows():
                print(f"{row['Station.City']}, {row['Station.State']}: {row['mean']:.1f}°F")
            
            return {'highest': highest_temps, 'lowest': lowest_temps}
        return None

    def task2_temperature_variability(self):
        """3 штата с самым высоким и 3 с самым низким разбросом среднемесячных температур"""
        print("\nЗадание 2: Штаты с наибольшим и наименьшим разбросом температур")
        
        pipeline = self.csv_reader()
        pipeline = self.data_extractor(pipeline, ['Station.State', 'Date.Full', 'Data.Temperature.Avg Temp'])
        
        all_data = []
        for chunk in pipeline:
            all_data.append(chunk)
        
        if all_data:
            full_data = pd.concat(all_data)
            
            full_data['Date.Full'] = pd.to_datetime(full_data['Date.Full'])
            full_data['Month'] = full_data['Date.Full'].dt.month
            
            monthly_temps = full_data.groupby(['Station.State', 'Month'])['Data.Temperature.Avg Temp'].mean().reset_index()
            
            temp_variability = monthly_temps.groupby('Station.State')['Data.Temperature.Avg Temp'].std().reset_index()
            temp_variability.columns = ['Station.State', 'Temperature_Std']
            temp_variability = temp_variability.dropna()
            
            highest_variability = temp_variability.nlargest(3, 'Temperature_Std')
            lowest_variability = temp_variability.nsmallest(3, 'Temperature_Std')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            bars1 = ax1.bar(range(len(highest_variability)), highest_variability['Temperature_Std'], color='orange', alpha=0.7)
            ax1.set_title('3 штата с наибольшим разбросом температур')
            ax1.set_xlabel('Штат')
            ax1.set_ylabel('Стандартное отклонение температуры (°F)')
            ax1.set_xticks(range(len(highest_variability)))
            ax1.set_xticklabels(highest_variability['Station.State'], rotation=45)
            
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}°F', ha='center', va='bottom')
            
            bars2 = ax2.bar(range(len(lowest_variability)), lowest_variability['Temperature_Std'], color='green', alpha=0.7)
            ax2.set_title('3 штата с наименьшим разбросом температур')
            ax2.set_xlabel('Штат')
            ax2.set_ylabel('Стандартное отклонение температуры (°F)')
            ax2.set_xticks(range(len(lowest_variability)))
            ax2.set_xticklabels(lowest_variability['Station.State'], rotation=45)
            
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}°F', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
            print("\n3 штата с наибольшим разбросом температур:")
            for _, row in highest_variability.iterrows():
                print(f"{row['Station.State']}: σ = {row['Temperature_Std']:.2f}°F")
            
            print("\n3 штата с наименьшим разбросом температур:")
            for _, row in lowest_variability.iterrows():
                print(f"{row['Station.State']}: σ = {row['Temperature_Std']:.2f}°F")
            
            return {'highest_variability': highest_variability, 'lowest_variability': lowest_variability}
        return None

    def task3_windiest_state(self):
        """Самый ветренный штат и его скорость ветра"""
        print("\nЗадание 3: Самый ветренный штат")
        
        pipeline = self.csv_reader()
        pipeline = self.data_extractor(pipeline, ['Station.State', 'Data.Wind.Speed'])
        
        all_data = []
        for chunk in pipeline:
            all_data.append(chunk)
        
        if all_data:
            full_data = pd.concat(all_data)
            
            state_winds = full_data.groupby('Station.State')['Data.Wind.Speed'].agg(['mean', 'count']).reset_index()
            state_winds = state_winds[state_winds['count'] >= 3]
            
            windiest_state = state_winds.nlargest(1, 'mean').iloc[0]
            
            top_10_windy = state_winds.nlargest(10, 'mean')
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(top_10_windy)), top_10_windy['mean'], 
                          color=['red' if x == windiest_state['Station.State'] else 'skyblue' for x in top_10_windy['Station.State']],
                          alpha=0.7)
            
            plt.title('Топ-10 самых ветреных штатов')
            plt.xlabel('Штат')
            plt.ylabel('Средняя скорость ветра (м/с)')
            plt.xticks(range(len(top_10_windy)), top_10_windy['Station.State'], rotation=45)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.2f} м/с', ha='center', va='bottom', 
                        fontweight='bold' if top_10_windy.iloc[i]['Station.State'] == windiest_state['Station.State'] else 'normal')
            
            plt.tight_layout()
            plt.show()
            
            print(f"\nСамый ветренный штат: {windiest_state['Station.State']}")
            print(f"Средняя скорость ветра: {windiest_state['mean']:.2f} м/с")
            print(f"Количество наблюдений: {windiest_state['count']}")
            
            return windiest_state
        return None

    def task4_wind_precipitation_correlation(self):
        """Корреляция между скоростью ветра и осадками с использованием Parquet"""
        print("\nДополнительное задание: Корреляция ветра и осадков (с использованием Parquet)")
        
        self.convert_to_parquet()
        
        if not os.path.exists(self.parquet_file):
            print("Parquet файл не найден, невозможно выполнить анализ")
            return None
        
        try:
            columns_to_read = ['Data.Wind.Speed', 'Data.Precipitation', 'Station.State']
            table = pq.read_table(self.parquet_file, columns=columns_to_read)
            full_data = table.to_pandas()
            
            print(f"Загружено {len(full_data)} записей из Parquet файла")
            
            clean_data = full_data.dropna(subset=['Data.Wind.Speed', 'Data.Precipitation'])
            print(f"После очистки: {len(clean_data)} записей")
            
            if len(clean_data) == 0:
                print("Нет данных для анализа корреляции")
                return None
            
            overall_correlation = clean_data['Data.Wind.Speed'].corr(clean_data['Data.Precipitation'])
            
            state_correlations = []
            for state in clean_data['Station.State'].unique():
                state_data = clean_data[clean_data['Station.State'] == state]
                if len(state_data) >= 5:
                    corr = state_data['Data.Wind.Speed'].corr(state_data['Data.Precipitation'])
                    state_correlations.append({
                        'State': state,
                        'Correlation': corr,
                        'Count': len(state_data)
                    })
            
            state_corr_df = pd.DataFrame(state_correlations).sort_values('Correlation', ascending=False)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.scatter(clean_data['Data.Wind.Speed'], clean_data['Data.Precipitation'], alpha=0.5, s=10)
            ax1.set_xlabel('Скорость ветра (м/с)')
            ax1.set_ylabel('Осадки')
            ax1.set_title(f'Корреляция ветра и осадков\nr = {overall_correlation:.3f}')
            ax1.grid(True, alpha=0.3)
            
            z = np.polyfit(clean_data['Data.Wind.Speed'], clean_data['Data.Precipitation'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(clean_data['Data.Wind.Speed'].min(), clean_data['Data.Wind.Speed'].max(), 100)
            ax1.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
            
            top_states = state_corr_df.head(10)
            colors = ['green' if x > 0 else 'red' for x in top_states['Correlation']]
            bars = ax2.bar(range(len(top_states)), top_states['Correlation'], color=colors, alpha=0.7)
            ax2.set_xlabel('Штат')
            ax2.set_ylabel('Корреляция')
            ax2.set_title('Корреляция ветра и осадков по штатам (Топ-10)')
            ax2.set_xticks(range(len(top_states)))
            ax2.set_xticklabels(top_states['State'], rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                        f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
            
            plt.tight_layout()
            plt.show()
            
            print(f"Общая корреляция между скоростью ветра и осадками: {overall_correlation:.3f}")
            
            if abs(overall_correlation) > 0.7:
                strength = "сильная"
            elif abs(overall_correlation) > 0.5:
                strength = "умеренная"
            elif abs(overall_correlation) > 0.3:
                strength = "слабая"
            else:
                strength = "очень слабая"
            
            direction = "положительная" if overall_correlation > 0 else "отрицательная"
            print(f"Характер связи: {strength} {direction} корреляция")
            
            print("\nТоп-5 штатов с самой высокой корреляцией:")
            for _, row in state_corr_df.head().iterrows():
                print(f"{row['State']}: r = {row['Correlation']:.3f} (n={row['Count']})")
            
            print("\nТоп-5 штатов с самой низкой корреляцией:")
            for _, row in state_corr_df.tail().iterrows():
                print(f"{row['State']}: r = {row['Correlation']:.3f} (n={row['Count']})")
            
            return {
                'overall_correlation': overall_correlation,
                'state_correlations': state_corr_df,
                'data_source': 'parquet'
            }
            
        except Exception as e:
            print(f"{e}")
            return None

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
        
        start_time = time.time()
        csv_data = pd.concat([chunk for chunk in self.csv_reader()])
        csv_time = time.time() - start_time
        print(f"CSV чтение: {csv_time:.4f} секунд")
        
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