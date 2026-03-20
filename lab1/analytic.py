import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter, defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import os

INPUT_FILE = "output.csv"

# Завантаження даних
print(f"Читання даних з {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
df['дата_dt'] = pd.to_datetime(df['дата'], format='%d.%m.%Y')

# Графічне подання результатів
print("Генерація візуалізацій...")
# Хмара слів (Стовпчик 3 та 4)
# Словник для зберігання точної сумарної частоти кожного слова
word_freq = defaultdict(int)

for index, row in df.iterrows():
    # Розбиваємо рядок зі словами на список
    terms = [t.strip() for t in row['топ-5 термінів'].split(',')]
    # Розбиваємо рядок з частотами на список чисел
    freqs = [int(f.strip()) for f in str(row['частота топ-5 тремінів']).split(',')]
    
    # Додаємо частоти до відповідних слів
    for t, f in zip(terms, freqs):
        word_freq[t] += f

# Генеруємо хмару слів
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Хмара найчастіших термінів')
plt.savefig('wordcloud.png')
plt.close()

# Лінійний графік часового ряду (Стовпчик 5)
daily_sum = df.groupby('дата_dt')['сума топ-5 термінів'].sum().reset_index()
plt.figure(figsize=(10, 5))
plt.plot(daily_sum['дата_dt'], daily_sum['сума топ-5 термінів'], marker='o', label='Сумарна частота', color='blue')
plt.title('Динаміка сумарної частоти термінів')
plt.xlabel('Дата')
plt.ylabel('Частота (Сума)')
plt.grid(True)
plt.legend()
plt.savefig('daily_sum.png')
plt.close()

# Прогнозування тенденцій (Метод найменших квадратів)
def build_forecast(dates, values, title, filename):
    X = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values)
    
    # Навчання моделі
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Прогноз на 7 днів
    X_future = np.arange(len(values), len(values) + 7).reshape(-1, 1)
    y_future = model.predict(X_future)
    future_dates = [dates.iloc[-1] + datetime.timedelta(days=i) for i in range(1, 8)]
    
    # Побудова графіка
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y, marker='o', label='Фактичні дані (Часовий ряд)')
    plt.plot(dates, y_pred, linestyle='--', color='red', label='Лінія тренду')
    plt.plot(future_dates, y_future, marker='x', color='green', label='Прогноз')
    plt.title(f"{title}\nОцінка моделі -> MSE: {mse:.2f}, R²: {r2:.2f}")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return model, mse, r2

# Прогноз для загальної сумарної частоти
print("Розрахунок трендів та прогнозування...")
build_forecast(daily_sum['дата_dt'], daily_sum['сума топ-5 термінів'], 
               'Прогноз загальної частоти термінів', 'forecast_sum.png')

# Пошук топ-3 термінів за весь час
term_totals = defaultdict(int)
for _, row in df.iterrows():
    terms = [t.strip() for t in row['топ-5 термінів'].split(',')]
    freqs = [int(f.strip()) for f in str(row['частота топ-5 тремінів']).split(',')]
    for t, f in zip(terms, freqs):
        term_totals[t] += f

top_3 = sorted(term_totals, key=term_totals.get, reverse=True)[:3]
print(f"Топ-3 найпопулярніші терміни за весь період: {top_3}")

# Формування трьох часових рядів для топ-3 термінів
daily_top3 = []
for date, group in df.groupby('дата_dt'):
    daily_freq = {t: 0 for t in top_3}
    for _, row in group.iterrows():
        terms = [t.strip() for t in row['топ-5 термінів'].split(',')]
        freqs = [int(f.strip()) for f in str(row['частота топ-5 тремінів']).split(',')]
        for t, f in zip(terms, freqs):
            if t in top_3:
                daily_freq[t] += f
    daily_top3.append({'дата': date, **daily_freq})

df_top3 = pd.DataFrame(daily_top3)
# Запис даних в .csv
df_top3.to_csv('top_3_trends.csv', index=False, encoding='utf-8-sig', sep=';')

# Прогнозування для кожного з Топ-3 термінів
for term in top_3:
    build_forecast(df_top3['дата'], df_top3[term], 
                   f'Тренд та прогноз для терміну: "{term.upper()}"', 
                   f'forecast_term_{term}.png')

print("--- Аналіз завершено ---")