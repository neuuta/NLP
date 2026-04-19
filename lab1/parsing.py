import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import datetime

# ==========================================
# 1. Веб-скрапінг сайту pravda.com.ua (через архіви)
# ==========================================
def scrape_pravda_news(limit=200):
    news_data = []
    seen_links = set()
    
    # Починаємо з сьогоднішньої дати
    current_date = datetime.datetime.now()
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    print("Починаємо збір новин з архівів...")

    # Гортаємо дні назад, поки не зберемо потрібну кількість новин
    while len(news_data) < limit:
        date_str = current_date.strftime('%d%m%Y')
        url = f"https://www.pravda.com.ua/archives/date_{date_str}/"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Шукаємо всі посилання
            links = soup.find_all('a')
            
            for a_tag in links:
                if len(news_data) >= limit:
                    break
                    
                link = a_tag.get('href', '')
                title = a_tag.text.strip()
                
                # Відфільтровуємо порожні посилання та залишаємо лише статті/новини
                if title and len(title) > 15 and ('/news/' in link or '/articles/' in link or '/columns/' in link):
                    if link in seen_links:
                        continue
                    seen_links.add(link)
                    
                    # Евристика: визначаємо категорію (лейбл) на основі URL для "навчання з учителем"
                    category = 'інше'
                    if 'epravda' in link or 'economics' in link or 'економік' in title.lower() or 'finance' in link:
                        category = 'економічна'
                    elif 'politics' in link or 'rada' in link or 'політик' in title.lower():
                        category = 'політична'
                    elif 'life' in link or 'society' in link or 'соціум' in title.lower() or 'health' in link:
                        category = 'соціальна'
                        
                    news_data.append({'text': title, 'category': category})
            
            print(f"Зібрано {len(news_data)} новин (пройдено дату: {current_date.strftime('%d.%m.%Y')})")
            
        except Exception as e:
            print(f"Помилка при завантаженні {url}: {e}")
            
        # Переходимо на попередній день
        current_date -= datetime.timedelta(days=1)
        
    return pd.DataFrame(news_data)

# Збільшимо ліміт до 250, щоб модель мала більше даних для тренування
df = scrape_pravda_news(limit=250)
print(f"\nЗагалом зібрано {len(df)} новин.\n")

if df.empty:
    print("Помилка: Дані не зібрано. Перевірте підключення або структуру сайту.")
else:
    print("Розподіл за категоріями (до збагачення):")
    print(df['category'].value_counts(), "\n")

    # ==========================================
    # 2. "Кластеризація" / Класифікація (Навчання з учителем)
    # ==========================================
    
    # Перевіряємо, чи маємо достатньо різних категорій для навчання (мінімум 2)
    if len(df['category'].unique()) > 1:
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(df['text'])
        y = df['category']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = MultinomialNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # ==========================================
        # 3. Оцінка ефективності (> 60% за методичкою)
        # ==========================================
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Метрика ефективності (Accuracy): {accuracy * 100:.2f}%\n")
        print("Детальний звіт класифікації:")
        print(classification_report(y_test, y_pred, zero_division=0))
    else:
        print("Зібрано новини лише однієї категорії. Для навчання моделі потрібно зібрати новини хоча б з двох різних рубрик.")