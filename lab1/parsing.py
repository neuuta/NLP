import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter

# Початок та Конфігурація
DEPTH_DAYS = 7 # Глибина моніторингу
OUTPUT_FILE = "output.csv" 

STOP_WORDS = {
    'і', 'в', 'на', 'з', 'що', 'до', 'як', 'за', 'та', 'про', 'це', 'від', 
    'але', 'чи', 'й', 'у', 'щоб', 'для', 'а', 'не', 'по', 'під', 'над', 
    'через', 'при', 'після', 'вже', 'буде', 'був', 'була', 'були', 'його', 
    'її', 'їх', 'він', 'вона', 'вони', 'ми', 'ви', 'я', 'ти', 'який', 
    'яка', 'яке', 'які', 'новини', 'тсн', 'укрінформ', 'правда', 'року', 'років', 'рф', 'проти'
}

# Текстова нормалізація
def clean_and_tokenize(text):
    text = text.lower()
    words = re.findall(r'\b[а-яіїєґ]+\b', text)
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]

def get_time_slot(hour):
    if 6 <= hour < 12: return 'Ранок'
    elif 12 <= hour < 18: return 'День'
    else: return 'Вечір'

# Точний пошук реального часу на сторінці
def extract_hour(html_block):
    # Спочатку шукаємо в тегах часу
    time_tag = html_block.find(['time', 'span', 'div'], class_=re.compile(r'time|date', re.I))
    if time_tag:
        match = re.search(r'\b([0-2]?[0-9]):[0-5][0-9]\b', time_tag.text)
        if match: return int(match.group(1))
        
    # Резервний пошук часу в тексті блоку
    match = re.search(r'\b([0-2]?[0-9]):[0-5][0-9]\b', html_block.text)
    if match: return int(match.group(1))
    return None

# Генерація часової шкали та скрапінг
def scrape_news(depth_days):
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(1, depth_days + 1)]
    all_news = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for date in dates:
        print(f"[{date.strftime('%Y-%m-%d')}] Моніторинг новин...")

        urls = [
            f"https://www.pravda.com.ua/archives/date_{date.strftime('%d%m%Y')}/",
            f"https://ua.korrespondent.net/all/{date.strftime('%Y-%m-%d')}/"
        ]
        
        seen_texts = set() # Захист від дублікатів
        
        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Шукаємо контейнери з новинами
                quotes = soup.find_all(['div', 'article'], class_=re.compile(r'news|article|item', re.I))
                
                for q in quotes:
                    a_tag = q.find('a')
                    if not a_tag: continue
                    
                    text = a_tag.text.strip()
                    # Фільтруємо лише заголовки новин
                    if 30 < len(text) < 250 and len(text.split()) > 4:
                        clean_text = re.sub(r'\s+', ' ', text)
                        
                        ident = clean_text[:30]
                        if ident in seen_texts: continue
                        seen_texts.add(ident)
                        
                        tokens = clean_and_tokenize(clean_text)
                        
                        if tokens:
                            hour = extract_hour(q)
                            if hour is not None:
                                all_news.append({
                                    'date': date.strftime('%d.%m.%Y'),
                                    'time_slot': get_time_slot(hour),
                                    'tokens': tokens
                                })
            except Exception as e:
                print(f"Помилка скрапінгу {url}: {e}")
                
    return all_news

# Консолідація та структурування аналітичної таблиці
def build_analytics_table(news_data):
    df_raw = pd.DataFrame(news_data)
    results = []
    
    if df_raw.empty:
        return pd.DataFrame()
        
    for (date, slot), group in df_raw.groupby(['date', 'time_slot']):
        all_tokens = []
        for tokens in group['tokens']:
            all_tokens.extend(tokens)
            
        counter = Counter(all_tokens)
        top_5 = counter.most_common(5)
        
        if not top_5:
            continue
            
        terms = [t[0] for t in top_5]
        freqs = [t[1] for t in top_5]
        
        results.append({
            'дата': date,
            'час': slot,
            'топ-5 термінів': ", ".join(terms),
            'частота топ-5 тремінів': ", ".join(map(str, freqs)), 
            'сума топ-5 термінів': sum(freqs)
        })
        
    df_result = pd.DataFrame(results)
    
    if not df_result.empty:
        slot_order = {'Ранок': 1, 'День': 2, 'Вечір': 3}
        df_result['sort_slot'] = df_result['час'].map(slot_order)
        df_result['sort_date'] = pd.to_datetime(df_result['дата'], format='%d.%m.%Y')
        df_result = df_result.sort_values(['sort_date', 'sort_slot']).drop(columns=['sort_slot', 'sort_date'])
    
    return df_result

if __name__ == '__main__':
    print("--- Початок скрапінгу ---")
    news = scrape_news(DEPTH_DAYS)
    df_final = build_analytics_table(news)
    
    if not df_final.empty:
        df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"--- Дані збережено у файл {OUTPUT_FILE} ---")
    else:
        print("--- Дані не зібрано. ---")