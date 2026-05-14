import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def fetch_news_rss(rss_url):
    """
    Парсинг RSS-стрічки новин за допомогою requests та BeautifulSoup.
    """
    try:
        response = requests.get(rss_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, features="xml")
        articles = soup.findAll("item")
        
        news_data = []
        for a in articles:
            title = a.find("title").text if a.find("title") else ""
            description = a.find("description").text if a.find("description") else ""
            news_data.append(f"{title}. {description}")
        return news_data
    except Exception as e:
        print(f"Помилка при отриманні новин: {e}")
        return []

def categorize_topic(text):
    """
    Визначення соціальної сфери (тематики) на основі ключових слів.
    """
    text_lower = text.lower()
    topics = {
        "Політика": ["politics", "government", "president", "minister", "election", "policy", "war", "parliament"],
        "Економіка": ["economy", "business", "finance", "market", "bank", "trade", "money", "inflation"],
        "Технології": ["tech", "technology", "software", "apple", "google", "ai", "internet", "space"],
        "Охорона здоров'я": ["health", "covid", "virus", "hospital", "doctor", "medicine", "disease"],
        "Спорт": ["sport", "football", "tennis", "champion", "olympics", "match", "tournament"]
    }
    
    for topic, keywords in topics.items():
        if any(kw in text_lower for kw in keywords):
            return topic
    return "Загальні новини / Інше"

def analyze_sentiment(texts):
    """
    Аналіз тональності (настрою) тексту за допомогою NLTK VADER.
    """
    sia = SentimentIntensityAnalyzer()
    results = []
    
    for text in texts:
        score = sia.polarity_scores(text)
        compound = score['compound']
        
        if compound >= 0.05:
            sentiment = "Позитивний"
        elif compound <= -0.05:
            sentiment = "Негативний"
        else:
            sentiment = "Нейтральний"
            
        topic = categorize_topic(text)
        results.append({
            "Текст новин": text, 
            "Соціальна сфера": topic, 
            "Настрій": sentiment, 
            "Оцінка (Compound)": compound
        })
        
    return pd.DataFrame(results)

def main():
    rss_url = "http://feeds.bbci.co.uk/news/world/rss.xml"
    print(f"Отримання новин з глобального інформаційного простору: {rss_url}...")
    
    news_list = fetch_news_rss(rss_url)
    if not news_list:
        print("Не вдалося завантажити новини. Перевірте з'єднання.")
        return
        
    print(f"Успішно зібрано {len(news_list)} новин. Виконується NLP-аналіз...")
    df = analyze_sentiment(news_list)
    
    print("\n--- СТАТИСТИКА ЗА СОЦІАЛЬНИМИ СФЕРАМИ ---")
    print(df['Соціальна сфера'].value_counts().to_string())
    
    print("\n--- СТАТИСТИКА ЗА НАСТРОЯМИ ---")
    print(df['Настрій'].value_counts().to_string())
    
    print("\nГенерація графіків статистики...")
    
    summary = df.groupby(['Соціальна сфера', 'Настрій']).size().unstack(fill_value=0)
    
    colors_map = []
    for col in summary.columns:
        if col == "Позитивний": colors_map.append('mediumseagreen')
        elif col == "Негативний": colors_map.append('indianred')
        else: colors_map.append('lightslategray')

    ax = summary.plot(kind='bar', stacked=True, color=colors_map if colors_map else None, figsize=(10, 6))
    plt.title('Аналіз глобального інфопростору: Сфери та Настрої')
    plt.xlabel('Соціальні сфери')
    plt.ylabel('Кількість новин')
    plt.xticks(rotation=15, ha='right')
    
    for c in ax.containers:
        ax.bar_label(c, label_type='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()