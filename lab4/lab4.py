import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import bigrams
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans

nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# 1. Веб-скрапінг сайту pravda.com.ua
def scrape_pravda_news(limit=250):
    news_data = []
    seen_links = set()
    current_date = datetime.datetime.now()
    headers = {'User-Agent': 'Mozilla/5.0'}

    while len(news_data) < limit:
        date_str = current_date.strftime('%d%m%Y')
        url = f"https://www.pravda.com.ua/archives/date_{date_str}/"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a')
            
            for a_tag in links:
                if len(news_data) >= limit:
                    break
                    
                link = a_tag.get('href', '')
                title = a_tag.text.strip()
                
                if title and len(title) > 15 and ('/news/' in link or '/articles/' in link or '/columns/' in link):
                    if link in seen_links:
                        continue
                    seen_links.add(link)
                    
                    title_lower = title.lower()
                    category = 'інше'
                    
                    if 'epravda' in link or any(kw in title_lower for kw in ['економік', 'гривн', 'бюджет', 'банк', 'фінанс', 'подат', 'грош', 'тариф', 'експорт', 'імпорт']):
                        category = 'економічна'
                    elif any(kw in link for kw in ['eurointegration', 'politics']) or any(kw in title_lower for kw in ['зеленськ', 'рад', 'міністр', 'закон', 'суд', 'уряд', 'депутат', 'парламент', 'мзс', 'єс', 'нато', 'саміт']):
                        category = 'політична'
                    elif 'life.pravda' in link or any(kw in title_lower for kw in ['соціум', 'люд', 'освіт', 'здоров', 'лікар', 'школ', 'допомог', 'пенсі', 'погод', 'хвороб', 'студент', 'діти']):
                        category = 'соціальна'
                        
                    news_data.append({'text': title, 'category': category})
                    
        except Exception:
            pass
            
        current_date -= datetime.timedelta(days=1)
        
    return pd.DataFrame(news_data)

df = scrape_pravda_news(limit=250)

print("І рівень:\n")

if not df.empty:
    print(f"Загалом зібрано {len(df)} новин.\n")
    print("Розподіл за категоріями:\n", df['category'].value_counts(), "\n")

    # 1. Класифікація (Навчання з учителем)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text'])
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Метрика ефективності (Accuracy): {accuracy * 100:.2f}%\n")
    print("Детальний звіт класифікації:\n", classification_report(y_test, y_pred, zero_division=0))

print("\nІІ рівень:\n")

all_text = " ".join(df['text'].tolist()).lower()
tokens = word_tokenize(all_text)
words = [word for word in tokens if word.isalpha() and len(word) > 2]

# 1. Топ слів за TF-IDF
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = X.sum(axis=0).A1
tfidf_dict = dict(zip(feature_names, tfidf_scores))

print("Топ-10 слів за вагою TF-IDF:")
top_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:10]
for word, score in top_tfidf:
    print(f"{word}: {score:.4f}")

# 2. Біграмний аналіз слів
bigram_list = list(bigrams(words))
bigram_freq = FreqDist(bigram_list)

print("\nТоп-10 найчастіших біграм:")
for b, freq in bigram_freq.most_common(10):
    print(f"{b[0]} {b[1]}: {freq}")

# 3. Розподіл довжини слів
word_lengths = [len(w) for w in words]
plt.figure(figsize=(10, 5))
plt.hist(word_lengths, bins=range(1, max(word_lengths)+2), edgecolor='black', alpha=0.7, color='skyblue')
plt.title('Розподіл довжини слів у заголовках новин')
plt.xlabel('Довжина слова (кількість символів)')
plt.ylabel('Частота (кількість слів)')
plt.grid(axis='y', alpha=0.5)
plt.xticks(range(1, max(word_lengths)+1))
plt.savefig('word_length_distribution.png')
print("\nГрафік збережено: word_length_distribution.png")

print("\nІІІ рівень:\n")

# Кластеризація без учителя
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X)
df['unsupervised_cluster'] = kmeans.labels_

print("Розподіл новин за кластерами:")
print(df['unsupervised_cluster'].value_counts())

# 4. Лексична дисперсія
target_words = ['україна', 'київ', 'війна', 'росія', 'зеленський']
nltk_text = nltk.Text(words)
plt.figure(figsize=(10, 5))
plt.title('Лексична дисперсія ключових слів')
nltk_text.dispersion_plot(target_words)