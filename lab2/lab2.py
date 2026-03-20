import re
import nltk
import spacy
import requests
from bs4 import BeautifulSoup
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize, WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer

# Завантажуємо пакети NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Завантажуємо модель Spacy
nlp = spacy.load("en_core_web_sm")


# Парсинг даних – web-скрапінг з Hugging Face
url = "https://huggingface.co/papers"
print(f"Завантажуємо текст за посиланням: {url}")
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Збираємо текст з абзаців та заголовків 
text_elements = soup.find_all(['p', 'h3', 'a'])
raw_text = " ".join([elem.text.strip() for elem in text_elements if elem.text.strip()])
raw_text = raw_text[:5000]

def save_to_file(filename, content):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

# 1.1. Збереження вхідного тексту
save_to_file("1_input_text.txt", raw_text)

# 1.2. Фільтрація 
filtered_text = re.sub(r'[0-9]+', '', raw_text)
filtered_text = re.sub(r'[^\w\s\.,\-]', '', filtered_text) 
save_to_file("2_filtered.txt", filtered_text)

# 1.3. Нормалізація
normalized_text = " ".join(filtered_text.lower().split())
save_to_file("3_normalized.txt", normalized_text)

# 1.4. Токенізація 
tokens_word = word_tokenize(normalized_text) 
tokens_sent = sent_tokenize(raw_text)        
tokenizer_wp = WordPunctTokenizer()          
tokens_wp = tokenizer_wp.tokenize(normalized_text) 

tokenization_result = f"--- Word Tokenization (перші 50) ---\n{tokens_word[:50]}\n\n--- Sentence Tokenization (перші 5) ---\n{tokens_sent[:5]}\n\n--- WordPunct Tokenization (перші 50) ---\n{tokens_wp[:50]}"
save_to_file("4_tokenized.txt", tokenization_result)

# 1.5. Видалення стоп-слів 
stop_words = nlp.Defaults.stop_words
clean_tokens = [word for word in tokens_word if word.isalpha() and word not in stop_words and len(word) > 2]
save_to_file("5_no_stopwords.txt", " ".join(clean_tokens))

# 1.6. Лематизація 
doc = nlp(" ".join(clean_tokens))
lemmas = [token.lemma_ for token in doc]
save_to_file("6_lemmatized.txt", " ".join(lemmas))

# 1.7. Стемінг
stemmer = SnowballStemmer("english")
stems = [stemmer.stem(word) for word in clean_tokens]
save_to_file("7_stemmed.txt", " ".join(stems))

# 1.8. Топ 10 слів
word_counts = Counter(lemmas)
top_10 = word_counts.most_common(10)

top_10_result = "Топ 10 слів тексту:\n"
for word, count in top_10:
    top_10_result += f"{word}: {count}\n"
save_to_file("8_top_10_words.txt", top_10_result)