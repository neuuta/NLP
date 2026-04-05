import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp_uk = spacy.load("uk_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

# Каркас тематики 
framework = {
    "uk": {
        "програмування": "програмування розробка код javascript front-end додаток застосунок алгоритм софт",
        "машинобудування": "машинобудування проектування bim моделювання креслення конструкція інженерія механізм",
        "радіоелектроніка": "радіоелектроніка плата маршрутизація сигнал мережа резистор протокол обладнання"
    },
    "en": {
        "programming": "programming development code javascript front-end application algorithm software",
        "mechanical engineering": "mechanical engineering design bim modeling drawing construction machinery mechanism",
        "radio electronics": "radio electronics board routing signal network resistor protocol equipment"
    }
}

def pos_tagging(text, lang='uk'):
    nlp = nlp_uk if lang == 'uk' else nlp_en
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def preprocess_text(text, lang='uk'):
    nlp = nlp_uk if lang == 'uk' else nlp_en
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def classify_topic(text, lang='uk'):
    preprocessed_text = preprocess_text(text, lang)
    topics = list(framework[lang].keys())
    corpus = [framework[lang][topic] for topic in topics]
    corpus.append(preprocessed_text)
    
    # Векторизація
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Порівняння останнього елемента з каркасами
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    max_sim_index = similarities.argmax()
    max_sim_value = similarities[max_sim_index]
    
    # Поріг відсікання для "невідомої" тематики 
    if max_sim_value < 0.10:
        return "невідомий" if lang == 'uk' else "unknown", max_sim_value
    return topics[max_sim_index], max_sim_value

texts_uk = [
    "Розробка сучасних веб-додатків вимагає глибокого розуміння front-end технологій, зокрема JavaScript, а також вміння створювати мобільні застосунки.",
    "Для проектування складних інженерних систем та будівельних конструкцій активно застосовуються технології інформаційного моделювання (BIM).",
    "Налаштування комунікаційного обладнання включає конфігурацію протоколів маршрутизації та аналіз електричних плат мережевих вузлів.",
    "Сьогодні ми розглянемо особливості перекладу популярної японської манги українською мовою."
]

texts_en = [
    "Developing modern web applications requires a deep understanding of front-end technologies, especially JavaScript, as well as the ability to build mobile apps.",
    "For designing complex engineering systems and building structures, building information modeling (BIM) technologies are actively used.",
    "Setting up communication equipment involves configuring routing protocols and analyzing electrical boards of network nodes.",
    "Today we will look at the features of translating a popular Japanese manga."
]

print("POS Tagging (UK):")
print(pos_tagging("Розробка сучасних веб-додатків вимагає розуміння.", 'uk'))

print("\nPOS Tagging (EN):")
print(pos_tagging("Developing modern web applications requires understanding.", 'en'))

print("\nКласифікація текстів (UK):")
for i, txt in enumerate(texts_uk, 1):
    topic, score = classify_topic(txt, 'uk')
    print(f"Текст {i} Тематика: {topic.upper()} (Впевненість: {score:.2f})")

print("\nКласифікація текстів (EN):")
for i, txt in enumerate(texts_en, 1):
    topic, score = classify_topic(txt, 'en')
    print(f"Text {i} Topic: {topic.upper()} (Confidence: {score:.2f})")