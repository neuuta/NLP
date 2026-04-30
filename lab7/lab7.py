import requests
from bs4 import BeautifulSoup
import speech_recognition as sr
from gtts import gTTS
import pygame
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

class NewsAudioBot:
    def __init__(self):
        print("[Система] Ініціалізація нейромережевих моделей...")
        self.qa_model = pipeline(model="timpal0l/mdeberta-v3-base-squad2", framework="pt")
        
        self.news_data = []
        self.recognizer = sr.Recognizer()

    def parse_news(self):
        """Парсинг новин (web-scraping) - збираємо понад 50 записів"""
        print("[Система] Починаю парсинг новин...")
        url = "https://rss.unian.net/site/news_ukr.rss"
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, features="xml")
            items = soup.findAll('item')
            
            for item in items[:60]:
                title = item.title.text
                description = item.description.text
                full_text = f"{title}. {description}"
                self.news_data.append(full_text)
                
            print(f"[Система] Успішно зібрано {len(self.news_data)} новин.")
        except Exception as e:
            print(f"[Помилка] Не вдалося спарсити новини: {e}")

    def find_relevant_context(self, question):
        """Знаходить найбільш релевантну новину до запитання за допомогою TF-IDF та косинусної подібності"""
        if not self.news_data:
            return ""
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.news_data + [question])
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        best_match_idx = cosine_similarities.argmax()
        return self.news_data[best_match_idx]

    def generate_answer(self, question):
        """Використовує нейромережу для генерації/витягування відповіді з контексту"""
        context = self.find_relevant_context(question)
        
        if not context:
            return "Вибачте, я не знайшов відповідних новин у своїй базі."

        try:
            result = self.qa_model(question=question, context=context)
            if result['score'] < 0.05:
                return f"Ось що я знайшов по цій темі: {context}"
            
            return f"За моїми даними: {result['answer']}."
        except Exception as e:
            return f"Знайдена новина: {context}"

    def text_to_speech(self, text):
        """Генерація аудіо з тексту (Text-to-Speech)"""
        print(f"[Бот]: {text}")
        tts = gTTS(text=text, lang='uk')
        filename = "answer.mp3"
        tts.save(filename)
        
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()

    def listen_to_user(self):
        """Запис аудіо з мікрофона та переведення в текст (Speech-to-Text)"""
        with sr.Microphone() as source:
            print("\n[Система] Слухаю вас... (задайте питання про новини, або скажіть 'вихід')")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = self.recognizer.listen(source)

        try:
            print("[Система] Розпізнаю текст...")
            text = self.recognizer.recognize_google(audio, language="uk-UA")
            print(f"[Ви сказали]: {text}")
            return text.lower()
        except sr.UnknownValueError:
            print("[Помилка] Не вдалося розпізнати мову.")
            return ""
        except sr.RequestError:
            print("[Помилка] Проблема з сервісом розпізнавання (можливо немає інтернету).")
            return ""

    def run(self):
        """Головний цикл бота"""
        self.parse_news()
        
        greeting = "Привіт! Я ваш голосовий помічник. Я прочитав останні новини. Що вас цікавить?"
        self.text_to_speech(greeting)

        while True:
            user_text = self.listen_to_user()
            
            if not user_text:
                continue
                
            if "вихід" in user_text or "стоп" in user_text:
                self.text_to_speech("До побачення! Гарного дня.")
                break
                
            answer = self.generate_answer(user_text)
            self.text_to_speech(answer)

if __name__ == "__main__":
    bot = NewsAudioBot()
    bot.run()