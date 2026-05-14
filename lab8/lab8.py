import cv2
import google.generativeai as genai
from PIL import Image
from gtts import gTTS
import pygame
import os
import time

API_KEY = "AIzaSyCFP8GRYmvRB-Y_ZpX78YTr_POFerP55Vw"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('models/gemini-2.5-flash')

def capture_image_from_webcam():
    """Функція для захоплення кадру з web-камери."""
    print("Запускаємо web-камеру... Посміхніться!")
    cap = cv2.VideoCapture(0)
    
    time.sleep(1) 
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Помилка: Не вдалося отримати зображення з web-камери.")
        return None
        
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    return pil_image

def play_audio(filename):
    """Функція для відтворення аудіофайлу за допомогою pygame."""
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy(): 
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

def main():
    image = capture_image_from_webcam()
    if image is None:
        return

    print("Зображення отримано. Відправляємо до LLM на аналіз...")

    prompt = """
    Ти — розумний голосовий помічник. Опиши візуальне оточення на цьому фото одним-двома короткими реченнями. 
    Спочатку напиши опис українською мовою. 
    Потім додай рівно один рядок з текстом '---' (три дефіси) як розділювач.
    Після цього напиши цей самий опис англійською мовою.
    """

    try:
        response = model.generate_content([prompt, image])
        response_text = response.text.strip()
        parts = response_text.split('---')
        
        if len(parts) >= 2:
            text_uk = parts[0].strip()
            text_en = parts[1].strip()
            
            print("\nРЕЗУЛЬТАТИ")
            print(f"Українською: {text_uk}")
            print(f"Англійською: {text_en}\n")

            print("Озвучую українською...")
            tts_uk = gTTS(text=text_uk, lang='uk')
            tts_uk.save("temp_uk.mp3")
            play_audio("temp_uk.mp3")
            
            print("Озвучую англійською...")
            tts_en = gTTS(text=text_en, lang='en')
            tts_en.save("temp_en.mp3")
            play_audio("temp_en.mp3")
            
        else:
            print("Модель повернула відповідь у неочікуваному форматі:")
            print(response_text)

    except Exception as e:
        print(f"Сталася помилка під час звернення до LLM або обробки аудіо: {e}")

if __name__ == "__main__":
    main()