import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import re

class NewsScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'uk-UA,uk;q=0.9,en-US;q=0.8,en;q=0.7'
        }

    def fetch_news(self, url, selectors):
        try:
            res = requests.get(url, headers=self.headers, timeout=15)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'html.parser')
            
            titles = []
            
            for selector in selectors:
                for item in soup.select(selector):
                    text = item.get_text(strip=True)
                    if 25 < len(text) < 150 and text not in titles:
                        titles.append(text)
            
            if not titles:
                for a_tag in soup.find_all('a'):
                    text = a_tag.get_text(strip=True)
                    if 25 < len(text) < 150 and text not in titles:
                        titles.append(text)
            
            print(f"Success: Found {len(titles)} titles on {url}")
            return titles
            
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return []

    def analyze_content(self, data_dict):
        results = []
        for site, titles in data_dict.items():
            if not titles:
                continue
                
            all_text = " ".join(titles).lower()
            
            words = re.findall(r'\b[а-яіїєґa-z]{4,}\b', all_text)
            
            stop_words = {'щодо', 'після', 'через', 'який', 'буде', 'цього', 'тому', 'яка', 'яких', 'коли', 'якщо', 'його'}
            words = [w for w in words if w not in stop_words]
            
            top_words = Counter(words).most_common(5)
            
            results.append({
                'Site': site,
                'Total Articles': len(titles),
                'Avg Title Length (chars)': round(sum(len(t) for t in titles) / len(titles), 1),
                'Top Keywords': ", ".join([w[0] for w in top_words])
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    scraper = NewsScraper()
    
    sites = {
        "pravda": (
            "https://www.pravda.com.ua/", 
            [".article_header a", ".article__title a"]
        ),
        "korrespondent": (
            "https://korrespondent.net/", 
            [".article__title a", ".time-articles a"]
        ),
        "vechirniy_kyiv": (
            "https://vechirniy.kyiv.ua/", 
            [".news-title a", ".title a"]
        )
    }
    
    scraped_data = {}
    
    for name, (url, selectors) in sites.items():
        print(f"Scraping {name}...")
        news = scraper.fetch_news(url, selectors)
        scraped_data[name] = news
        
        if news:
            df = pd.DataFrame(news, columns=['Title'])
            df.to_csv(f"{name}_news.csv", index=False, encoding='utf-8-sig')
        
    analysis_df = scraper.analyze_content(scraped_data)
    
    if not analysis_df.empty:
        print("\n--- Comparative Analysis ---")
        print(analysis_df.to_string(index=False))
        analysis_df.to_csv("comparative_analysis.csv", index=False, encoding='utf-8-sig')
    else:
        print("\nNo data to analyze.")