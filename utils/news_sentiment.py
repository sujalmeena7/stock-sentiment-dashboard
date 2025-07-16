from GoogleNews import GoogleNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def fetch_news_with_sentiment(query, limit=20):
    googlenews = GoogleNews(lang='en')
    googlenews.search(query)
    raw_articles = googlenews.result()[:limit]

    analyzer = SentimentIntensityAnalyzer()
    headlines = []

    for article in raw_articles:
        title = article.get('title', '')
        link = article.get('link', '')
        date = article.get('date', '')

        if title:
            sentiment = analyzer.polarity_scores(title)['compound']
            headlines.append({
                "Date": date,
                "Headline": title,
                "Sentiment Score": round(sentiment, 2),
                "Link": link
            })

    return pd.DataFrame(headlines)
