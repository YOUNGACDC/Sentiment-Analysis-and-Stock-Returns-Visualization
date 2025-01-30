# %% Import Libraries
#pip install newsapi 

from newsapi import NewsApiClient
from transformers import pipeline
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# %% Setup
newsapi = NewsApiClient(api_key="xxx")
sentiment_model = pipeline("sentiment-analysis")
ticker = "NVDA"
company = "Nvidia"

#%% Fetch News relevant to the company we chose and sort by relevance( Calculate Sentiment Scores )
articles = newsapi.get_everything(
    q=company, language="en", sort_by="relevancy", page_size=100
)

#Convert to a pandas DF and make it easeir to understand
news = pd.DataFrame([
    {"headline": a["title"], "date": a["publishedAt"][:10]}
    for a in articles["articles"]
])

# The sentiment analysis model is applied to each headline to calculate a sentiment score:
# Positive sentiment gives a positive score.
# Negative sentiment gives a negative score.
# The sentiment score is adjusted to be positive for "POSITIVE" labels and  negative for "NEGATIVE" labels



news["score"] = news["headline"].apply(
    lambda x: sentiment_model(x)[0]["score"] * 
              (1 if sentiment_model(x)[0]["label"] == "POSITIVE" else -1)
)



# %% Aggregate Sentiment by Date  and calculates the average sentiment score for each day.
daily_sentiment = news.groupby("date")["score"].mean().reset_index()
daily_sentiment.rename(columns={"score": "avg_sentiment_score"}, inplace=True)

# %% Fetch Stock Data
stocks = yf.download(ticker, period="1mo", interval="1d").reset_index()
stocks["date"] = stocks["Date"].dt.strftime("%Y-%m-%d")
stocks["return"] = ((stocks["Close"] - stocks["Open"]) / stocks["Open"] * 100)

# %% Merge Sentiment and Stock Data
data = pd.merge(daily_sentiment, stocks, on="date")

# %% Plot Sentiment and Stock Data

# Creates a figure with two subplots: one for sentiment and one for stock returns.
# sharex=True ensures that both subplots share the same x-axis (date), and hspace=0.3 adds space between the plots.

fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True, 
                         gridspec_kw={"hspace": 0.3})

# Sentiment Scores
axes[0].bar(data["date"], data["avg_sentiment_score"], 
            color=["green" if s > 0 else "red" for s in data["avg_sentiment_score"]])
axes[0].set_title(f"Daily Average Sentiment Score for {company}")
axes[0].set_ylabel("Sentiment Score")
axes[0].tick_params(axis="x", rotation=90)

# Stock Returns
axes[1].plot(data["date"], data["return"], color="blue", marker="o")
axes[1].set_title(f"Daily Stock Returns (%) for {company}")
axes[1].set_ylabel("Return (%)")
axes[1].set_xlabel("Date")
axes[1].tick_params(axis="x", rotation=90)
plt.tight_layout()
plt.show()
