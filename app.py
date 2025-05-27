# app.py

import streamlit as st
import requests
import pandas as pd
import yfinance as yf
from textblob import TextBlob

# —————————————————————————
#              CONFIG
# —————————————————————————
PUSHSHIFT_BASE = "https://api.pushshift.io/reddit/search"
HEADERS = {"User-Agent": "crypto-sentiment-analyzer/1.0"}

CRYPTO_TICKERS = {
    "Bitcoin (BTC)":   "BTC",
    "Ethereum (ETH)":  "ETH",
    "Solana (SOL)":    "SOL",
    "Ripple (XRP)":    "XRP",
    "Tether (USDT)":   "USDT",
    "USD Coin (USDC)": "USDC",
    "Dai (DAI)":       "DAI",
}

# —————————————————————————
#       FETCH REDDIT DATA
# —————————————————————————

def fetch_reddit_data(ticker, subreddits, post_limit):
    results = []
    for kind in ("submission", "comment"):
        for sub in subreddits:
            params = {
                "subreddit":  sub,
                "q":          ticker,
                "size":       post_limit,
                "fields":     ["title","selftext","created_utc"] if kind=="submission" else ["body","created_utc"],
                "sort":       "desc",
                "sort_type":  "created_utc"
            }
            url = f"{PUSHSHIFT_BASE}/{kind}/"
            resp = requests.get(url, params=params, headers=HEADERS)
            resp.raise_for_status()   # this will now succeed over HTTPS
            for item in resp.json().get("data", []):
                text = (item.get("title") or "") + " " + (item.get("selftext") or "") \
                       if kind=="submission" else item.get("body","")
                results.append({
                    "text":        text,
                    "created_utc": item.get("created_utc", 0)
                })
    return results


# —————————————————————————
#       SENTIMENT ANALYSIS
# —————————————————————————
def analyze_sentiments(data):
    rows = []
    for entry in data:
        txt = entry["text"]
        blob = TextBlob(txt)
        rows.append({
            "date": pd.to_datetime(entry["created_utc"], unit="s"),
            "polarity": blob.sentiment.polarity
        })
    df = pd.DataFrame(rows).sort_values("date")
    return df

# —————————————————————————
#        FETCH PRICE DATA
# —————————————————————————
def get_price_df(ticker_full, period, interval):
    df = yf.download(f"{ticker_full}-USD", period=period, interval=interval, progress=False).reset_index()
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    return df[["date", "Close"]].sort_values("date")

# —————————————————————————
#     MERGE SENTIMENT & PRICE
# —————————————————————————
def merge_df(sent_df, price_df):
    sent_df = sent_df.sort_values("date").reset_index(drop=True)
    price_df = price_df.sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(sent_df, price_df, on="date", direction="backward")
    return merged

# —————————————————————————
#             UI
# —————————————————————————
st.set_page_config(layout="wide", page_title="Crypto Sentiment Analyzer")
st.title("🚀 Reddit Crypto Market Sentiment Analyzer")

with st.sidebar:
    crypto_name = st.selectbox("Select Crypto", list(CRYPTO_TICKERS.keys()))
    num_posts   = st.slider("Number of posts/comments to fetch", 50, 500, 150, step=50)
    subs        = st.multiselect("Subreddits", ["CryptoCurrency", "CryptoMarkets", "Bitcoin", "Ethereum"], default=["CryptoCurrency"])
    period      = st.selectbox("Price period", ["1d","7d","1mo","3mo"], index=1)
    interval    = st.selectbox("Price interval", ["5m","15m","1h","1d"], index=2)
    run         = st.button("Run Analysis")

if run:
    st.info("Fetching Reddit…")
    data = fetch_reddit_data(CRYPTO_TICKERS[crypto_name], subs, num_posts)

    if not data:
        st.warning("No data returned from Pushshift.")
        st.stop()

    st.info("Analyzing sentiment…")
    sent_df = analyze_sentiments(data)

    st.info("Fetching price…")
    price_df = get_price_df(CRYPTO_TICKERS[crypto_name], period, interval)

    st.info("Merging…")
    merged = merge_df(sent_df, price_df)

    st.success("Done!")

    # Metrics
    avg_sent = merged["polarity"].mean()
    last_price = merged["Close"].iloc[-1]
    c1, c2 = st.columns(2)
    c1.metric("Avg Sentiment", f"{avg_sent:.3f}")
    c2.metric("Latest Close", f"${last_price:,.2f}")

    # Plot
    st.subheader("Sentiment vs Price")
    chart = merged.set_index("date")[["polarity","Close"]]
    st.line_chart(chart)

    # Recent
    st.subheader("Recent Sentiment Samples")
    st.dataframe(sent_df.tail(10), height=300)

    # Download
    csv = merged.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="merged.csv", mime="text/csv")
else:
    st.write("Configure options in the sidebar and click **Run Analysis**.")
