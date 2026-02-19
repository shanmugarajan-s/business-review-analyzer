import streamlit as st
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Business Review Exploitation",
    page_icon="⭐",
    layout="centered"
)

st.title("⭐ Exploiting Business Reviews")
st.write("Enhanced sentiment analysis for business decision making")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Review Dataset (CSV)",
    type=["csv"]
)

# -------------------------
# SENTIMENT FUNCTIONS
# -------------------------
def get_textblob_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def get_vader_sentiment(text, analyzer):
    scores = analyzer.polarity_scores(text)
    if scores["compound"] > 0.05:
        return "Positive"
    elif scores["compound"] < -0.05:
        return "Negative"
    else:
        return "Neutral"

# -------------------------
# MAIN LOGIC
# -------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Ensure review_text column exists
    if "review_text" not in df.columns:
        st.error("CSV must contain a 'review_text' column")
    else:
        # TextBlob sentiment
        df["TextBlob_Sentiment"] = df["review_text"].astype(str).apply(get_textblob_sentiment)
        df["Polarity"] = df["review_text"].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)

        # VADER sentiment
        nltk.download("vader_lexicon")
        analyzer = SentimentIntensityAnalyzer()
        df["VADER_Sentiment"] = df["review_text"].astype(str).apply(lambda x: get_vader_sentiment(x, analyzer))
        df["Compound_Score"] = df["review_text"].astype(str).apply(lambda x: analyzer.polarity_scores(x)["compound"])

        st.subheader("📊 Sentiment Result")
        st.dataframe(df)

        # Sentiment count
        sentiment_count = df["VADER_Sentiment"].value_counts()

        st.subheader("📈 Sentiment Distribution (VADER)")
        st.bar_chart(sentiment_count)

        # Business insight
        st.subheader("💡 Business Insight")
        if sentiment_count.idxmax() == "Positive":
            st.success("Customers are mostly satisfied 👍")
        elif sentiment_count.idxmax() == "Negative":
            st.error("Customers are unhappy ❌ Immediate action required")
        else:
            st.warning("Customer opinions are mixed ⚠️")

        # Word Clouds
        st.subheader("☁️ Word Clouds")
        positive_text = " ".join(df[df["VADER_Sentiment"]=="Positive"]["review_text"])
        negative_text = " ".join(df[df["VADER_Sentiment"]=="Negative"]["review_text"])

        if positive_text:
            wc_pos = WordCloud(width=600, height=400, background_color="white").generate(positive_text)
            st.write("🌟 Positive Reviews")
            st.pyplot(plt.imshow(wc_pos, interpolation="bilinear"))
            plt.axis("off")
            plt.close()

        if negative_text:
            wc_neg = WordCloud(width=600, height=400, background_color="black", colormap="Reds").generate(negative_text)
            st.write("❌ Negative Reviews")
            st.pyplot(plt.imshow(wc_neg, interpolation="bilinear"))
            plt.axis("off")
            plt.close()

        # Trend over time (if date column exists)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            trend = df.groupby(df["date"].dt.to_period("M"))["VADER_Sentiment"].value_counts().unstack().fillna(0)
            st.subheader("📆 Sentiment Trend Over Time")
            st.line_chart(trend)

        # Category analysis (if product column exists)
        if "product" in df.columns:
            st.subheader("📦 Sentiment by Product")
            category_sentiment = df.groupby("product")["VADER_Sentiment"].value_counts().unstack().fillna(0)
            st.bar_chart(category_sentiment)

else:
    st.info("👆 Upload a CSV file to start analysis")
