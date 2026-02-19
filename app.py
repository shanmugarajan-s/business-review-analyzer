import streamlit as st
import pandas as pd
from textblob import TextBlob

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Business Review Exploitation",
    page_icon="⭐",
    layout="centered"
)

st.title("⭐ Exploiting Business Reviews")
st.write("Simple sentiment analysis for business decision making")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Review Dataset (CSV)",
    type=["csv"]
)

# -------------------------
# SENTIMENT FUNCTION
# -------------------------
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
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

    # Sentiment analysis
    df["Sentiment"] = df["review_text"].astype(str).apply(get_sentiment)

    st.subheader("📊 Sentiment Result")
    st.dataframe(df)

    # Sentiment count
    sentiment_count = df["Sentiment"].value_counts()

    st.subheader("📈 Sentiment Distribution")
    st.bar_chart(sentiment_count)

    # Business insight
    st.subheader("💡 Business Insight")
    if sentiment_count.idxmax() == "Positive":
        st.success("Customers are mostly satisfied 👍")
    elif sentiment_count.idxmax() == "Negative":
        st.error("Customers are unhappy ❌ Immediate action required")
    else:
        st.warning("Customer opinions are mixed ⚠️")

else:
    st.info("👆 Upload a CSV file to start analysis")
