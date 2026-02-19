import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Business Review Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Sora', sans-serif;
}

.main {
    background: #0f1117;
    color: #e8eaf0;
}

.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%);
}

.metric-card {
    background: linear-gradient(135deg, #1e2235, #252a40);
    border: 1px solid #2e3354;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    margin: 6px 0;
}

.metric-card h2 {
    font-size: 2.2rem;
    margin: 0;
    font-family: 'Sora', sans-serif;
}

.metric-card p {
    margin: 4px 0 0 0;
    color: #8b92b0;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.insight-box {
    background: linear-gradient(135deg, #1a2438, #1e2d3d);
    border-left: 4px solid #4f8ef7;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 10px 0;
    color: #c8d8f0;
}

.insight-box.warning {
    border-left-color: #f7a44f;
    background: linear-gradient(135deg, #2a1f12, #2d2318);
}

.insight-box.success {
    border-left-color: #4ff7a0;
    background: linear-gradient(135deg, #122a1f, #182d23);
}

.insight-box.danger {
    border-left-color: #f75f4f;
    background: linear-gradient(135deg, #2a1212, #2d1818);
}

.section-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #c8d0f0;
    border-bottom: 2px solid #2e3354;
    padding-bottom: 8px;
    margin: 28px 0 16px 0;
}

.stDataFrame {
    background: #1a1d2e !important;
}

.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4f8ef7, #a44ff7, #4ff7c8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}

.subtitle {
    color: #6b73a0;
    font-size: 1.05rem;
    margin-top: -8px;
}

.model-winner {
    background: linear-gradient(135deg, #1a2e1a, #1e3520);
    border: 2px solid #4ff7a0;
    border-radius: 12px;
    padding: 14px 20px;
    color: #4ff7a0;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    text-align: center;
}

div[data-testid="stMetric"] {
    background: #1e2235;
    border: 1px solid #2e3354;
    border-radius: 12px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ──────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def assign_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

def assign_sentiment_label(rating):
    if rating >= 4:
        return 1
    elif rating == 3:
        return 0
    else:
        return -1

def get_business_recommendations(pos_pct, neg_pct, neu_pct, top_neg_words, top_pos_words):
    recs = []
    if neg_pct > 40:
        recs.append(("🚨 Critical Alert", f"Negative reviews மிகவும் அதிகமா இருக்கு ({neg_pct:.1f}%)! Immediate action தேவை.", "danger"))
    elif neg_pct > 20:
        recs.append(("⚠️ Improvement Needed", f"{neg_pct:.1f}% negative reviews இருக்கு. Customer experience improve பண்ணணும்.", "warning"))
    else:
        recs.append(("✅ Good Standing", f"Negative reviews கம்மியா இருக்கு ({neg_pct:.1f}%). Keep it up!", "success"))

    if pos_pct > 60:
        recs.append(("💪 Strength", f"Customers உன்னோட product/service-ஐ love பண்றாங்க ({pos_pct:.1f}% positive)!", "success"))

    if top_neg_words:
        recs.append(("🔧 Fix These Issues", f"Customers அதிகமா complaint பண்றது: **{', '.join(top_neg_words[:5])}** — இதை first priority-ஆ fix பண்ணு!", "warning"))

    if top_pos_words:
        recs.append(("⭐ Your USP", f"Customers praise பண்றது: **{', '.join(top_pos_words[:5])}** — இதை marketing-ல highlight பண்ணு!", "success"))

    return recs

def make_dark_chart():
    fig, ax = plt.subplots(facecolor='#1a1d2e')
    ax.set_facecolor('#1a1d2e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2e3354')
    ax.tick_params(colors='#8b92b0')
    ax.xaxis.label.set_color('#8b92b0')
    ax.yaxis.label.set_color('#8b92b0')
    return fig, ax


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Upload Dataset")
    uploaded_file = st.file_uploader("CSV File Upload பண்ணுங்க", type=['csv'])
    st.markdown("---")
    st.markdown("**CSV Format:**")
    st.code("review_text, rating\n'Good product!', 5\n'Bad quality', 1", language="text")
    st.markdown("---")
    st.markdown("**Columns Required:**")
    st.markdown("- `review` or `review_text` or `text`")
    st.markdown("- `rating` or `score` or `stars`")
    st.markdown("---")
    st.markdown("*Built with ❤️ using Streamlit + Sklearn*")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📊 Business Review Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Exploiting Business Insights from Customer Reviews using ML</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ─── Main Logic ───────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Sidebar-ல CSV file upload பண்ணுங்க!")

    st.markdown("### 📋 Sample CSV Format")
    sample_df = pd.DataFrame({
        'review': [
            'The product quality is excellent and delivery was super fast!',
            'Terrible experience. Package arrived broken and customer service ignored me.',
            'Average product. Nothing special but does the job okay.',
            'Absolutely love this! Best purchase I made this year.',
            'Very poor quality. Stopped working after 2 days.'
        ],
        'rating': [5, 1, 3, 5, 1]
    })
    st.dataframe(sample_df, use_container_width=True)

    csv_bytes = sample_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Sample CSV", csv_bytes, "sample_reviews.csv", "text/csv")

else:
    # ── Load & Detect Columns ──
    df_raw = pd.read_csv(uploaded_file)
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    review_col = next((c for c in df_raw.columns if c in ['review','review_text','text','comment','feedback']), None)
    rating_col = next((c for c in df_raw.columns if c in ['rating','score','stars','rate']), None)

    if not review_col or not rating_col:
        st.error(f"❌ Column detect ஆகல! உன் columns: {list(df_raw.columns)}\n\nCSV-ல 'review' and 'rating' column இருக்கணும்!")
        st.stop()

    df = df_raw[[review_col, rating_col]].dropna().copy()
    df.columns = ['review', 'rating']
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna()
    df['rating'] = df['rating'].clip(1, 5)
    df['cleaned'] = df['review'].apply(clean_text)
    df['sentiment'] = df['rating'].apply(assign_sentiment)
    df['label'] = df['rating'].apply(assign_sentiment_label)

    st.success(f"✅ {len(df)} reviews successfully loaded!")

    # ── Overview Metrics ──
    pos = (df['sentiment'] == 'Positive').sum()
    neg = (df['sentiment'] == 'Negative').sum()
    neu = (df['sentiment'] == 'Neutral').sum()
    total = len(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📝 Total Reviews", total)
    col2.metric("😊 Positive", f"{pos} ({pos/total*100:.1f}%)")
    col3.metric("😞 Negative", f"{neg} ({neg/total*100:.1f}%)")
    col4.metric("😐 Neutral", f"{neu} ({neu/total*100:.1f}%)")

    st.markdown("---")

    # ══════════════════════════════════════════════════
    # SECTION 1: ML MODEL COMPARISON
    # ══════════════════════════════════════════════════
    st.markdown('<div class="section-title">🤖 ML Model Comparison</div>', unsafe_allow_html=True)

    with st.spinner("All models training பண்றோம்... சற்று காத்திருங்க! ⏳"):
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))

        # Binary: Positive vs Negative (drop neutral for cleaner model)
        df_model = df[df['sentiment'] != 'Neutral'].copy()
        df_model['bin_label'] = (df_model['sentiment'] == 'Positive').astype(int)

        X = tfidf.fit_transform(df_model['cleaned'])
        y = df_model['bin_label']

        if len(y.unique()) < 2 or len(df_model) < 20:
            st.warning("Dataset மிகவும் small! Minimum 20 reviews வேணும்.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Naive Bayes": MultinomialNB(),
            "Linear SVM": LinearSVC(max_iter=2000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        results = {}
        trained_models = {}
        for name, m in models.items():
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            acc = accuracy_score(y_test, pred)
            results[name] = acc
            trained_models[name] = m

    # Display model results
    best_model_name = max(results, key=results.get)
    best_acc = results[best_model_name]

    cols = st.columns(4)
    colors = ['#4f8ef7', '#a44ff7', '#4ff7c8', '#f7a44f']
    for i, (name, acc) in enumerate(results.items()):
        is_best = name == best_model_name
        badge = " 🏆" if is_best else ""
        cols[i].markdown(f"""
        <div class="metric-card">
            <h2 style="color: {colors[i]}">{acc*100:.1f}%</h2>
            <p>{name}{badge}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="model-winner">
        🏆 Best Model: {best_model_name} — {best_acc*100:.1f}% Accuracy
    </div>
    """, unsafe_allow_html=True)

    # Model Accuracy Bar Chart
    fig, ax = make_dark_chart()
    bars = ax.bar(list(results.keys()), [v*100 for v in results.values()],
                  color=colors, edgecolor='none', width=0.5)
    for bar, val in zip(bars, results.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val*100:.1f}%', ha='center', va='bottom', color='#e8eaf0', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_title('Model Accuracy Comparison', color='#c8d0f0', pad=15, fontsize=13)
    ax.set_ylabel('Accuracy (%)', color='#8b92b0')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ══════════════════════════════════════════════════
    # SECTION 2: SENTIMENT DISTRIBUTION CHARTS
    # ══════════════════════════════════════════════════
    st.markdown('<div class="section-title">📊 Sentiment Distribution</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig, ax = make_dark_chart()
        sentiment_counts = df['sentiment'].value_counts()
        pie_colors = {'Positive': '#4ff7a0', 'Negative': '#f75f4f', 'Neutral': '#f7c44f'}
        pie_clrs = [pie_colors.get(s, '#888') for s in sentiment_counts.index]
        wedges, texts, autotexts = ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            colors=pie_clrs,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': '#e8eaf0'},
            wedgeprops={'edgecolor': '#1a1d2e', 'linewidth': 2}
        )
        for at in autotexts:
            at.set_fontweight('bold')
        ax.set_title('Sentiment Breakdown', color='#c8d0f0', pad=15, fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with chart_col2:
        fig, ax = make_dark_chart()
        rating_counts = df['rating'].value_counts().sort_index()
        bar_colors = ['#f75f4f', '#f7a44f', '#f7c44f', '#a4f74f', '#4ff7a0']
        bars = ax.bar(rating_counts.index, rating_counts.values,
                      color=bar_colors, edgecolor='none', width=0.6)
        for bar, val in zip(bars, rating_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha='center', va='bottom', color='#e8eaf0', fontsize=10)
        ax.set_title('Rating Distribution (1★ - 5★)', color='#c8d0f0', pad=15, fontsize=13)
        ax.set_xlabel('Rating', color='#8b92b0')
        ax.set_ylabel('Count', color='#8b92b0')
        ax.set_xticks([1,2,3,4,5])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ══════════════════════════════════════════════════
    # SECTION 3: WORD CLOUDS
    # ══════════════════════════════════════════════════
    st.markdown('<div class="section-title">☁️ Word Cloud Analysis</div>', unsafe_allow_html=True)

    wc_col1, wc_col2 = st.columns(2)

    pos_text = ' '.join(df[df['sentiment'] == 'Positive']['cleaned'])
    neg_text = ' '.join(df[df['sentiment'] == 'Negative']['cleaned'])

    with wc_col1:
        st.markdown("**😊 Positive Reviews**")
        if pos_text.strip():
            wc = WordCloud(
                width=500, height=300,
                background_color='#1a1d2e',
                colormap='YlGn',
                max_words=80,
                collocations=False
            ).generate(pos_text)
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#1a1d2e')
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig)
            plt.close()

    with wc_col2:
        st.markdown("**😞 Negative Reviews**")
        if neg_text.strip():
            wc = WordCloud(
                width=500, height=300,
                background_color='#1a1d2e',
                colormap='OrRd',
                max_words=80,
                collocations=False
            ).generate(neg_text)
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#1a1d2e')
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # ══════════════════════════════════════════════════
    # SECTION 4: TOP KEYWORDS
    # ══════════════════════════════════════════════════
    st.markdown('<div class="section-title">🔑 Top Keywords (Positive vs Negative)</div>', unsafe_allow_html=True)

    from sklearn.feature_extraction.text import CountVectorizer
    stopwords_extra = ['product', 'item', 'one', 'got', 'get', 'also', 'would', 'really', 'very', 'just']

    def get_top_words(texts, n=10):
        if not texts or all(t.strip() == '' for t in texts):
            return [], []
        cv = CountVectorizer(stop_words='english', max_features=1000)
        try:
            X_cv = cv.fit_transform(texts)
            word_freq = zip(cv.get_feature_names_out(), X_cv.toarray().sum(axis=0))
            sorted_words = sorted(word_freq, key=lambda x: x[1], reverse=True)
            filtered = [(w, c) for w, c in sorted_words if w not in stopwords_extra][:n]
            return [w for w, _ in filtered], [c for _, c in filtered]
        except:
            return [], []

    pos_reviews = df[df['sentiment'] == 'Positive']['cleaned'].tolist()
    neg_reviews = df[df['sentiment'] == 'Negative']['cleaned'].tolist()

    pos_words, pos_counts = get_top_words(pos_reviews)
    neg_words, neg_counts = get_top_words(neg_reviews)

    kw_col1, kw_col2 = st.columns(2)

    with kw_col1:
        if pos_words:
            fig, ax = make_dark_chart()
            y_pos = range(len(pos_words))
            bars = ax.barh(y_pos, pos_counts, color='#4ff7a0', edgecolor='none')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pos_words, color='#c8d0f0')
            ax.set_title('Top Positive Keywords', color='#c8d0f0', fontsize=12)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with kw_col2:
        if neg_words:
            fig, ax = make_dark_chart()
            y_pos = range(len(neg_words))
            bars = ax.barh(y_pos, neg_counts, color='#f75f4f', edgecolor='none')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(neg_words, color='#c8d0f0')
            ax.set_title('Top Negative Keywords', color='#c8d0f0', fontsize=12)
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # ══════════════════════════════════════════════════
    # SECTION 5: BUSINESS RECOMMENDATIONS
    # ══════════════════════════════════════════════════
    st.markdown('<div class="section-title">💼 Business Insights & Recommendations</div>', unsafe_allow_html=True)

    pos_pct = pos / total * 100
    neg_pct = neg / total * 100
    neu_pct = neu / total * 100

    recommendations = get_business_recommendations(pos_pct, neg_pct, neu_pct, neg_words[:5], pos_words[:5])

    for title, msg, style in recommendations:
        st.markdown(f"""
        <div class="insight-box {style}">
            <strong>{title}</strong><br>{msg}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Summary Table
    st.markdown("**📋 Executive Summary**")
    summary_data = {
        "Metric": ["Total Reviews Analyzed", "Positive Sentiment", "Negative Sentiment",
                   "Neutral Sentiment", "Average Rating", "Best ML Model", "Model Accuracy"],
        "Value": [
            str(total),
            f"{pos} ({pos_pct:.1f}%)",
            f"{neg} ({neg_pct:.1f}%)",
            f"{neu} ({neu_pct:.1f}%)",
            f"{df['rating'].mean():.2f} / 5.0",
            best_model_name,
            f"{best_acc*100:.1f}%"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Download Results
    st.markdown("---")
    df_export = df[['review', 'rating', 'sentiment']].copy()
    df_export['predicted_sentiment'] = trained_models[best_model_name].predict(
        tfidf.transform(df_model['cleaned'])
    )
    csv_out = df_export.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Results CSV",
        csv_out,
        "analyzed_reviews.csv",
        "text/csv",
        use_container_width=True
    )
