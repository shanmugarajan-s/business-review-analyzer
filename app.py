import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import numpy as np
import json
from datetime import datetime
import io
import base64

# ==================== ADVANCED NLP LIBRARIES ====================
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Try to import advanced NLP libraries
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    ADVANCED_NLP = True
except ImportError:
    ADVANCED_NLP = False

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Business Conquest Pro",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS STYLING ====================
st.markdown("""
<style>
    .game-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: white;
        min-height: 100vh;
        padding: 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .game-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 15px;
        margin-bottom: 30px;
        border: 3px solid #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
        50% { box-shadow: 0 0 30px rgba(0, 212, 255, 0.6); }
        100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); }
    }
    .academic-section {
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid #4CC9F0;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
    }
    .methodology-card {
        background: linear-gradient(135deg, rgba(76, 201, 240, 0.1), rgba(29, 78, 216, 0.1));
        border: 1px solid #4CC9F0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        transition: transform 0.3s;
    }
    .methodology-card:hover {
        transform: translateY(-5px);
        border-color: #F72585;
    }
    .welcome-container {
        background: linear-gradient(135deg, rgba(106, 17, 203, 0.3), rgba(37, 117, 252, 0.3));
        border: 3px solid #00d4ff;
        border-radius: 20px;
        padding: 40px;
        margin: 50px auto;
        max-width: 900px;
        box-shadow: 0 10px 40px rgba(0, 212, 255, 0.3);
    }
    .level-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        height: 100%;
    }
    .level-card:hover {
        transform: translateY(-10px);
        border-color: #00d4ff;
        box-shadow: 0 10px 20px rgba(0, 212, 255, 0.2);
    }
    .level-badge {
        background: linear-gradient(45deg, #ff0080, #ff8c00);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    .stat-card {
        background: rgba(0, 0, 0, 0.3);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
        margin: 10px 0;
    }
    .battle-arena {
        background: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #ff0080;
        margin: 20px 0;
        animation: glow 3s infinite alternate;
    }
    @keyframes glow {
        from { box-shadow: 0 0 10px rgba(255, 0, 128, 0.3); }
        to { box-shadow: 0 0 20px rgba(255, 0, 128, 0.6); }
    }
    .recommendation-card {
        background: linear-gradient(135deg, rgba(106, 17, 203, 0.2), rgba(37, 117, 252, 0.2));
        border: 2px solid #00d4ff;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
    }
    .action-item {
        background: rgba(0, 212, 255, 0.1);
        border-left: 4px solid #00d4ff;
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
    }
    .progress-bar {
        height: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        margin: 10px 0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #00b09b, #96c93d);
        border-radius: 5px;
        transition: width 0.5s ease;
    }
    .feature-box {
        background: rgba(0, 212, 255, 0.1);
        border: 2px solid #00d4ff;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    .start-button {
        background: linear-gradient(45deg, #6a11cb, #2575fc);
        color: white;
        padding: 15px 40px;
        border: none;
        border-radius: 30px;
        font-size: 1.2rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
    }
    .start-button:hover {
        transform: scale(1.1);
        box-shadow: 0 5px 20px rgba(106, 17, 203, 0.5);
    }
    .download-btn {
        background: linear-gradient(45deg, #00b09b, #96c93d);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        text-decoration: none;
        display: inline-block;
        margin: 5px;
        transition: all 0.3s;
    }
    .download-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0, 176, 155, 0.4);
    }
    .tab-content {
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'player_xp' not in st.session_state:
    st.session_state.player_xp = 0
if 'player_level' not in st.session_state:
    st.session_state.player_level = 1
if 'completed_missions' not in st.session_state:
    st.session_state.completed_missions = []
if 'current_battle' not in st.session_state:
    st.session_state.current_battle = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'advanced_features' not in st.session_state:
    st.session_state.advanced_features = False

# ==================== ENHANCED HELPER FUNCTIONS ====================
def detect_columns(df):
    """Auto-detect relevant columns in uploaded dataset"""
    mapping = {}
    
    # Detect review text column
    text_keywords = ['review', 'comment', 'feedback', 'text', 'description', 'content']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in text_keywords):
            mapping['review'] = col
            break
    
    # Detect rating column
    rating_keywords = ['rating', 'star', 'score', 'rate', 'stars']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in rating_keywords):
            mapping['rating'] = col
            break
    
    # Detect brand/company column
    brand_keywords = ['brand', 'company', 'name', 'product', 'restaurant', 'hotel', 'business', 'store']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in brand_keywords):
            mapping['brand'] = col
            break
    
    # Detect date column
    date_keywords = ['date', 'time', 'timestamp', 'created']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            try:
                pd.to_datetime(df[col])
                mapping['date'] = col
            except:
                pass
    
    return mapping

def advanced_text_preprocessing(text):
    """Enhanced text preprocessing for better analysis"""
    if not ADVANCED_NLP:
        return str(text)
    
    try:
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        text = str(text).lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalnum() and token not in stop_words]
        
        return ' '.join(tokens)
    except:
        return str(text).lower()

def analyze_sentiment(text):
    """Enhanced sentiment analysis with confidence scores"""
    if not NLP_AVAILABLE:
        return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(str(text))

def extract_keywords_advanced(reviews_list, top_n=10):
    """Enhanced keyword extraction using TF-IDF if available"""
    all_text = ' '.join([str(r) for r in reviews_list]).lower()
    
    if ADVANCED_NLP and len(reviews_list) > 5:
        try:
            # Use TF-IDF for better keyword extraction
            vectorizer = TfidfVectorizer(max_features=top_n*2, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(reviews_list)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate average TF-IDF scores
            scores = tfidf_matrix.mean(axis=0).A1
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_scores[:top_n]
        except:
            pass
    
    # Fallback to frequency-based extraction
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has',
                  'this', 'that', 'it', 'phone', 'product', 'very', 'good', 'bad', 'not',
                  'just', 'really', 'would', 'one', 'get', 'my', 'me', 'i'}
    
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    words = [w for w in words if w not in stop_words]
    word_freq = Counter(words)
    return word_freq.most_common(top_n)

def perform_topic_modeling(reviews, n_topics=3):
    """Basic topic modeling using LDA"""
    if not ADVANCED_NLP or len(reviews) < 10:
        return []
    
    try:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf = vectorizer.fit_transform(reviews)
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tfidf)
        
        topics = []
        for idx, topic in enumerate(lda.components_):
            top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]
            topics.append({
                'topic_id': idx,
                'keywords': top_words,
                'weight': topic.sum()
            })
        
        return topics
    except:
        return []

def analyze_brand_performance(df, brand_name, col_mapping):
    """Enhanced brand performance analysis"""
    brand_col = col_mapping['brand']
    review_col = col_mapping['review']
    rating_col = col_mapping['rating']
    
    brand_df = df[df[brand_col].str.contains(brand_name, case=False, na=False)]
    
    if len(brand_df) == 0:
        return None
    
    total_reviews = len(brand_df)
    avg_rating = brand_df[rating_col].mean()
    
    # Enhanced sentiment analysis
    sentiment_scores = []
    positive_count = 0
    negative_count = 0
    
    for review in brand_df[review_col]:
        sentiment = analyze_sentiment(review)
        sentiment_scores.append(sentiment['compound'])
        if sentiment['compound'] > 0.1:
            positive_count += 1
        elif sentiment['compound'] < -0.1:
            negative_count += 1
    
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    positive_pct = (positive_count / total_reviews) * 100 if total_reviews > 0 else 0
    negative_pct = (negative_count / total_reviews) * 100 if total_reviews > 0 else 0
    
    # Extract reviews for analysis
    negative_reviews = brand_df[brand_df[rating_col] <= 2][review_col].tolist()
    positive_reviews = brand_df[brand_df[rating_col] >= 4][review_col].tolist()
    
    complaints = extract_keywords_advanced(negative_reviews, top_n=5) if negative_reviews else []
    strengths = extract_keywords_advanced(positive_reviews, top_n=5) if positive_reviews else []
    
    # Calculate advanced metrics
    review_lengths = brand_df[review_col].apply(lambda x: len(str(x).split()))
    avg_review_length = review_lengths.mean()
    
    # Calculate review frequency if date column exists
    review_trend = []
    if 'date' in col_mapping and col_mapping['date'] in brand_df.columns:
        try:
            brand_df['date_parsed'] = pd.to_datetime(brand_df[col_mapping['date']])
            monthly_reviews = brand_df.groupby(brand_df['date_parsed'].dt.to_period('M')).size()
            review_trend = monthly_reviews.tolist()[-6:]  # Last 6 months
        except:
            pass
    
    return {
        'total_reviews': total_reviews,
        'avg_rating': avg_rating,
        'avg_sentiment': avg_sentiment,
        'positive_pct': positive_pct,
        'negative_pct': negative_pct,
        'neutral_pct': 100 - positive_pct - negative_pct,
        'complaints': complaints,
        'strengths': strengths,
        'customer_satisfaction': min(100, max(0, (avg_sentiment + 1) * 50)),
        'quality_score': min(100, avg_rating * 20) if avg_rating else 0,
        'engagement_score': min(100, avg_review_length * 0.5),  # Normalized
        'review_trend': review_trend,
        'topics': perform_topic_modeling(brand_df[review_col].tolist(), 2) if ADVANCED_NLP else []
    }

def generate_strategic_recommendations(your_analysis, competitor_analysis, your_brand, competitor_brand):
    """Enhanced strategic recommendations with academic frameworks"""
    recommendations = {
        'swot': {'strengths': [], 'weaknesses': [], 'opportunities': [], 'threats': []},
        'action_items': [],
        'competitive_advantages': [],
        'risk_warnings': [],
        'academic_frameworks': [],
        'kpis': []
    }
    
    # SWOT Analysis
    your_score = your_analysis['customer_satisfaction'] + your_analysis['quality_score']
    comp_score = competitor_analysis['customer_satisfaction'] + competitor_analysis['quality_score']
    
    # STRENGTHS
    if your_analysis['customer_satisfaction'] > competitor_analysis['customer_satisfaction']:
        diff = your_analysis['customer_satisfaction'] - competitor_analysis['customer_satisfaction']
        recommendations['swot']['strengths'].append(
            f"Superior customer satisfaction ({diff:.1f}% higher than {competitor_brand})"
        )
        recommendations['academic_frameworks'].append(
            "Customer Satisfaction Index (CSI) analysis shows competitive edge"
        )
    
    if your_analysis['avg_rating'] > competitor_analysis['avg_rating']:
        recommendations['swot']['strengths'].append(
            f"Higher average rating ({your_analysis['avg_rating']:.1f}★ vs {competitor_analysis['avg_rating']:.1f}★)"
        )
    
    for keyword, score in your_analysis['strengths'][:2]:
        recommendations['swot']['strengths'].append(
            f"Positive brand association with '{keyword}' (TF-IDF score: {score:.3f})"
        )
    
    # WEAKNESSES
    if your_analysis['negative_pct'] > competitor_analysis['negative_pct']:
        diff = your_analysis['negative_pct'] - competitor_analysis['negative_pct']
        recommendations['swot']['weaknesses'].append(
            f"Higher negative sentiment by {diff:.1f}% (requires immediate attention)"
        )
        recommendations['action_items'].append(
            f"Priority 1: Address root causes of negative feedback"
        )
    
    for keyword, score in your_analysis['complaints'][:3]:
        recommendations['swot']['weaknesses'].append(
            f"Recurring complaint: '{keyword}' (frequency score: {score:.3f})"
        )
        recommendations['action_items'].append(
            f"Implement quality control measures for '{keyword}'"
        )
    
    # OPPORTUNITIES (Porter's Five Forces Analysis)
    for keyword, score in competitor_analysis['complaints'][:3]:
        recommendations['swot']['opportunities'].append(
            f"Competitor vulnerability: '{keyword}' ({competitor_brand}'s weakness)"
        )
        recommendations['action_items'].append(
            f"Launch marketing campaign highlighting superiority in '{keyword}'"
        )
        recommendations['academic_frameworks'].append(
            "Porter's Five Forces: Exploit competitor weaknesses"
        )
    
    # THREATS
    if competitor_analysis['customer_satisfaction'] > your_analysis['customer_satisfaction']:
        diff = competitor_analysis['customer_satisfaction'] - your_analysis['customer_satisfaction']
        recommendations['swot']['threats'].append(
            f"Competitor leads in customer satisfaction by {diff:.1f}%"
        )
        recommendations['risk_warnings'].append(
            f"Risk of customer churn to {competitor_brand}"
        )
    
    # KPIs for monitoring
    recommendations['kpis'].extend([
        f"Monthly Customer Satisfaction Score: Target > {your_analysis['customer_satisfaction']:.1f}%",
        f"Negative Review Reduction: Target < {your_analysis['negative_pct']:.1f}%",
        f"Competitive Gap Closure: Reduce gap by 20% in Q1"
    ])
    
    # Strategic positioning
    if your_score > comp_score:
        recommendations['action_items'].insert(0, 
            f"✅ DOMINANT POSITION: Leverage market leadership for expansion"
        )
        recommendations['academic_frameworks'].append(
            "Blue Ocean Strategy: Expand into untapped market segments"
        )
    else:
        gap = comp_score - your_score
        recommendations['action_items'].insert(0, 
            f"⚠️ COMPETITIVE GAP: {gap:.1f} points behind. Focus on differentiation strategy"
        )
        recommendations['academic_frameworks'].append(
            "Red Ocean Strategy: Compete on quality and customer experience"
        )
    
    return recommendations

def create_download_link(data, filename, file_type):
    """Create download links for reports"""
    if file_type == 'csv':
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a class="download-btn" href="data:file/csv;base64,{b64}" download="{filename}">📥 {filename}</a>'
    elif file_type == 'json':
        json_str = json.dumps(data, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a class="download-btn" href="data:application/json;base64,{b64}" download="{filename}">📥 {filename}</a>'
    elif file_type == 'txt':
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a class="download-btn" href="data:text/plain;base64,{b64}" download="{filename}">📥 {filename}</a>'
    else:
        href = ""
    
    return href

def generate_academic_report(analysis_data, recommendations):
    """Generate comprehensive academic report"""
    report = f"""
BUSINESS CONQUEST - ACADEMIC ANALYSIS REPORT
============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. EXECUTIVE SUMMARY
--------------------
- Competitive Analysis between {analysis_data['your_brand']} and {analysis_data['competitor_brand']}
- Based on {analysis_data['your_total_reviews']} and {analysis_data['comp_total_reviews']} reviews
- Overall Competitive Advantage: {analysis_data['win_pct']}%

2. METHODOLOGY
--------------
2.1 Text Analytics Techniques Applied:
   • VADER Sentiment Analysis
   • TF-IDF Keyword Extraction
   • Topic Modeling (LDA)
   • Frequency Analysis
   • Statistical Comparative Analysis

2.2 Business Frameworks Applied:
   • SWOT Analysis
   • Porter's Five Forces
   • Customer Satisfaction Index
   • Competitive Gap Analysis

3. DETAILED FINDINGS
--------------------
3.1 {analysis_data['your_brand']} Performance:
   • Customer Satisfaction: {analysis_data['your_satisfaction']:.1f}%
   • Quality Score: {analysis_data['your_quality']:.1f}%
   • Positive Sentiment: {analysis_data['your_positive']:.1f}%

3.2 {analysis_data['competitor_brand']} Performance:
   • Customer Satisfaction: {analysis_data['comp_satisfaction']:.1f}%
   • Quality Score: {analysis_data['comp_quality']:.1f}%
   • Negative Sentiment: {analysis_data['comp_negative']:.1f}%

4. STRATEGIC RECOMMENDATIONS
-----------------------------
{chr(10).join(['• ' + item for item in recommendations['action_items'][:5]])}

5. ACADEMIC CONTRIBUTIONS
-------------------------
This analysis demonstrates practical application of:
• Natural Language Processing in Business Intelligence
• Data Mining for Competitive Strategy
• Text Analytics for Customer Insights
• Quantitative Methods for Business Decision Making

6. LIMITATIONS & FUTURE WORK
----------------------------
• Dataset size constraints
• Language limitations (English only)
• Cross-domain generalizability
• Real-time analysis potential

CONCLUSION
----------
This analysis provides actionable insights for strategic decision-making,
demonstrating the value of text analytics in business competition.
"""
    return report

# ==================== WELCOME PAGE WITH ACADEMIC CONTEXT ====================
def show_welcome_page():
    st.markdown("""
    <div class="game-container">
        <div class="welcome-container">
            <h1 style="text-align: center; font-size: 3.5rem; margin-bottom: 10px;">
                🎮 BUSINESS CONQUEST PRO
            </h1>
            <h3 style="text-align: center; opacity: 0.9; margin-bottom: 10px;">
                Capstone Project: Text Analytics for Business Strategy
            </h3>
            <p style="text-align: center; font-size: 1.1rem; margin-bottom: 40px;">
                Topic: <b>Exploiting Business Intelligence from Customer Reviews</b>
            </p>
            
            <div class="academic-section">
                <h3>🎓 ACADEMIC CONTEXT & METHODOLOGY</h3>
                <p>This project implements cutting-edge Text Analytics techniques for competitive business intelligence:</p>
                
                <div class="methodology-card">
                    <h4>📊 TEXT ANALYTICS METHODOLOGIES</h4>
                    <ul>
                        <li><b>Sentiment Analysis:</b> VADER algorithm for polarity scoring</li>
                        <li><b>Keyword Extraction:</b> TF-IDF and frequency analysis</li>
                        <li><b>Topic Modeling:</b> Latent Dirichlet Allocation (LDA)</li>
                        <li><b>Natural Language Processing:</b> Text preprocessing & feature engineering</li>
                    </ul>
                </div>
                
                <div class="methodology-card">
                    <h4>💼 BUSINESS STRATEGY FRAMEWORKS</h4>
                    <ul>
                        <li><b>SWOT Analysis:</b> Automated strengths/weaknesses identification</li>
                        <li><b>Competitive Intelligence:</b> Head-to-head brand comparison</li>
                        <li><b>Customer Insights:</b> Review mining for business decisions</li>
                        <li><b>Strategic Recommendations:</b> Data-driven action plans</li>
                    </ul>
                </div>
            </div>
            
            <div class="feature-box">
                <h3>🎯 PROJECT OBJECTIVES</h3>
                <ol style="font-size: 1.1rem; line-height: 2;">
                    <li>Demonstrate practical application of Text Analytics in business</li>
                    <li>Implement NLP algorithms for customer review analysis</li>
                    <li>Generate actionable business intelligence from unstructured data</li>
                    <li>Provide competitive advantage through data-driven insights</li>
                    <li>Create an interactive platform for strategic decision-making</li>
                </ol>
            </div>
            
            <div class="feature-box">
                <h3>📁 DATASET REQUIREMENTS</h3>
                <p style="font-size: 1.1rem; line-height: 1.8;">
                    <b>Minimum Required Columns:</b><br>
                    • <code>Review_Text</code> (Customer feedback)<br>
                    • <code>Rating</code> (Numeric score 1-5)<br>
                    • <code>Brand_Name</code> (Company/Product identifier)<br><br>
                    <b>Optional:</b> Date, Product_Category, Customer_ID
                </p>
            </div>
            
            <div class="feature-box">
                <h3>📚 THEORETICAL FOUNDATIONS</h3>
                <p>This project integrates concepts from:</p>
                <ul>
                    <li><b>Information Retrieval</b> (Text mining techniques)</li>
                    <li><b>Business Intelligence</b> (Competitive analysis)</li>
                    <li><b>Customer Relationship Management</b> (Sentiment analysis)</li>
                    <li><b>Strategic Management</b> (SWOT, Porter's frameworks)</li>
                </ul>
            </div>
            
            <div style="text-align: center; margin-top: 40px;">
                <p style="font-size: 1.2rem; color: #00d4ff; margin-bottom: 20px;">
                    🚀 Ready to analyze your business competition?
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 START ANALYSIS & UPLOAD DATA", type="primary", use_container_width=True):
            st.session_state.page = 'analysis'
            st.rerun()

# ==================== ENHANCED ANALYSIS PAGE ====================
def show_analysis_page():
    st.markdown('<div class="game-container">', unsafe_allow_html=True)
    
    # Header with academic context
    st.markdown(f"""
    <div class="game-header">
        <h1 style="font-size: 3rem; margin: 0;">🎓 BUSINESS CONQUEST PRO</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Text Analytics Capstone Project</p>
        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px;">
            <div>
                <div style="font-size: 0.9rem; opacity: 0.8;">ANALYSIS LEVEL</div>
                <div style="font-size: 2rem; font-weight: bold;">{st.session_state.player_level}</div>
            </div>
            <div>
                <div style="font-size: 0.9rem; opacity: 0.8;">DATA POINTS</div>
                <div style="font-size: 2rem; font-weight: bold;">{st.session_state.player_xp}</div>
            </div>
            <div>
                <div style="font-size: 0.9rem; opacity: 0.8;">INSIGHTS</div>
                <div style="font-size: 2rem; font-weight: bold;">{len(st.session_state.completed_missions)}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced Features Toggle
    with st.sidebar:
        st.markdown("### ⚙️ ANALYSIS SETTINGS")
        st.session_state.advanced_features = st.toggle("Enable Advanced NLP", value=False)
        if st.session_state.advanced_features:
            st.info("✅ Advanced features enabled: TF-IDF, LDA Topic Modeling")
        
        if st.session_state.data_loaded:
            st.markdown("### 📊 DATASET INFO")
            st.write(f"**Reviews:** {len(st.session_state.df):,}")
            st.write(f"**Brands:** {len(st.session_state.available_brands)}")
            st.write(f"**Columns:** {len(st.session_state.df.columns)}")
            
            if st.button("🔄 Load New Dataset"):
                st.session_state.data_loaded = False
                st.session_state.df = None
                st.rerun()
    
    # File Upload Section
    if not st.session_state.data_loaded:
        st.markdown("<h2 style='text-align: center;'>📁 UPLOAD YOUR DATASET</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader("Choose a CSV file with customer reviews", type=['csv'])
        
        with col2:
            st.markdown("### 📚 Sample Datasets")
            st.markdown("""
            Try with:
            - Amazon product reviews
            - Yelp restaurant reviews
            - TripAdvisor hotel reviews
            - Custom business reviews
            """)
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Show dataset preview
                st.markdown("### 📋 Dataset Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Auto-detect columns
                col_mapping = detect_columns(df)
                
                if not all(key in col_mapping for key in ['review', 'rating', 'brand']):
                    st.error("❌ Cannot auto-detect required columns.")
                    
                    # Manual column selection
                    st.markdown("### 🔧 Manual Column Mapping")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        review_col = st.selectbox("Review Text Column:", df.columns)
                    with c2:
                        rating_col = st.selectbox("Rating Column:", df.columns)
                    with c3:
                        brand_col = st.selectbox("Brand Column:", df.columns)
                    
                    col_mapping = {
                        'review': review_col,
                        'rating': rating_col,
                        'brand': brand_col
                    }
                else:
                    st.success("✅ Columns auto-detected successfully!")
                
                if st.button("✅ PROCESS DATASET", type="primary"):
                    # Process dataset
                    df_processed = df.copy()
                    df_processed = df_processed.dropna(subset=[col_mapping['review']])
                    
                    # Standardize columns
                    df_processed['Reviews'] = df_processed[col_mapping['review']].astype(str)
                    df_processed['Rating'] = pd.to_numeric(df_processed[col_mapping['rating']], errors='coerce')
                    df_processed['Brand Name'] = df_processed[col_mapping['brand']].str.strip().str.title()
                    
                    # Add date if available
                    if 'date' in col_mapping:
                        try:
                            df_processed['Date'] = pd.to_datetime(df_processed[col_mapping['date']])
                        except:
                            pass
                    
                    st.session_state.df = df_processed
                    st.session_state.column_mapping = {
                        'review': 'Reviews',
                        'rating': 'Rating',
                        'brand': 'Brand Name'
                    }
                    
                    if 'date' in col_mapping and 'Date' in df_processed.columns:
                        st.session_state.column_mapping['date'] = 'Date'
                    
                    st.session_state.data_loaded = True
                    st.session_state.available_brands = sorted(df_processed['Brand Name'].unique().tolist())
                    st.success(f"🎉 Loaded {len(df_processed)} reviews from {len(st.session_state.available_brands)} brands!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 MISSIONS", "⚔️ BATTLE", "📊 INSIGHTS", "📚 ACADEMIC", "🏆 ACHIEVEMENTS"])
    
    with tab1:
        st.markdown("<h2 style='text-align: center;'>🎯 SELECT YOUR BATTLE</h2>", unsafe_allow_html=True)
        
        available = st.session_state.available_brands[:6] if len(st.session_state.available_brands) >= 2 else []
        
        if len(available) >= 2:
            missions = [
                {"id": 1, "title": "MARKET LEADER BATTLE", "target": available[0], "enemy": available[1], 
                 "xp": 100, "icon": "🏆", "color": "#6a11cb", "difficulty": "Advanced"},
                {"id": 2, "title": "COMPETITIVE ANALYSIS", "target": available[2] if len(available) > 2 else available[0], 
                 "enemy": available[3] if len(available) > 3 else available[1], "xp": 150, "icon": "📊", "color": "#2575fc", "difficulty": "Intermediate"}
            ]
            
            cols = st.columns(2)
            for idx, m in enumerate(missions):
                with cols[idx]:
                    completed = m["id"] in st.session_state.completed_missions
                    st.markdown(f"""
                    <div class="level-card" style="border-color: {m['color']};">
                        <div class="level-badge">{m['icon']} • {m['xp']} XP • {m['difficulty']}</div>
                        <h3>{m['title']}</h3>
                        <p>YOUR BRAND: <b>{m['target']}</b></p>
                        <p>COMPETITOR: <b>{m['enemy']}</b></p>
                        {'✅ COMPLETED' if completed else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if not completed:
                        if st.button(f"🚀 START", key=f"m{m['id']}"):
                            st.session_state.current_battle = {"target": m['target'], "enemy": m['enemy'], 
                                                              "mission_id": m['id'], "xp_reward": m['xp']}
                            st.session_state.show_recommendations = False
                            st.rerun()
        
        st.markdown("<h3 style='text-align: center;'>CREATE CUSTOM BATTLE</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            target = st.selectbox("Your Brand:", st.session_state.available_brands)
        with c2:
            enemy = st.selectbox("Competitor:", [b for b in st.session_state.available_brands if b != target])
        
        if st.button("🚀 CREATE CUSTOM BATTLE", type="primary"):
            st.session_state.current_battle = {"target": target, "enemy": enemy, 
                                              "is_custom": True, "xp_reward": 75}
            st.session_state.show_recommendations = False
            st.rerun()
    
    with tab2:
        if st.session_state.current_battle:
            battle = st.session_state.current_battle
            
            st.markdown(f"""
            <div class="battle-arena">
                <h2 style='text-align: center;'>⚔️ COMPETITIVE ANALYSIS</h2>
                <div style='text-align: center; font-size: 1.5rem; margin: 20px 0;'>
                    <span style='color: #00d4ff;'>{battle['target']}</span> 
                    <span style='margin: 0 20px;'>VS</span>
                    <span style='color: #ff0080;'>{battle['enemy']}</span>
                </div>
                <p style='text-align: center; opacity: 0.8;'>Real-time Text Analytics in action</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner('🔍 Analyzing reviews with NLP algorithms...'):
                your_data = analyze_brand_performance(st.session_state.df, battle['target'], st.session_state.column_mapping)
                comp_data = analyze_brand_performance(st.session_state.df, battle['enemy'], st.session_state.column_mapping)
            
            if your_data and comp_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"<h3 style='text-align: center;'>🎯 {battle['target']}</h3>", unsafe_allow_html=True)
                    
                    metrics = [
                        ("Customer Satisfaction", int(your_data['customer_satisfaction']), "#00d4ff"),
                        ("Quality Score", int(your_data['quality_score']), "#00b09b"),
                        ("Positive Reviews", int(your_data['positive_pct']), "#4CC9F0")
                    ]
                    
                    for name, val, color in metrics:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div style="display: flex; justify-content: space-between;">
                                <span>{name}</span><span style="color: {color};">{val}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {val}%; background: {color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<h4>💪 Key Strengths:</h4>", unsafe_allow_html=True)
                    for idx, (kw, score) in enumerate(your_data['strengths'][:3], 1):
                        st.markdown(f"<div class='stat-card'>✅ {idx}. {kw} (score: {score:.3f})</div>", unsafe_allow_html=True)
                    
                    st.markdown("<h4>⚠️ Top Complaints:</h4>", unsafe_allow_html=True)
                    for idx, (kw, score) in enumerate(your_data['complaints'][:3], 1):
                        st.markdown(f"<div class='stat-card'>❌ {idx}. {kw} (score: {score:.3f})</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"<h3 style='text-align: center;'>🎯 {battle['enemy']}</h3>", unsafe_allow_html=True)
                    
                    metrics = [
                        ("Customer Satisfaction", int(comp_data['customer_satisfaction']), "#ff0080"),
                        ("Quality Score", int(comp_data['quality_score']), "#ff8c00"),
                        ("Negative Reviews", int(comp_data['negative_pct']), "#F72585")
                    ]
                    
                    for name, val, color in metrics:
                        st.markdown(f"""
                        <div class="stat-card" style="border-color: {color};">
                            <div style="display: flex; justify-content: space-between;">
                                <span>{name}</span><span style="color: {color};">{val}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {val}%; background: {color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<h4>💪 Their Strengths:</h4>", unsafe_allow_html=True)
                    for idx, (kw, score) in enumerate(comp_data['strengths'][:3], 1):
                        st.markdown(f"<div class='stat-card' style='border-color: #ff0080;'>✅ {idx}. {kw} (score: {score:.3f})</div>", unsafe_allow_html=True)
                    
                    st.markdown("<h4>🔍 Their Weaknesses:</h4>", unsafe_allow_html=True)
                    for idx, (kw, score) in enumerate(comp_data['complaints'][:3], 1):
                        st.markdown(f"<div class='stat-card' style='border-color: #ff0080;'>❌ {idx}. {kw} (score: {score:.3f})</div>", unsafe_allow_html=True)
                
                # Competitive Advantage Analysis
                your_score = your_data['customer_satisfaction'] + your_data['quality_score']
                comp_score = comp_data['customer_satisfaction'] + comp_data['quality_score']
                win_pct = int((your_score / (your_score + comp_score)) * 100) if (your_score + comp_score) > 0 else 50
                
                st.markdown(f"""
                <div class="stat-card" style="border-color: gold; text-align: center; background: rgba(255,215,0,0.1);">
                    <h3>📊 COMPETITIVE ADVANTAGE ANALYSIS</h3>
                    <div style="font-size: 2rem; color: gold; margin: 10px 0;">
                        Your Advantage: {win_pct}%
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 20px 0;">
                        <div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">Total Reviews</div>
                            <div style="font-size: 1.5rem; color: #00d4ff;">{your_data['total_reviews']:,}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">Avg Rating</div>
                            <div style="font-size: 1.5rem; color: #00b09b;">{your_data['avg_rating']:.1f}★</div>
                        </div>
                        <div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">Sentiment</div>
                            <div style="font-size: 1.5rem; color: #4CC9F0;">{your_data['avg_sentiment']:.2f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # RECOMMENDATIONS
                if st.session_state.show_recommendations:
                    recs = st.session_state.recommendations
                    
                    st.markdown("<br><h2 style='text-align: center;'>💡 STRATEGIC RECOMMENDATIONS</h2>", unsafe_allow_html=True)
                    
                    # SWOT Analysis
                    st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
                    st.markdown("<h3>📊 SWOT ANALYSIS (Strategic Framework)</h3>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<h4 style='color: #00d4ff;'>💪 STRENGTHS</h4>", unsafe_allow_html=True)
                        for s in recs['swot']['strengths'][:3]:
                            st.markdown(f"<div class='action-item'>✅ {s}</div>", unsafe_allow_html=True)
                        
                        st.markdown("<h4 style='color: #ff8c00;'>⚠️ WEAKNESSES</h4>", unsafe_allow_html=True)
                        for w in recs['swot']['weaknesses'][:3]:
                            st.markdown(f"<div class='action-item' style='border-color: #ff8c00;'>❌ {w}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<h4 style='color: #00b09b;'>🎯 OPPORTUNITIES</h4>", unsafe_allow_html=True)
                        for o in recs['swot']['opportunities'][:3]:
                            st.markdown(f"<div class='action-item' style='border-color: #00b09b;'>💡 {o}</div>", unsafe_allow_html=True)
                        
                        st.markdown("<h4 style='color: #ff0080;'>🚨 THREATS</h4>", unsafe_allow_html=True)
                        for t in recs['swot']['threats'][:3]:
                            st.markdown(f"<div class='action-item' style='border-color: #ff0080;'>⚠️ {t}</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Action Items
                    st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
                    st.markdown("<h3>🎯 IMMEDIATE ACTION ITEMS</h3>", unsafe_allow_html=True)
                    for idx, action in enumerate(recs['action_items'][:5], 1):
                        st.markdown(f"<div class='action-item'><b>{idx}.</b> {action}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Academic Frameworks
                    if recs['academic_frameworks']:
                        st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
                        st.markdown("<h3>📚 APPLIED ACADEMIC FRAMEWORKS</h3>", unsafe_allow_html=True)
                        for framework in recs['academic_frameworks']:
                            st.markdown(f"<div class='action-item' style='border-color: #4CC9F0;'>🎓 {framework}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Report Generation
                    st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
                    st.markdown("<h3>📄 GENERATE REPORTS</h3>", unsafe_allow_html=True)
                    
                    analysis_data = {
                        'your_brand': battle['target'],
                        'competitor_brand': battle['enemy'],
                        'your_total_reviews': your_data['total_reviews'],
                        'comp_total_reviews': comp_data['total_reviews'],
                        'win_pct': win_pct,
                        'your_satisfaction': your_data['customer_satisfaction'],
                        'comp_satisfaction': comp_data['customer_satisfaction'],
                        'your_quality': your_data['quality_score'],
                        'comp_quality': comp_data['quality_score'],
                        'your_positive': your_data['positive_pct'],
                        'comp_negative': comp_data['negative_pct']
                    }
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        report_text = generate_academic_report(analysis_data, recs)
                        st.markdown(create_download_link(report_text, "academic_report.txt", "txt"), unsafe_allow_html=True)
                    
                    with col2:
                        # Create summary DataFrame
                        summary_df = pd.DataFrame({
                            'Metric': ['Total Reviews', 'Avg Rating', 'Customer Satisfaction', 'Quality Score', 'Positive %'],
                            battle['target']: [your_data['total_reviews'], your_data['avg_rating'], 
                                            f"{your_data['customer_satisfaction']:.1f}%", 
                                            f"{your_data['quality_score']:.1f}%", 
                                            f"{your_data['positive_pct']:.1f}%"],
                            battle['enemy']: [comp_data['total_reviews'], comp_data['avg_rating'],
                                            f"{comp_data['customer_satisfaction']:.1f}%",
                                            f"{comp_data['quality_score']:.1f}%",
                                            f"{comp_data['positive_pct']:.1f}%"]
                        })
                        st.markdown(create_download_link(summary_df, "comparison_summary.csv", "csv"), unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(create_download_link(recs, "strategic_recommendations.json", "json"), unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Complete mission button
                    if st.button("✅ COMPLETE ANALYSIS & EARN XP", type="primary", use_container_width=True):
                        xp = battle['xp_reward']
                        st.session_state.player_xp += xp
                        
                        if not battle.get("is_custom") and battle.get('mission_id') not in st.session_state.completed_missions:
                            st.session_state.completed_missions.append(battle['mission_id'])
                        
                        if st.session_state.player_xp >= st.session_state.player_level * 1000:
                            st.session_state.player_level += 1
                            st.balloons()
                        
                        # Save to history
                        st.session_state.analysis_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'your_brand': battle['target'],
                            'competitor': battle['enemy'],
                            'xp_earned': xp,
                            'win_pct': win_pct
                        })
                        
                        st.success(f"🎉 Analysis Complete! +{xp} XP earned")
                        st.session_state.current_battle = None
                        st.session_state.show_recommendations = False
                        st.rerun()
                
                else:
                    # Generate recommendations button
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💡 GENERATE STRATEGIC INSIGHTS", type="primary", use_container_width=True):
                            st.session_state.show_recommendations = True
                            st.session_state.recommendations = generate_strategic_recommendations(
                                your_data, comp_data, battle['target'], battle['enemy']
                            )
                            st.rerun()
                    with col2:
                        if st.button("🏁 END BATTLE", use_container_width=True):
                            st.session_state.current_battle = None
                            st.rerun()
            
            else:
                st.error("❌ Insufficient data for analysis. Try different brands.")
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 50px;">
                <div style="font-size: 4rem;">⚔️</div>
                <h2>NO ACTIVE BATTLE</h2>
                <p style="opacity: 0.7;">Select a mission from the Missions tab to begin analysis!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h2 style='text-align: center;'>📊 BUSINESS INTELLIGENCE DASHBOARD</h2>", unsafe_allow_html=True)
        
        # Summary Stats
        col1, col2, col3, col4 = st.columns(4)
        total_reviews = len(st.session_state.df)
        total_brands = len(st.session_state.available_brands)
        avg_rating = st.session_state.df['Rating'].mean()
        avg_review_length = st.session_state.df['Reviews'].apply(lambda x: len(str(x).split())).mean()
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 0.9em; opacity: 0.7;">TOTAL REVIEWS</div>
                <div style="font-size: 2em; color: #00d4ff;">{total_reviews:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 0.9em; opacity: 0.7;">BRANDS</div>
                <div style="font-size: 2em; color: #00b09b;">{total_brands}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 0.9em; opacity: 0.7;">AVG RATING</div>
                <div style="font-size: 2em; color: #ff0080;">{avg_rating:.1f}★</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 0.9em; opacity: 0.7;">AVG REVIEW LENGTH</div>
                <div style="font-size: 2em; color: gold;">{avg_review_length:.0f} words</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Top Brands Analysis
        st.markdown("<br><h3>📊 Top 5 Brands Performance</h3>", unsafe_allow_html=True)
        
        top_brands = st.session_state.df['Brand Name'].value_counts().head(5).index.tolist()
        brand_perf = []
        
        for brand in top_brands:
            analysis = analyze_brand_performance(st.session_state.df, brand, st.session_state.column_mapping)
            if analysis:
                brand_perf.append({
                    'Brand': brand,
                    'Reviews': analysis['total_reviews'],
                    'Avg Rating': analysis['avg_rating'],
                    'Satisfaction': analysis['customer_satisfaction'],
                    'Sentiment': analysis['avg_sentiment']
                })
        
        if brand_perf:
            perf_df = pd.DataFrame(brand_perf)
            
            # Create visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=perf_df['Brand'],
                y=perf_df['Avg Rating'],
                name='Average Rating',
                marker_color='#00d4ff',
                text=perf_df['Avg Rating'].round(2),
                textposition='auto'
            ))
            
            fig.add_trace(go.Scatter(
                x=perf_df['Brand'],
                y=perf_df['Satisfaction']/20,  # Normalize for scaling
                name='Satisfaction Score',
                mode='lines+markers',
                line=dict(color='#ff0080', width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                xaxis_title="Brand",
                yaxis_title="Average Rating",
                yaxis2=dict(title="Satisfaction Score", overlaying='y', side='right'),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed table
            st.markdown("<h4>📋 Detailed Brand Metrics</h4>", unsafe_allow_html=True)
            st.dataframe(perf_df.style.background_gradient(subset=['Avg Rating', 'Satisfaction'], cmap='RdYlGn'),
                        use_container_width=True)
        
        # Sentiment Distribution
        st.markdown("<br><h3>📈 Overall Sentiment Distribution</h3>", unsafe_allow_html=True)
        
        if NLP_AVAILABLE:
            with st.spinner('Analyzing overall sentiment...'):
                sample_reviews = st.session_state.df['Reviews'].sample(min(1000, len(st.session_state.df)))
                sentiments = sample_reviews.apply(analyze_sentiment)
                sentiment_values = [s['compound'] for s in sentiments]
            
            fig = px.histogram(x=sentiment_values, nbins=20,
                             title="Distribution of Sentiment Scores",
                             labels={'x': 'Sentiment Score', 'y': 'Count'},
                             color_discrete_sequence=['#00d4ff'])
            
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("<h2 style='text-align: center;'>📚 ACADEMIC CONTEXT & METHODOLOGY</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="academic-section">
            <h3>🎓 PROJECT CONTRIBUTION TO TEXT ANALYTICS</h3>
            <p>This project demonstrates practical application of Text Analytics in Business Intelligence:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="methodology-card">
                <h4>📊 TEXT ANALYTICS TECHNIQUES</h4>
                <ul>
                    <li><b>Sentiment Analysis:</b> VADER algorithm implementation</li>
                    <li><b>Keyword Extraction:</b> TF-IDF with scikit-learn</li>
                    <li><b>Topic Modeling:</b> LDA for review clustering</li>
                    <li><b>Text Preprocessing:</b> NLTK for tokenization & lemmatization</li>
                    <li><b>Feature Engineering:</b> Sentiment scores, satisfaction metrics</li>
                </ul>
            </div>
            
            <div class="methodology-card">
                <h4>📈 BUSINESS ANALYTICS</h4>
                <ul>
                    <li>Competitive Benchmarking Analysis</li>
                    <li>Customer Satisfaction Metrics</li>
                    <li>Quality Score Calculation</li>
                    <li>Engagement Scoring</li>
                    <li>Trend Analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="methodology-card">
                <h4>💼 STRATEGIC FRAMEWORKS</h4>
                <ul>
                    <li><b>SWOT Analysis:</b> Automated generation</li>
                    <li><b>Porter's Five Forces:</b> Competitive positioning</li>
                    <li><b>Blue Ocean Strategy:</b> Market expansion insights</li>
                    <li><b>KPI Development:</b> Performance metrics</li>
                    <li><b>Risk Assessment:</b> Threat identification</li>
                </ul>
            </div>
            
            <div class="methodology-card">
                <h4>🎯 PROJECT OUTCOMES</h4>
                <ul>
                    <li>Actionable business insights</li>
                    <li>Competitive advantage identification</li>
                    <li>Data-driven decision support</li>
                    <li>Strategic recommendation generation</li>
                    <li>Interactive business intelligence platform</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="academic-section">
            <h3>🔬 RESEARCH METHODOLOGY</h3>
            <p><b>1. Data Collection:</b> Customer review datasets from various sources</p>
            <p><b>2. Preprocessing:</b> Text cleaning, normalization, and feature extraction</p>
            <p><b>3. Analysis:</b> Sentiment scoring, keyword extraction, topic modeling</p>
            <p><b>4. Interpretation:</b> Business intelligence generation and strategy formulation</p>
            <p><b>5. Visualization:</b> Interactive dashboards and reporting</p>
        </div>
        
        <div class="academic-section">
            <h3>📝 LIMITATIONS & FUTURE WORK</h3>
            <p><b>Current Limitations:</b></p>
            <ul>
                <li>English language bias in NLP models</li>
                <li>Dataset size dependencies</li>
                <li>Review authenticity verification</li>
                <li>Cross-domain generalization</li>
            </ul>
            
            <p><b>Future Enhancements:</b></p>
            <ul>
                <li>Multilingual sentiment analysis</li>
                <li>Deep learning models for accuracy</li>
                <li>Real-time analysis capabilities</li>
                <li>Integration with social media data</li>
                <li>Predictive analytics for market trends</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab5:
        st.markdown("<h2 style='text-align: center;'>🏆 ACHIEVEMENTS & PROGRESS</h2>", unsafe_allow_html=True)
        
        achievements = [
            {"name": "Data Loader", "desc": "Successfully loaded and processed dataset", 
             "earned": st.session_state.data_loaded, "reward": "🎖️", "xp": 50},
            {"name": "First Analysis", "desc": "Complete your first competitive analysis", 
             "earned": len(st.session_state.completed_missions) > 0, "reward": "🏆", "xp": 100},
            {"name": "Strategic Thinker", "desc": "Complete 3 different analyses", 
             "earned": len(st.session_state.completed_missions) >= 3, "reward": "⭐", "xp": 200},
            {"name": "Data Explorer", "desc": "Analyze 500+ data points", 
             "earned": st.session_state.player_xp >= 500, "reward": "💰", "xp": 500},
            {"name": "Master Analyst", "desc": "Reach Analysis Level 3", 
             "earned": st.session_state.player_level >= 3, "reward": "👑", "xp": 300},
            {"name": "Competitive Genius", "desc": "Complete 5 analyses", 
             "earned": len(st.session_state.completed_missions) >= 5, "reward": "⚔️", "xp": 400}
        ]
        
        cols = st.columns(3)
        for idx, achievement in enumerate(achievements):
            with cols[idx % 3]:
                earned_style = "border-color: gold; background: rgba(255,215,0,0.1);" if achievement["earned"] else "opacity: 0.5;"
                
                st.markdown(f"""
                <div class="level-card" style="{earned_style}">
                    <div style="font-size: 2.5rem;">{achievement['reward']}</div>
                    <h4 style="margin: 10px 0;">{achievement['name']}</h4>
                    <p style="font-size: 0.9em; opacity: 0.8;">{achievement['desc']}</p>
                    <div style="margin-top: 10px; font-weight: bold; color: #00d4ff;">
                        {achievement['xp']} XP
                    </div>
                    <div style="margin-top: 5px; font-size: 0.9em;">
                        {"✅ EARNED" if achievement['earned'] else "🔒 LOCKED"}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Progress Summary
        st.markdown("<br><h3>📊 Progress Summary</h3>", unsafe_allow_html=True)
        
        progress_data = {
            'Category': ['XP Earned', 'Analyses Completed', 'Brands Analyzed', 'Data Points'],
            'Value': [st.session_state.player_xp, len(st.session_state.completed_missions), 
                     len(st.session_state.analysis_history), len(st.session_state.df) if st.session_state.df is not None else 0]
        }
        
        progress_df = pd.DataFrame(progress_data)
        st.dataframe(progress_df.style.background_gradient(subset=['Value'], cmap='Blues'),
                    use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== MAIN APP ROUTER ====================
if st.session_state.page == 'welcome':
    show_welcome_page()
else:
    show_analysis_page()

# ==================== FOOTER ====================
st.markdown("""
<div style="text-align: center; opacity: 0.7; font-size: 0.9em; padding: 20px; border-top: 1px solid rgba(255,255,255,0.1);">
    <p>🎓 BUSINESS CONQUEST PRO v3.0 | Capstone Project: Text Analytics for Business Strategy</p>
    <p>Topic: <b>Exploiting Business Intelligence from Customer Reviews</b></p>
    <p>Powered by: VADER Sentiment Analysis • TF-IDF • LDA • NLTK • scikit-learn</p>
    <p>© 2024 Text Analytics Capstone Project | Academic Use Only</p>
</div>
""", unsafe_allow_html=True)
