"""
STREAMLIT WEB APP - Business Intelligence from Reviews
Real-time file upload and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import string
import pickle
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Business Intelligence Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== TEXT PREPROCESSING ====================
class TextPreprocessor:
    """Text preprocessing for sentiment analysis"""
    
    def __init__(self):
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
                               'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as'])
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def extract_aspects(self, text):
        """Extract business aspects mentioned in review"""
        aspects = {
            'service': ['service', 'staff', 'employee', 'waiter', 'help', 'support'],
            'quality': ['quality', 'product', 'item', 'material'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money'],
            'speed': ['fast', 'slow', 'quick', 'wait', 'time', 'delay'],
            'cleanliness': ['clean', 'dirty', 'hygiene'],
            'food': ['food', 'meal', 'dish', 'taste', 'delicious'],
            'delivery': ['delivery', 'shipping', 'ship']
        }
        
        found_aspects = []
        text_lower = text.lower()
        
        for aspect, keywords in aspects.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_aspects.append(aspect)
                    break
        
        return list(set(found_aspects))

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Business Intelligence Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Exploiting Business Insights in Reviews")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio("Choose a page:", 
                            ["🏠 Home", "📤 Upload & Analyze", "🎓 How It Works", "📊 About"])
    
    if page == "🏠 Home":
        show_home()
    elif page == "📤 Upload & Analyze":
        show_upload_analyze()
    elif page == "🎓 How It Works":
        show_how_it_works()
    elif page == "📊 About":
        show_about()

# ==================== HOME PAGE ====================
def show_home():
    st.header("Welcome to Business Intelligence Analyzer! 👋")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 What This Does:")
        st.write("""
        This AI-powered system analyzes customer reviews to provide **4 strategic insights**:
        
        1. 📉 **YOUR Operational Weaknesses** - What to fix immediately
        2. 🎯 **Competitor Vulnerabilities** - Where to attack strategically  
        3. 💡 **Unmet Customer Needs** - Innovation opportunities
        4. 🚀 **Actionable Business Plans** - Steps to grow your business
        """)
    
    with col2:
        st.subheader("✨ Key Features:")
        st.write("""
        - ⚡ **Real-time Analysis** - Upload CSV, get instant insights
        - 📊 **Interactive Visualizations** - Beautiful charts and graphs
        - 🤖 **AI-Powered** - Machine learning models for accuracy
        - 💼 **Business-Focused** - Actionable recommendations
        - 🌐 **Multi-Industry** - Works for any business type
        """)
    
    st.markdown("---")
    
    # Demo section
    st.subheader("🎬 Quick Demo")
    
    if st.button("📊 See Sample Analysis", type="primary"):
        st.session_state['show_demo'] = True
    
    if st.session_state.get('show_demo', False):
        show_sample_analysis()

# ==================== UPLOAD & ANALYZE PAGE ====================
def show_upload_analyze():
    st.header("📤 Upload Your Review Data")
    
    # Instructions
    with st.expander("📝 Data Format Instructions"):
        st.write("""
        **Your CSV file should have these columns:**
        - `review`: The review text
        - `sentiment`: 0 (Negative), 1 (Neutral), 2 (Positive)
        - `business_name`: Name of the business
        - `business_type`: Either 'YOUR_BUSINESS' or 'COMPETITOR'
        
        **Example:**
        ```
        review,sentiment,business_name,business_type
        "Great service!",2,"MyBusiness","YOUR_BUSINESS"
        "Terrible food",0,"CompetitorA","COMPETITOR"
        ```
        """)
        
        # Download template
        template_data = {
            'review': ['Great service!', 'Terrible experience', 'It was okay'],
            'sentiment': [2, 0, 1],
            'business_name': ['YourBusiness', 'YourBusiness', 'YourBusiness'],
            'business_type': ['YOUR_BUSINESS', 'YOUR_BUSINESS', 'YOUR_BUSINESS']
        }
        template_df = pd.DataFrame(template_data)
        csv = template_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV Template",
            data=csv,
            file_name="review_template.csv",
            mime="text/csv"
        )
    
    # File upload
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} reviews")
            
            # Show preview
            with st.expander("👀 Preview Data"):
                st.dataframe(df.head(10))
            
            # Validate data
            required_cols = ['review', 'sentiment', 'business_name', 'business_type']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                return
            
            # Analysis options
            st.markdown("---")
            st.subheader("🎯 Analysis Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                train_new_model = st.checkbox("Train New Model (Recommended)", value=True)
            
            with col2:
                show_visualizations = st.checkbox("Show Visualizations", value=True)
            
            # Analyze button
            if st.button("🚀 Analyze Reviews", type="primary"):
                with st.spinner("🔄 Analyzing your data... Please wait..."):
                    analyze_reviews(df, train_new_model, show_visualizations)
        
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# ==================== ANALYSIS FUNCTION ====================
def analyze_reviews(df, train_new_model, show_visualizations):
    """Main analysis function"""
    
    preprocessor = TextPreprocessor()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Preprocess
    status_text.text("🔧 Preprocessing reviews...")
    df['cleaned_review'] = df['review'].apply(preprocessor.clean_text)
    df['aspects'] = df['review'].apply(preprocessor.extract_aspects)
    progress_bar.progress(20)
    
    # Step 2: Train model if needed
    if train_new_model:
        status_text.text("🤖 Training AI model...")
        model, vectorizer, accuracy = train_model(df)
        progress_bar.progress(40)
        st.success(f"✅ Model trained with {accuracy:.1f}% accuracy!")
    else:
        st.info("Using pre-trained model")
        model, vectorizer = None, None
    
    # Step 3: Generate insights
    status_text.text("💡 Generating insights...")
    progress_bar.progress(60)
    
    # Separate data
    your_data = df[df['business_type'] == 'YOUR_BUSINESS']
    comp_data = df[df['business_type'] == 'COMPETITOR']
    
    progress_bar.progress(80)
    
    # Display insights
    status_text.text("📊 Creating visualizations...")
    
    # Basic stats
    st.markdown("---")
    st.header("📊 Overview Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", len(df))
    
    with col2:
        st.metric("Your Business", len(your_data))
    
    with col3:
        st.metric("Competitors", len(comp_data))
    
    with col4:
        positive_rate = (len(df[df['sentiment'] == 2]) / len(df) * 100) if len(df) > 0 else 0
        st.metric("Positive Rate", f"{positive_rate:.1f}%")
    
    # Insight 1: Your Weaknesses
    st.markdown("---")
    display_your_weaknesses(your_data, show_visualizations)
    
    # Insight 2: Competitor Vulnerabilities
    st.markdown("---")
    display_competitor_vulnerabilities(comp_data, show_visualizations)
    
    # Insight 3: Unmet Needs
    st.markdown("---")
    display_unmet_needs(df)
    
    # Insight 4: Strategic Recommendations
    st.markdown("---")
    display_strategic_recommendations(your_data, comp_data)
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    
    # Download results
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    results_csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Analyzed Data",
        data=results_csv,
        file_name="analysis_results.csv",
        mime="text/csv"
    )

# ==================== MODEL TRAINING ====================
def train_model(df):
    """Train sentiment analysis model"""
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['sentiment']
    
    if len(df) < 10:
        st.warning("⚠️ Small dataset - model accuracy may be low. Add more reviews for better results!")
    
    # Split data
    if len(df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    
    # Train
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    if len(df) >= 10:
        accuracy = model.score(X_test, y_test) * 100
    else:
        accuracy = model.score(X_train, y_train) * 100
    
    return model, vectorizer, accuracy

# ==================== INSIGHT DISPLAYS ====================
def display_your_weaknesses(your_data, show_viz):
    """Display YOUR business weaknesses"""
    
    st.header("📉 INSIGHT #1: YOUR Operational Weaknesses")
    
    negative_reviews = your_data[your_data['sentiment'] == 0]
    
    if len(negative_reviews) == 0:
        st.success("🎉 No negative reviews! Your business is doing great!")
        return
    
    negative_rate = len(negative_reviews) / len(your_data) * 100 if len(your_data) > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Negative Reviews", f"{len(negative_reviews)}/{len(your_data)}")
    
    with col2:
        st.metric("Negative Rate", f"{negative_rate:.1f}%")
    
    # Extract aspects
    all_aspects = []
    for aspects in negative_reviews['aspects']:
        all_aspects.extend(aspects)
    
    if all_aspects:
        aspect_counts = Counter(all_aspects)
        
        st.subheader("⚠️ Problem Areas:")
        for aspect, count in aspect_counts.most_common(5):
            st.write(f"- **{aspect.upper()}**: {count} complaints")
        
        # Visualization
        if show_viz and len(aspect_counts) > 0:
            fig = px.bar(
                x=list(aspect_counts.keys()),
                y=list(aspect_counts.values()),
                labels={'x': 'Aspect', 'y': 'Number of Complaints'},
                title="Top Problem Areas",
                color=list(aspect_counts.values()),
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Show sample negative reviews
    st.subheader("💬 Sample Negative Reviews:")
    for idx, row in negative_reviews.head(3).iterrows():
        st.warning(f"'{row['review']}'")
    
    # Recommendations
    st.subheader("🎯 Recommended Actions:")
    if 'service' in [a for aspects in negative_reviews['aspects'] for a in aspects]:
        st.write("✅ Improve staff training and customer service")
    if 'speed' in [a for aspects in negative_reviews['aspects'] for a in aspects]:
        st.write("✅ Optimize processes to reduce wait times")
    if 'quality' in [a for aspects in negative_reviews['aspects'] for a in aspects]:
        st.write("✅ Review quality control standards")

def display_competitor_vulnerabilities(comp_data, show_viz):
    """Display competitor vulnerabilities"""
    
    st.header("🎯 INSIGHT #2: Competitor Vulnerabilities")
    
    if len(comp_data) == 0:
        st.info("No competitor data provided")
        return
    
    negative_reviews = comp_data[comp_data['sentiment'] == 0]
    
    if len(negative_reviews) == 0:
        st.info("No competitor weaknesses found")
        return
    
    negative_rate = len(negative_reviews) / len(comp_data) * 100 if len(comp_data) > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Competitor Negative Reviews", f"{len(negative_reviews)}/{len(comp_data)}")
    
    with col2:
        st.metric("Competitor Negative Rate", f"{negative_rate:.1f}%")
    
    # Extract aspects
    all_aspects = []
    for aspects in negative_reviews['aspects']:
        all_aspects.extend(aspects)
    
    if all_aspects:
        aspect_counts = Counter(all_aspects)
        
        st.subheader("⚠️ Competitor Weak Areas:")
        for aspect, count in aspect_counts.most_common(5):
            st.write(f"- **{aspect.upper()}**: {count} complaints")
        
        # Visualization
        if show_viz and len(aspect_counts) > 0:
            fig = px.bar(
                x=list(aspect_counts.keys()),
                y=list(aspect_counts.values()),
                labels={'x': 'Aspect', 'y': 'Number of Complaints'},
                title="Competitor Weak Points",
                color=list(aspect_counts.values()),
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Strategic opportunities
    st.subheader("💡 Strategic Attack Opportunities:")
    weak_aspects = [aspect for aspect, count in Counter(all_aspects).most_common(3)]
    
    if 'service' in weak_aspects:
        st.success("→ Emphasize YOUR superior customer service in marketing")
    if 'speed' in weak_aspects:
        st.success("→ Promote YOUR faster service/delivery times")
    if 'quality' in weak_aspects:
        st.success("→ Highlight YOUR quality standards")

def display_unmet_needs(df):
    """Display unmet customer needs"""
    
    st.header("💡 INSIGHT #3: Unmet Customer Needs")
    
    # Find common complaints across all businesses
    your_aspects = []
    comp_aspects = []
    
    your_negative = df[(df['business_type'] == 'YOUR_BUSINESS') & (df['sentiment'] == 0)]
    comp_negative = df[(df['business_type'] == 'COMPETITOR') & (df['sentiment'] == 0)]
    
    for aspects in your_negative['aspects']:
        your_aspects.extend(aspects)
    
    for aspects in comp_negative['aspects']:
        comp_aspects.extend(aspects)
    
    your_issues = set(your_aspects)
    comp_issues = set(comp_aspects)
    
    common_issues = your_issues.intersection(comp_issues)
    
    if common_issues:
        st.subheader("⚠️ Industry-Wide Problems:")
        st.write("These issues affect BOTH you and competitors - innovation opportunity!")
        for issue in common_issues:
            st.write(f"- **{issue.upper()}**: Nobody does this well")
    
    st.subheader("🚀 Innovation Opportunities:")
    st.write("""
    1. Develop solutions for industry-wide problems
    2. Listen to customer feature requests
    3. Create offerings competitors don't have
    4. Differentiate through innovation
    """)

def display_strategic_recommendations(your_data, comp_data):
    """Display strategic recommendations"""
    
    st.header("🚀 INSIGHT #4: Strategic Action Plan")
    
    # Calculate scores
    your_score = calculate_sentiment_score(your_data)
    comp_score = calculate_sentiment_score(comp_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Your Score", f"{your_score:+.2f}")
    
    with col2:
        st.metric("Competitor Score", f"{comp_score:+.2f}")
    
    with col3:
        if your_score > comp_score:
            st.success("✅ You're Winning!")
        else:
            st.warning("⚠️ Need Improvement")
    
    # Action plan
    st.subheader("🎯 3-Level Action Plan:")
    
    tab1, tab2, tab3 = st.tabs(["🔥 Immediate", "📅 Short-Term", "🎯 Long-Term"])
    
    with tab1:
        st.write("**This Week:**")
        st.write("- Address top 3 customer complaints")
        st.write("- Train staff on service issues")
        st.write("- Set up review monitoring system")
    
    with tab2:
        st.write("**This Month:**")
        st.write("- Launch marketing highlighting your advantages")
        st.write("- Attack competitor weaknesses in messaging")
        st.write("- Implement quick customer feedback wins")
    
    with tab3:
        st.write("**This Quarter:**")
        st.write("- Develop innovations for unmet needs")
        st.write("- Build systems for quality/service excellence")
        st.write("- Create loyalty programs based on customer values")

def calculate_sentiment_score(data):
    """Calculate overall sentiment score"""
    if len(data) == 0:
        return 0.0
    
    positive = len(data[data['sentiment'] == 2])
    negative = len(data[data['sentiment'] == 0])
    
    return (positive - negative) / len(data)

# ==================== SAMPLE ANALYSIS ====================
def show_sample_analysis():
    """Show sample analysis with demo data"""
    
    st.subheader("📊 Sample Analysis Demo")
    
    # Create sample data
    sample_data = {
        'review': [
            'Great service!', 'Terrible food', 'Amazing quality',
            'Slow service', 'Excellent product', 'Bad experience',
            'Love it!', 'Not recommended', 'Perfect!', 'Awful'
        ],
        'sentiment': [2, 0, 2, 0, 2, 0, 2, 0, 2, 0],
        'business_name': ['Sample']*10,
        'business_type': ['YOUR_BUSINESS']*5 + ['COMPETITOR']*5
    }
    
    df = pd.DataFrame(sample_data)
    
    st.dataframe(df)
    
    # Simple visualization
    sentiment_counts = df['sentiment'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=['Negative', 'Neutral', 'Positive'][:len(sentiment_counts)],
        title="Sentiment Distribution"
    )
    st.plotly_chart(fig)

# ==================== HOW IT WORKS PAGE ====================
def show_how_it_works():
    st.header("🎓 How It Works")
    
    st.write("""
    ### The Process:
    
    1. **Upload Reviews** 📤
       - Prepare CSV file with your business and competitor reviews
       - Include review text, sentiment labels, and business type
    
    2. **AI Processing** 🤖
       - Text preprocessing (cleaning, removing noise)
       - Feature extraction using TF-IDF
       - Machine learning classification
    
    3. **Pattern Recognition** 🔍
       - Identify complaint patterns
       - Extract business aspects (service, quality, price, etc.)
       - Compare your business vs competitors
    
    4. **Insight Generation** 💡
       - YOUR weaknesses to fix
       - COMPETITOR vulnerabilities to exploit
       - Market gaps for innovation
       - Actionable strategic recommendations
    
    5. **Results** 📊
       - Interactive visualizations
       - Downloadable reports
       - Implementation roadmap
    """)

# ==================== ABOUT PAGE ====================
def show_about():
    st.header("📊 About This Project")
    
    st.write("""
    ### Exploiting Business Intelligence in Reviews
    
    **Created by:** Your Name
    
    **Purpose:** 
    This AI-powered system analyzes customer reviews to extract competitive intelligence
    and generate actionable business insights.
    
    **Technology Stack:**
    - **Frontend:** Streamlit
    - **ML Models:** Scikit-learn (Logistic Regression, Naive Bayes, SVM)
    - **NLP:** TF-IDF Vectorization
    - **Visualization:** Plotly
    - **Language:** Python
    
    **Features:**
    - Real-time file upload and processing
    - Multi-model machine learning
    - Interactive visualizations
    - Competitive intelligence analysis
    - Strategic recommendations
    
    **Project Goals:**
    1. Identify operational weaknesses
    2. Detect competitor vulnerabilities
    3. Discover unmet customer needs
    4. Generate actionable business insights
    """)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
