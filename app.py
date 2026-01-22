import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Business Review Analyzer",
    page_icon="📊",
    layout="wide"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# ==================== APP TITLE ====================
st.markdown('<h1 class="main-header">📈 Business Review Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Capstone Project - Text Analytics & Business Intelligence</h2>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload Reviews CSV",
        type=['csv', 'xlsx'],
        help="Upload a CSV file with columns: Review, Rating, Brand, Date"
    )
    
    st.markdown("---")
    
    # Analysis options
    st.subheader("Analysis Settings")
    show_raw_data = st.checkbox("Show Raw Data", value=True)
    show_charts = st.checkbox("Show Charts", value=True)
    auto_calculate = st.checkbox("Auto Calculate Metrics", value=True)
    
    st.markdown("---")
    
    # Sample data
    if st.button("📊 Load Sample Data"):
        st.session_state.sample_loaded = True
        st.success("Sample data loaded! Check main area.")
    
    st.markdown("---")
    st.caption("Developed by: Your Name")
    st.caption("Course: Text Analytics Capstone")

# ==================== SAMPLE DATA ====================
def create_sample_data():
    return pd.DataFrame({
        'Review': [
            'Excellent camera quality and battery life',
            'Screen is amazing but too expensive',
            'Good value for money, recommended',
            'Customer service needs improvement',
            'Fast delivery and good packaging',
            'Product stopped working after 1 month',
            'Best purchase of the year',
            'Too heavy and bulky design',
            'User interface is very intuitive',
            'Network connectivity issues frequently'
        ],
        'Rating': [5, 3, 4, 2, 4, 1, 5, 2, 4, 2],
        'Brand': ['Apple', 'Samsung', 'Xiaomi', 'Apple', 'OnePlus', 
                 'Samsung', 'Apple', 'Samsung', 'Xiaomi', 'OnePlus'],
        'Date': pd.date_range('2024-01-01', periods=10).strftime('%Y-%m-%d'),
        'Category': ['Electronics'] * 10
    })

# ==================== MAIN APP LOGIC ====================
def main():
    # Initialize session state
    if 'sample_loaded' not in st.session_state:
        st.session_state.sample_loaded = False
    
    # Check for uploaded file or sample data
    if uploaded_file is not None:
        # Load uploaded file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            data_source = "Uploaded File"
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
    elif st.session_state.sample_loaded:
        # Load sample data
        df = create_sample_data()
        data_source = "Sample Data"
    else:
        # Show welcome screen
        show_welcome_screen()
        return
    
    # ==================== DISPLAY SUCCESS ====================
    st.success(f"✅ **{data_source} loaded successfully!**")
    st.info(f"**Total Reviews:** {len(df)} | **Data Period:** {df['Date'].min()} to {df['Date'].max()}")
    
    # ==================== METRICS ROW ====================
    st.markdown("## 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Reviews", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_rating = df['Rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        brands = df['Brand'].nunique()
        st.metric("Unique Brands", brands)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        positive = (df['Rating'] >= 4).sum()
        positive_pct = (positive / len(df)) * 100
        st.metric("Positive Reviews", f"{positive_pct:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== TABS FOR DIFFERENT ANALYSES ====================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Data Overview", 
        "📈 Visual Analysis", 
        "🏆 Brand Comparison", 
        "💡 Business Insights"
    ])
    
    with tab1:
        # Data Overview Tab
        st.subheader("Raw Data Preview")
        
        if show_raw_data:
            st.dataframe(df, use_container_width=True)
        
        # Data statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Statistics")
            stats_df = df.describe().round(2)
            st.dataframe(stats_df)
        
        with col2:
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.notnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info)
    
    with tab2:
        # Visual Analysis Tab
        if show_charts:
            st.subheader("Visual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Rating distribution
                st.markdown("**Rating Distribution**")
                rating_counts = df['Rating'].value_counts().sort_index()
                fig1 = px.bar(
                    x=rating_counts.index, 
                    y=rating_counts.values,
                    labels={'x': 'Rating', 'y': 'Count'},
                    color=rating_counts.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Sentiment pie chart
                st.markdown("**Sentiment Analysis**")
                df['Sentiment'] = df['Rating'].apply(
                    lambda x: 'Positive' if x >= 4 else ('Negative' if x <= 2 else 'Neutral')
                )
                sentiment_counts = df['Sentiment'].value_counts()
                fig2 = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    hole=0.3,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Time trend
            st.markdown("**Reviews Over Time**")
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                daily_counts = df.groupby(df['Date'].dt.date).size().reset_index(name='Count')
                fig3 = px.line(
                    daily_counts, 
                    x='Date', 
                    y='Count',
                    title='Number of Reviews Over Time'
                )
                st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        # Brand Comparison Tab
        st.subheader("Brand Performance Analysis")
        
        # Brand comparison table
        brand_stats = df.groupby('Brand').agg({
            'Rating': ['count', 'mean', 'min', 'max'],
            'Review': 'count'
        }).round(2)
        
        brand_stats.columns = ['Review Count', 'Avg Rating', 'Min Rating', 'Max Rating', 'Total Reviews']
        st.dataframe(brand_stats, use_container_width=True)
        
        # Brand comparison chart
        st.subheader("Brand Rating Comparison")
        fig4 = px.box(df, x='Brand', y='Rating', points="all")
        st.plotly_chart(fig4, use_container_width=True)
    
    with tab4:
        # Business Insights Tab
        st.subheader("Actionable Business Insights")
        
        # Generate insights
        insights = []
        
        # Insight 1: Top performing brand
        top_brand = df.groupby('Brand')['Rating'].mean().idxmax()
        top_rating = df.groupby('Brand')['Rating'].mean().max()
        insights.append(f"🏆 **{top_brand}** is the top performing brand with average rating of **{top_rating:.2f}**")
        
        # Insight 2: Most common complaint
        low_ratings = df[df['Rating'] <= 2]
        if len(low_ratings) > 0:
            common_brand = low_ratings['Brand'].mode()[0]
            insights.append(f"⚠️ **{common_brand}** has the most negative reviews - needs quality improvement")
        
        # Insight 3: Rating trend
        if 'Date' in df.columns:
            recent_avg = df.tail(5)['Rating'].mean()
            if recent_avg > avg_rating:
                insights.append("📈 **Positive trend**: Recent reviews show improvement")
            else:
                insights.append("📉 **Warning**: Recent reviews show decline in ratings")
        
        # Display insights
        for insight in insights:
            st.info(insight)
        
        # Recommendations
        st.subheader("📋 Strategic Recommendations")
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("""
            **🎯 Product Development:**
            1. Analyze 1-2 star reviews for improvement areas
            2. Enhance features mentioned in positive reviews
            3. Address common complaints systematically
            """)
        
        with rec_col2:
            st.markdown("""
            **📈 Marketing Strategy:**
            1. Highlight strengths in advertising
            2. Target competitor weaknesses
            3. Use positive reviews as testimonials
            """)
        
        # Export section
        st.markdown("---")
        st.subheader("📤 Export Results")
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Raw Data (CSV)",
                data=csv,
                file_name="business_reviews_raw.csv",
                mime="text/csv"
            )
        
        with col2:
            insights_text = "\n".join(insights)
            st.download_button(
                label="📥 Download Insights (TXT)",
                data=insights_text,
                file_name="business_insights.txt",
                mime="text/plain"
            )

# ==================== WELCOME SCREEN ====================
def show_welcome_screen():
    st.markdown("## 🎯 Welcome to Business Review Analyzer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 How to Use This Application:
        
        1. **Prepare your data** in CSV format with these columns:
           - `Review`: Customer review text
           - `Rating`: Numeric rating (1-5)
           - `Brand`: Brand name
           - `Date`: Review date (YYYY-MM-DD)
        
        2. **Upload your file** using the sidebar
        
        3. **Get instant insights** including:
           - Sentiment analysis
           - Brand performance comparison
           - Trend analysis
           - Actionable business recommendations
        
        4. **Export results** for presentations and reports
        
        ### 🎓 For Capstone Project:
        - This application demonstrates real-time text analytics
        - Shows practical business application of NLP
        - Provides competitive intelligence from reviews
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/1534/1534938.png", width=150)
        st.markdown("### Sample CSV Format:")
        st.code("""Review,Rating,Brand,Date
Great product!,5,Apple,2024-01-15
Not satisfied,2,Samsung,2024-01-16
Good value,4,Xiaomi,2024-01-17""")
    
    st.markdown("---")
    
    # Quick start options
    st.subheader("🚀 Quick Start")
    
    quick_col1, quick_col2 = st.columns(2)
    
    with quick_col1:
        st.info("""
        **Option 1: Upload Your Data**
        - Use the sidebar to upload CSV
        - Get analysis in seconds
        """)
    
    with quick_col2:
        st.info("""
        **Option 2: Try Sample Data**
        - Click 'Load Sample Data' in sidebar
        - Explore features with demo data
        """)
    
    st.markdown("---")
    st.success("**Ready to begin?** Use the sidebar to upload data or load sample data!")

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p><strong>🎓 Capstone Project: Text Analytics</strong></p>
    <p><em>Exploiting Business Intelligence from Customer Reviews</em></p>
    <p>Developed with ❤️ using Streamlit | Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
