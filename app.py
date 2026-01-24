import streamlit as st
import pandas as pd
import plotly.express as px
import random
from datetime import datetime, timedelta

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Dynamic Business Exploitation Analyzer",
    page_icon="🦅",
    layout="wide"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .attack-strategy {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .positive-insight {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    
    # Initialize session state
    if 'brand_name' not in st.session_state:
        st.session_state.brand_name = "Samsung"
    if 'industry' not in st.session_state:
        st.session_state.industry = "Electronics"
    
    # Brand input with on_change
    brand_input = st.text_input(
        "Enter ANY brand name:",
        value=st.session_state.brand_name,
        key="brand_input_key"
    )
    
    # Update session state when input changes
    if brand_input != st.session_state.brand_name:
        st.session_state.brand_name = brand_input
    
    # Industry selection
    industry = st.selectbox(
        "Select Industry:",
        ["Electronics", "Food & Beverage", "Automotive", "Fashion", 
         "Retail", "Technology", "Healthcare", "Other"],
        index=0,
        key="industry_key"
    )
    
    # Update industry in session state
    if industry != st.session_state.industry:
        st.session_state.industry = industry
    
    # Analysis type
    analysis_type = st.radio(
        "Analysis Depth:",
        ["Quick Scan", "Detailed Analysis", "Strategic Deep Dive"],
        key="analysis_type_key"
    )
    
    # Data source
    data_source = st.radio(
        "Data Source:",
        ["🌐 Simulated Live Data", "💾 Sample Dataset", "📁 Upload CSV"],
        key="data_source_key"
    )
    
    # File uploader
    uploaded_file = None
    if data_source == "📁 Upload CSV":
        uploaded_file = st.file_uploader("Upload your reviews CSV", type=['csv'], key="file_uploader_key")
    
    st.markdown("---")
    
    # Quick analyze buttons
    st.subheader("Quick Analyze:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📱 Samsung", key="btn_samsung"):
            st.session_state.brand_name = "Samsung"
            st.session_state.industry = "Electronics"
            st.rerun()
    with col2:
        if st.button("🍕 Dominos", key="btn_dominos"):
            st.session_state.brand_name = "Dominos"
            st.session_state.industry = "Food & Beverage"
            st.rerun()
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🚗 Tesla", key="btn_tesla"):
            st.session_state.brand_name = "Tesla"
            st.session_state.industry = "Automotive"
            st.rerun()
    with col4:
        if st.button("👟 Nike", key="btn_nike"):
            st.session_state.brand_name = "Nike"
            st.session_state.industry = "Fashion"
            st.rerun()
    
    st.markdown("---")
    st.caption("🎓 Capstone Project: Exploiting Business Intelligence from Reviews")

# ==================== DYNAMIC DATA GENERATION ====================
def generate_dynamic_data(brand, industry, num_reviews=50):
    """Generate realistic review data for ANY brand"""
    
    # Industry-specific review templates
    industry_templates = {
        "Electronics": [
            f"The {brand} phone's battery life is amazing, lasts all day",
            f"{brand} camera quality could be better in low light",
            f"Love the {brand} display, colors are so vibrant",
            f"{brand} customer service needs major improvement",
            f"Great value for money from {brand} during sale season",
            f"{brand} software updates are too slow compared to competitors",
            f"Facing heating issues with my new {brand} device",
            f"{brand} build quality feels premium and durable",
            f"Charger not included with {brand} device - disappointing",
            f"Best {brand} product I've purchased in years!"
        ],
        "Food & Beverage": [
            f"{brand} pizza always arrives hot and fresh",
            f"{brand} coffee tastes bitter, not like before",
            f"Love the ambiance at {brand} outlets",
            f"{brand} delivery is always faster than promised",
            f"Prices at {brand} are getting too expensive",
            f"{brand} customer service was rude on phone",
            f"Best quality ingredients at {brand}",
            f"{brand} app makes ordering so convenient",
            f"Portion sizes at {brand} have reduced recently",
            f"Would definitely recommend {brand} to friends"
        ],
        "Automotive": [
            f"{brand} mileage is better than advertised",
            f"{brand} service center experience was terrible",
            f"Love the safety features in {brand} cars",
            f"{brand} waiting period is too long for delivery",
            f"Best driving experience with {brand}",
            f"{brand} resale value is excellent",
            f"Facing issues with {brand} infotainment system",
            f"{brand} design looks stunning on road",
            f"{brand} maintenance costs are too high",
            f"Most comfortable seats in {brand} cars"
        ],
        "Fashion": [
            f"{brand} shoes are extremely comfortable for all-day wear",
            f"{brand} clothing sizes run too small",
            f"Love the latest {brand} collection design",
            f"{brand} quality has deteriorated over years",
            f"Best fitting jeans from {brand}",
            f"{brand} prices are too high for the quality",
            f"{brand} customer support helped with exchange quickly",
            f"Colors fade after few washes with {brand} clothes",
            f"{brand} brand value makes it worth the price",
            f"Would buy from {brand} again definitely"
        ]
    }
    
    # Get templates for selected industry or use general templates
    templates = industry_templates.get(industry, [
        f"{brand} product quality meets expectations",
        f"{brand} customer service could be better",
        f"Good value from {brand} products",
        f"Facing issues with {brand} recently",
        f"Would recommend {brand} to others"
    ])
    
    # Generate reviews
    reviews = []
    start_date = datetime.now() - timedelta(days=90)
    
    for i in range(num_reviews):
        review_date = start_date + timedelta(days=random.randint(0, 90))
        
        # Base rating based on sentiment words in template
        template = random.choice(templates)
        base_rating = 3  # Default neutral
        
        # Adjust rating based on sentiment words
        positive_words = ['amazing', 'love', 'great', 'best', 'excellent', 'comfortable', 'fresh', 'hot']
        negative_words = ['could be better', 'needs improvement', 'disappointing', 'terrible', 'bitter', 'rude', 'facing issues']
        
        if any(word in template.lower() for word in positive_words):
            base_rating = random.randint(4, 5)
        elif any(word in template.lower() for word in negative_words):
            base_rating = random.randint(1, 2)
        else:
            base_rating = random.randint(3, 4)
        
        # Add some random variation
        final_rating = max(1, min(5, base_rating + random.randint(-1, 1)))
        
        reviews.append({
            "review": template,
            "rating": final_rating,
            "date": review_date.strftime("%Y-%m-%d"),
            "source": random.choice(["Amazon", "Flipkart", "Twitter", "Google Reviews", "YouTube", "Reddit"]),
            "sentiment": "Positive" if final_rating >= 4 else ("Negative" if final_rating <= 2 else "Neutral")
        })
    
    return pd.DataFrame(reviews)

# ==================== BUSINESS EXPLOITATION ANALYSIS ====================
def analyze_for_exploitation(df, brand, industry):
    """Generate business exploitation insights"""
    
    insights = {
        "weaknesses": [],
        "strengths": [],
        "attack_opportunities": [],
        "defense_strategies": [],
        "market_gaps": []
    }
    
    # Calculate basic metrics
    avg_rating = df['rating'].mean()
    total_reviews = len(df)
    positive_pct = (len(df[df['rating'] >= 4]) / total_reviews) * 100
    negative_pct = (len(df[df['rating'] <= 2]) / total_reviews) * 100
    
    # Analyze text for specific aspects
    all_reviews_text = ' '.join(df['review'].str.lower())
    
    # Common business aspects to check
    aspects = {
        "price": ['expensive', 'cheap', 'price', 'cost', 'value', 'worth', 'affordable'],
        "quality": ['quality', 'durable', 'breaks', 'issue', 'problem', 'defect'],
        "service": ['service', 'support', 'customer', 'help', 'rude', 'polite', 'responsive'],
        "delivery": ['delivery', 'shipping', 'fast', 'slow', 'late', 'on time'],
        "features": ['feature', 'camera', 'battery', 'display', 'performance', 'speed']
    }
    
    # Find weaknesses (negative mentions)
    negative_reviews = df[df['rating'] <= 2]
    if len(negative_reviews) > 0:
        negative_text = ' '.join(negative_reviews['review'].str.lower())
        
        for aspect, keywords in aspects.items():
            keyword_count = sum(1 for keyword in keywords if keyword in negative_text)
            if keyword_count > 2:  # If mentioned multiple times
                insights["weaknesses"].append(f"**{aspect.capitalize()}**: {keyword_count} negative mentions")
                
                # Attack opportunity based on weakness
                if aspect == "price":
                    insights["attack_opportunities"].append(f"Attack {brand} on pricing - offer 20% better value")
                elif aspect == "service":
                    insights["attack_opportunities"].append(f"Highlight your superior customer service vs {brand}")
                elif aspect == "quality":
                    insights["attack_opportunities"].append(f"Showcase your quality assurance process vs {brand}")
    
    # Find strengths (positive mentions)
    positive_reviews = df[df['rating'] >= 4]
    if len(positive_reviews) > 0:
        positive_text = ' '.join(positive_reviews['review'].str.lower())
        
        for aspect, keywords in aspects.items():
            keyword_count = sum(1 for keyword in keywords if keyword in positive_text)
            if keyword_count > 2:
                insights["strengths"].append(f"**{aspect.capitalize()}**: {keyword_count} positive mentions")
                insights["defense_strategies"].append(f"Match {brand}'s strength in {aspect}")
    
    # Market gaps (wish statements)
    wish_keywords = ['wish', 'hope', 'should have', 'if only', 'would be better if']
    wish_reviews = [r for r in df['review'] if any(w in r.lower() for w in wish_keywords)]
    if wish_reviews:
        insights["market_gaps"].append(f"{len(wish_reviews)} customers expressed unmet needs")
        insights["market_gaps"].append(f"Sample wish: \"{wish_reviews[0][:100]}...\"")
    
    return insights, avg_rating, positive_pct, negative_pct

# ==================== MAIN APP LOGIC ====================
def main():
    # Get values from session state
    brand_name = st.session_state.brand_name
    industry = st.session_state.industry
    
    # ==================== APP TITLE ====================
    st.markdown('<h1 class="main-title">🦅 Dynamic Business Exploitation Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Analyze ANY Brand & Generate Competitive Attack Strategies</p>', unsafe_allow_html=True)
    
    # ==================== CURRENT TARGET DISPLAY ====================
    st.markdown("### 🎯 Current Analysis Target")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.info(f"**Brand:** {brand_name}")
    
    with col2:
        st.info(f"**Industry:** {industry}")
    
    with col3:
        if st.button("🔄 Update Settings", type="secondary"):
            st.rerun()
    
    st.markdown("---")
    
    # ==================== ANALYSIS BUTTON ====================
    analyze_clicked = st.button(
        f"🚀 ANALYZE {brand_name.upper()}", 
        type="primary", 
        use_container_width=True,
        key="analyze_button"
    )
    
    # Get uploaded file from session
    uploaded_file = None
    if st.session_state.get('data_source_key') == "📁 Upload CSV":
        uploaded_file = st.session_state.get('file_uploader_key')
    
    if analyze_clicked:
        
        # Generate or load data
        if st.session_state.get('data_source_key') == "📁 Upload CSV" and uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded {len(df)} reviews for {brand_name}")
        else:
            with st.spinner(f"Generating realistic data for {brand_name} in {industry}..."):
                df = generate_dynamic_data(brand_name, industry, 
                                         num_reviews=30 if analysis_type == "Quick Scan" else 50)
                st.success(f"✅ Generated {len(df)} realistic reviews for {brand_name}")
        
        # Perform analysis
        with st.spinner("Analyzing for business exploitation opportunities..."):
            insights, avg_rating, positive_pct, negative_pct = analyze_for_exploitation(df, brand_name, industry)
        
        # ==================== DISPLAY RESULTS ====================
        
        # Key Metrics
        st.markdown("## 📊 Brand Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Rating", f"{avg_rating:.2f}/5", 
                     delta="High" if avg_rating >= 4.0 else ("Low" if avg_rating <= 2.5 else "Average"))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Positive %", f"{positive_pct:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Negative %", f"{negative_pct:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Reviews", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Reviews Data", "🎯 Exploitation Analysis", "⚔️ Attack Strategy", 
            "🛡️ Defense Strategy", "📈 Visual Analytics"
        ])
        
        with tab1:
            # Data tab
            st.subheader(f"Sample Reviews for {brand_name}")
            st.dataframe(df[['review', 'rating', 'date', 'source']].head(10), use_container_width=True)
            
            # Rating distribution
            st.subheader("Rating Distribution")
            rating_counts = df['rating'].value_counts().sort_index()
            fig1 = px.bar(
                x=rating_counts.index, 
                y=rating_counts.values,
                labels={'x': 'Rating (1-5)', 'y': 'Number of Reviews'},
                title=f"{brand_name} Customer Ratings Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            # Exploitation analysis tab
            st.subheader("🔍 Weaknesses Identified")
            if insights["weaknesses"]:
                for weakness in insights["weaknesses"]:
                    st.markdown(f'<div class="attack-strategy">{weakness}</div>', unsafe_allow_html=True)
            else:
                st.info(f"No major weaknesses identified for {brand_name}")
            
            st.subheader("✨ Strengths Identified")
            if insights["strengths"]:
                for strength in insights["strengths"]:
                    st.markdown(f'<div class="positive-insight">{strength}</div>', unsafe_allow_html=True)
            else:
                st.info(f"No major strengths highlighted for {brand_name}")
            
            st.subheader("📈 Market Gaps")
            if insights["market_gaps"]:
                for gap in insights["market_gaps"]:
                    st.write(gap)
            else:
                st.info("No clear market gaps identified")
        
        with tab3:
            # Attack strategy tab
            st.subheader("⚔️ Competitive Attack Opportunities")
            
            if insights["attack_opportunities"]:
                st.warning(f"**PRIMARY TARGET: {brand_name}**")
                
                for i, opportunity in enumerate(insights["attack_opportunities"], 1):
                    st.markdown(f"**{i}. {opportunity}**")
                
                # Action plan
                st.subheader("📋 30-Day Attack Plan")
                
                attack_plan = [
                    f"Week 1: Social media campaign targeting {brand_name}'s weaknesses",
                    f"Week 2: Comparative advertising highlighting your advantages over {brand_name}",
                    f"Week 3: Special offers for {brand_name} customers looking to switch",
                    f"Week 4: PR campaign showcasing customer success stories vs {brand_name}"
                ]
                
                for item in attack_plan:
                    st.write(f"• {item}")
                
                # ROI estimate
                if negative_pct > 20:
                    st.success(f"💰 **Potential Impact:** Targeting {negative_pct:.0f}% dissatisfied customers could capture significant market share")
            else:
                st.info(f"{brand_name} appears strong. Consider differentiation strategy instead of direct attack.")
        
        with tab4:
            # Defense strategy tab
            st.subheader("🛡️ Defense & Improvement Strategy")
            
            if insights["weaknesses"]:
                st.warning(f"**AREAS NEEDING DEFENSE for {brand_name}:**")
                for weakness in insights["weaknesses"][:3]:  # Top 3 weaknesses
                    st.write(f"• {weakness.replace('**', '')}")
                
                st.subheader("📋 Improvement Roadmap")
                improvement_plan = [
                    "Address top customer complaints within 60 days",
                    "Enhance customer service training",
                    "Implement quality control measures",
                    "Launch customer satisfaction survey"
                ]
                
                for item in improvement_plan:
                    st.write(f"• {item}")
            else:
                st.success(f"✅ {brand_name} has good defensive position")
                st.write("**Maintenance Strategy:**")
                st.write("• Continue monitoring customer feedback")
                st.write("• Maintain quality standards")
                st.write("• Innovate to stay ahead of competitors")
            
            if insights["defense_strategies"]:
                st.subheader("💪 Strength Reinforcement")
                for strategy in insights["defense_strategies"]:
                    st.write(f"• {strategy}")
        
        with tab5:
            # Visual analytics tab
            st.subheader("📈 Sentiment Trend Over Time")
            
            # Convert date and analyze trends
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M').astype(str)
            
            monthly_avg = df.groupby('month')['rating'].mean().reset_index()
            
            fig2 = px.line(
                monthly_avg, 
                x='month', 
                y='rating',
                title=f"{brand_name} Monthly Average Rating Trend",
                markers=True
            )
            fig2.update_layout(yaxis_range=[1, 5])
            st.plotly_chart(fig2, use_container_width=True)
            
            # Source distribution
            st.subheader("Review Sources Distribution")
            source_counts = df['source'].value_counts()
            fig3 = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Where Reviews Come From"
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Download report
        st.markdown("---")
        st.subheader("📥 Download Analysis Report")
        
        # Generate report content
        report_content = f"""
        BUSINESS EXPLOITATION ANALYSIS REPORT
        =====================================
        Brand Analyzed: {brand_name}
        Industry: {industry}
        Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        SUMMARY METRICS:
        - Average Rating: {avg_rating:.2f}/5
        - Positive Reviews: {positive_pct:.1f}%
        - Negative Reviews: {negative_pct:.1f}%
        - Total Reviews Analyzed: {len(df)}
        
        KEY WEAKNESSES:
        {chr(10).join(['- ' + w.replace('**', '') for w in insights['weaknesses'][:3]]) if insights['weaknesses'] else '- None significant'}
        
        KEY STRENGTHS:
        {chr(10).join(['- ' + s.replace('**', '') for s in insights['strengths'][:3]]) if insights['strengths'] else '- None highlighted'}
        
        RECOMMENDED ATTACK STRATEGIES:
        {chr(10).join(['- ' + a for a in insights['attack_opportunities'][:3]]) if insights['attack_opportunities'] else '- Differentiation recommended over direct attack'}
        
        ACTION PLAN:
        1. Address top customer complaints
        2. Launch targeted marketing campaign
        3. Monitor competitor responses
        4. Measure impact monthly
        
        Generated by: Dynamic Business Exploitation Analyzer
        Capstone Project - Text Analytics
        """
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download Text Report",
                data=report_content,
                file_name=f"{brand_name}_exploitation_report.txt",
                mime="text/plain",
                key="download_report"
            )
        
        with col2:
            # Convert DataFrame to CSV for download
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Review Data (CSV)",
                data=csv,
                file_name=f"{brand_name}_reviews_data.csv",
                mime="text/csv",
                key="download_data"
            )
    
    else:
        # Welcome screen
        st.markdown("## 🎯 Welcome to Dynamic Business Exploitation Analyzer")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### How This System Works:
            
            1. **Enter ANY brand name** in the sidebar
            2. **Select the industry** for context-aware analysis
            3. **Choose analysis depth** (Quick, Detailed, Strategic)
            4. **Click 'ANALYZE' button** above
            
            ### 📊 You'll Get:
            - Competitor weaknesses to exploit
            - Attack strategies for market capture  
            - Defense strategies for your brand
            - Market gap identification
            - Downloadable business reports
            
            ### 🎓 Capstone Project Value:
            - Demonstrates REAL business application of text analytics
            - Shows how customer reviews can be weaponized for competitive advantage
            - Provides actionable business intelligence from unstructured data
            """)
        
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/1534/1534938.png", width=150)
            st.markdown("### Quick Start:")
            st.write("1. Enter brand name")
            st.write("2. Select industry")
            st.write("3. Click ANALYZE button")
        
        st.markdown("---")
        st.warning(f"⚠️ **Ready to analyze {brand_name}?** Click the 'ANALYZE' button above!")

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p><strong>🎓 Capstone Project: Text Analytics & Business Intelligence</strong></p>
    <p><em>"Exploiting Competitive Advantage from Customer Reviews"</em></p>
    <p>Dynamic Brand Analysis System | Works for ANY Industry | Real Business Strategies</p>
</div>
""", unsafe_allow_html=True)
