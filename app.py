import streamlit as st
import pandas as pd
import plotly.express as px
import random
from datetime import datetime, timedelta

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Business Exploitation Engine",
    page_icon="⚔️",
    layout="wide"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        color: #DC2626;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-title {
        font-size: 1.4rem;
        color: #7F1D1D;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .attack-card {
        background: linear-gradient(135deg, #DC2626 0%, #7F1D1D 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(220, 38, 38, 0.3);
    }
    .weakness-highlight {
        background-color: #FEE2E2;
        border-left: 5px solid #DC2626;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .strategy-box {
        background-color: #FEF3C7;
        border: 2px solid #D97706;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .exploit-tag {
        background-color: #DC2626;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚔️ Target Selection")
    
    # Initialize session state
    if 'target_brand' not in st.session_state:
        st.session_state.target_brand = "Samsung"
    if 'attacker_brand' not in st.session_state:
        st.session_state.attacker_brand = "Your Brand"
    if 'industry' not in st.session_state:
        st.session_state.industry = "Electronics"
    
    # TARGET brand (to attack)
    st.subheader("🎯 Target to Attack:")
    target_input = st.text_input(
        "Enter competitor brand to analyze:",
        value=st.session_state.target_brand,
        key="target_input"
    )
    
    if target_input != st.session_state.target_brand:
        st.session_state.target_brand = target_input
    
    # ATTACKER brand (your brand)
    st.subheader("🛡️ Your Brand:")
    attacker_input = st.text_input(
        "Enter your brand name:",
        value=st.session_state.attacker_brand,
        key="attacker_input"
    )
    
    if attacker_input != st.session_state.attacker_brand:
        st.session_state.attacker_brand = attacker_input
    
    # Industry
    industry = st.selectbox(
        "Industry:",
        ["Electronics", "Food & Beverage", "Automotive", "Fashion", 
         "Retail", "Technology", "Other"],
        index=0,
        key="industry_key"
    )
    
    if industry != st.session_state.industry:
        st.session_state.industry = industry
    
    # Data source (simplified)
    data_source = st.radio(
        "Data Source:",
        ["🌐 Generate Sample Data", "📁 Upload Your Data"]
    )
    
    uploaded_file = None
    if data_source == "📁 Upload Your Data":
        uploaded_file = st.file_uploader("Upload competitor reviews CSV", type=['csv'])
    
    st.markdown("---")
    
    # Quick targets
    st.subheader("Quick Targets:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📱 Attack Samsung"):
            st.session_state.target_brand = "Samsung"
            st.session_state.industry = "Electronics"
            st.rerun()
    with col2:
        if st.button("🍕 Attack Dominos"):
            st.session_state.target_brand = "Dominos"
            st.session_state.industry = "Food & Beverage"
            st.rerun()
    
    st.markdown("---")
    st.caption("⚔️ Capstone: Exploiting Business Intelligence from Reviews")

# ==================== DATA GENERATION ====================
def generate_attack_data(target_brand, industry):
    """Generate data for exploitation analysis"""
    
    # Industry-specific weaknesses
    industry_weaknesses = {
        "Electronics": [
            f"{target_brand} battery drains too fast",
            f"{target_brand} gets hot during gaming",
            f"{target_brand} software updates are slow",
            f"{target_brand} customer service is poor",
            f"{target_brand} price increased but features didn't",
            f"{target_brand} charger not included in box",
            f"{target_brand} camera struggles in low light",
            f"{target_brand} display scratches easily",
            f"{target_brand} face unlock doesn't work well",
            f"{target_brand} 5G connectivity issues"
        ],
        "Food & Beverage": [
            f"{target_brand} pizza arrives cold sometimes",
            f"{target_brand} prices increased recently",
            f"{target_brand} delivery is often late",
            f"{target_brand} portion sizes reduced",
            f"{target_brand} customer support rude",
            f"{target_brand} app has payment issues",
            f"{target_brand} food quality inconsistent",
            f"{target_brand} limited vegetarian options",
            f"{target_brand} waiting time too long",
            f"{target_brand} packaging not eco-friendly"
        ],
        "Automotive": [
            f"{target_brand} service center expensive",
            f"{target_brand} mileage less than advertised",
            f"{target_brand} waiting period too long",
            f"{target_brand} infotainment system laggy",
            f"{target_brand} spare parts costly",
            f"{target_brand} AC not powerful enough",
            f"{target_brand} suspension too stiff",
            f"{target_brand} resale value dropping",
            f"{target_brand} service appointments delayed",
            f"{target_brand} warranty claims rejected"
        ]
    }
    
    # Get weaknesses for industry
    weaknesses = industry_weaknesses.get(industry, [
        f"{target_brand} quality has decreased",
        f"{target_brand} customer service needs improvement",
        f"{target_brand} prices are too high",
        f"{target_brand} delivery takes too long"
    ])
    
    # Generate reviews (70% negative for exploitation analysis)
    reviews = []
    for i in range(40):
        is_negative = random.random() < 0.7  # 70% negative for attack analysis
        
        if is_negative:
            review = random.choice(weaknesses)
            rating = random.randint(1, 2)
            sentiment = "Negative"
        else:
            # Some positive reviews for balance
            positive_templates = [
                f"{target_brand} product works well",
                f"Happy with {target_brand} service",
                f"{target_brand} quality is good",
                f"Would recommend {target_brand}"
            ]
            review = random.choice(positive_templates)
            rating = random.randint(4, 5)
            sentiment = "Positive"
        
        reviews.append({
            "review": review,
            "rating": rating,
            "sentiment": sentiment,
            "source": random.choice(["Amazon", "Twitter", "Google", "Forum"]),
            "date": (datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d")
        })
    
    return pd.DataFrame(reviews)

# ==================== EXPLOITATION ANALYSIS ====================
def find_exploitation_opportunities(df, target_brand, attacker_brand, industry):
    """Find opportunities to exploit competitor weaknesses"""
    
    opportunities = {
        "critical_weaknesses": [],
        "marketing_attacks": [],
        "product_attacks": [],
        "price_attacks": [],
        "service_attacks": [],
        "urgent_actions": []
    }
    
    # Get negative reviews
    negative_df = df[df['sentiment'] == "Negative"]
    
    if len(negative_df) == 0:
        return opportunities
    
    # Analyze complaint patterns
    all_complaints = ' '.join(negative_df['review'].str.lower())
    
    # 1. PRICE EXPLOITATION
    price_keywords = ['expensive', 'price', 'cost', 'high', 'increased', 'costly']
    price_complaints = [c for c in price_keywords if c in all_complaints]
    if price_complaints:
        opportunities["price_attacks"].append(
            f"**💰 Price Attack:** {target_brand} seen as expensive"
        )
        opportunities["marketing_attacks"].append(
            f"Run ads: '{attacker_brand} offers same features at 20% lower price than {target_brand}'"
        )
        opportunities["urgent_actions"].append(
            f"Launch price comparison campaign against {target_brand}"
        )
    
    # 2. QUALITY EXPLOITATION
    quality_keywords = ['quality', 'poor', 'bad', 'decreased', 'scratches', 'issues']
    quality_complaints = [c for c in quality_keywords if c in all_complaints]
    if quality_complaints:
        opportunities["product_attacks"].append(
            f"**🎯 Quality Attack:** {target_brand} has quality issues"
        )
        opportunities["marketing_attacks"].append(
            f"Showcase your quality control vs {target_brand}'s failures"
        )
    
    # 3. SERVICE EXPLOITATION
    service_keywords = ['service', 'support', 'rude', 'slow', 'delayed', 'poor']
    service_complaints = [c for c in service_keywords if c in all_complaints]
    if service_complaints:
        opportunities["service_attacks"].append(
            f"**📞 Service Attack:** {target_brand} service complaints"
        )
        opportunities["urgent_actions"].append(
            f"Target {target_brand} customers with 'Better Service Guarantee'"
        )
    
    # 4. SPECIFIC WEAKNESSES
    # Find most frequent complaints
    from collections import Counter
    words = ' '.join(negative_df['review']).lower().split()
    common_words = Counter(words).most_common(10)
    
    for word, count in common_words:
        if count >= 3 and len(word) > 3:  # Significant complaints
            if word not in ['the', 'and', 'this', 'that', 'with', 'from']:
                opportunities["critical_weaknesses"].append(
                    f"'{word}' mentioned {count} times in complaints"
                )
    
    return opportunities

# ==================== MAIN APP ====================
def main():
    # Get session values
    target_brand = st.session_state.target_brand
    attacker_brand = st.session_state.attacker_brand
    industry = st.session_state.industry
    
    # ==================== TITLE ====================
    st.markdown('<h1 class="main-title">⚔️ BUSINESS EXPLOITATION ENGINE</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-title">Attack Plan: {attacker_brand} → {target_brand}</p>', unsafe_allow_html=True)
    
    # ==================== BATTLE DISPLAY ====================
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown(f'### 🛡️ {attacker_brand}')
        st.info("Your Brand")
    
    with col2:
        st.markdown("### ⚔️ VS")
        st.markdown('<div style="text-align: center; font-size: 2rem;">🎯</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'### 🎯 {target_brand}')
        st.warning("Target Competitor")
    
    st.markdown("---")
    
    # ==================== ATTACK BUTTON ====================
    attack_clicked = st.button(
        f"⚔️ GENERATE ATTACK PLAN AGAINST {target_brand.upper()}",
        type="primary",
        use_container_width=True,
        key="attack_button"
    )
    
    if attack_clicked:
        # Generate attack data
        with st.spinner(f"Analyzing {target_brand} weaknesses for exploitation..."):
            df = generate_attack_data(target_brand, industry)
            
            # Find exploitation opportunities
            opportunities = find_exploitation_opportunities(df, target_brand, attacker_brand, industry)
        
        st.success(f"✅ Found {len(df[df['sentiment']=='Negative'])} weaknesses in {target_brand}")
        
        # ==================== WEAKNESS DASHBOARD ====================
        st.markdown("## 🔍 Target Weakness Analysis")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        negative_count = len(df[df['sentiment'] == "Negative"])
        
        with col1:
            st.markdown('<div class="attack-card">', unsafe_allow_html=True)
            st.metric("Weaknesses Found", negative_count)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="attack-card">', unsafe_allow_html=True)
            weakness_percent = (negative_count / len(df)) * 100
            st.metric("Exploitation Potential", f"{weakness_percent:.0f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="attack-card">', unsafe_allow_html=True)
            st.metric("Avg Target Rating", f"{df['rating'].mean():.1f}/5")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ==================== EXPLOITATION STRATEGIES ====================
        st.markdown("## 🎯 Exploitation Strategies")
        
        # Critical Weaknesses
        if opportunities["critical_weaknesses"]:
            st.markdown("### ⚠️ Critical Weaknesses to Exploit")
            for weakness in opportunities["critical_weaknesses"][:5]:
                st.markdown(f'<div class="weakness-highlight">{weakness}</div>', unsafe_allow_html=True)
        
        # Marketing Attacks
        if opportunities["marketing_attacks"]:
            st.markdown("### 📢 Marketing Attack Campaigns")
            for attack in opportunities["marketing_attacks"]:
                st.markdown(f'<div class="strategy-box">{attack}</div>', unsafe_allow_html=True)
        
        # Price Attacks
        if opportunities["price_attacks"]:
            st.markdown("### 💰 Price-Based Attacks")
            for attack in opportunities["price_attacks"]:
                st.write(f"• {attack}")
        
        # Product Attacks  
        if opportunities["product_attacks"]:
            st.markdown("### 🎯 Product-Based Attacks")
            for attack in opportunities["product_attacks"]:
                st.write(f"• {attack}")
        
        # ==================== 30-DAY ATTACK PLAN ====================
        st.markdown("## 📅 30-Day Attack Implementation Plan")
        
        attack_plan = [
            f"**Week 1:** Social media blitz highlighting {target_brand}'s top 3 weaknesses",
            f"**Week 2:** Launch comparative ads '{attacker_brand} vs {target_brand}'",
            f"**Week 3:** Special offer for {target_brand} customers switching to {attacker_brand}",
            f"**Week 4:** PR campaign with case studies of successful switches from {target_brand}"
        ]
        
        for item in attack_plan:
            st.write(item)
        
        # ==================== SAMPLE COMPLAINTS ====================
        st.markdown("## 📝 Sample Customer Complaints (Use in Marketing)")
        
        negative_samples = df[df['sentiment'] == "Negative"].head(5)
        for idx, row in negative_samples.iterrows():
            st.write(f"**Complaint {idx+1}:** \"{row['review']}\"")
            st.markdown(f'<span class="exploit-tag">Use in ad copy</span>', unsafe_allow_html=True)
        
        # ==================== DOWNLOAD ATTACK PLAN ====================
        st.markdown("---")
        st.markdown("## 📥 Download Attack Materials")
        
        # Attack plan document
        attack_doc = f"""
        {attacker_brand.upper()} ATTACK PLAN vs {target_brand.upper()}
        ==============================================
        
        EXECUTIVE SUMMARY:
        - Target: {target_brand}
        - Weaknesses Found: {negative_count}
        - Exploitation Potential: {weakness_percent:.0f}%
        - Recommended Attack Type: {'Price-based' if opportunities['price_attacks'] else 'Quality-based' if opportunities['product_attacks'] else 'Service-based'}
        
        KEY WEAKNESSES:
        {chr(10).join(['- ' + w for w in opportunities['critical_weaknesses'][:3]])}
        
        MARKETING ATTACKS:
        {chr(10).join(['- ' + a for a in opportunities['marketing_attacks']])}
        
        30-DAY PLAN:
        1. Week 1-2: Awareness campaign
        2. Week 3-4: Conversion campaign
        3. Ongoing: Monitor competitor response
        
        SAMPLE AD COPY:
        "Tired of {target_brand}'s issues? Switch to {attacker_brand} for better!"
        
        TARGET METRICS:
        - Goal: Capture {weakness_percent:.0f}% of dissatisfied {target_brand} customers
        - Timeline: 90 days
        - Budget: Competitive advertising + Special offers
        """
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download Attack Plan",
                data=attack_doc,
                file_name=f"Attack_Plan_{target_brand}.txt",
                mime="text/plain"
            )
        
        with col2:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Weakness Data",
                data=csv,
                file_name=f"{target_brand}_Weaknesses.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome screen
        st.markdown("## 🎯 How to Use This Exploitation Engine")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ⚔️ Business Exploitation Process:
            
            1. **Select Target:** Choose competitor to attack
            2. **Enter Your Brand:** Who is doing the attacking
            3. **Click Attack:** Generate exploitation plan
            4. **Execute:** Use provided strategies
            
            ### 📊 What You Get:
            - Competitor weakness analysis
            - Marketing attack campaigns
            - 30-day implementation plan
            - Ready-to-use ad copy
            - Downloadable attack materials
            """)
        
        with col2:
            st.markdown("""
            ### 🎓 Capstone Focus:
            
            **Topic:** "Exploiting business in review"
            
            **Meaning:** Using competitor reviews to:
            - Find their weaknesses
            - Create attack strategies
            - Steal their customers
            - Grow your market share
            
            **Real Example:**
            - Analyze Samsung reviews
            - Find battery complaints
            - Apple attacks with "Better battery life"
            - Samsung customers switch to Apple
            """)
        
        st.markdown("---")
        st.warning(f"⚠️ **Ready to attack {target_brand}?** Click the ATTACK button above!")

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p><strong>⚔️ CAPSTONE PROJECT: BUSINESS EXPLOITATION ENGINE</strong></p>
    <p><em>"Turning Competitor Reviews into Attack Strategies"</em></p>
    <p>Attack Intelligence | Marketing Warfare | Customer Acquisition</p>
</div>
""", unsafe_allow_html=True)
