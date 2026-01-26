import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from datetime import datetime
import numpy as np

# ==================== NLP LIBRARIES ====================
try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    st.warning("⚠️ Install: pip install textblob vaderSentiment")

# ==================== CSS STYLING ====================
st.markdown("""
<style>
    .game-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
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
        font-size: 0.9em;
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
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv('amazon_reviews.csv')
        df = df.dropna(subset=['Reviews'])
        df['Reviews'] = df['Reviews'].astype(str)
        
        if 'Brand Name' in df.columns:
            df['Brand Name'] = df['Brand Name'].str.strip().str.title()
        if 'Rating' in df.columns:
            df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        
        return df
    except FileNotFoundError:
        st.error("❌ 'amazon_reviews.csv' not found! Download from Kaggle.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# ==================== SENTIMENT ANALYSIS ====================
def analyze_sentiment(text):
    if not NLP_AVAILABLE:
        return 0.0
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores['compound']

def extract_keywords(reviews_list, top_n=10):
    all_text = ' '.join(reviews_list).lower()
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has',
                  'this', 'that', 'it', 'phone', 'product', 'very', 'good', 'bad', 'not',
                  'just', 'really', 'would', 'one', 'get', 'my', 'me', 'i'}
    
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    words = [w for w in words if w not in stop_words]
    word_freq = Counter(words)
    return word_freq.most_common(top_n)

def analyze_brand_performance(df, brand_name):
    brand_df = df[df['Brand Name'].str.contains(brand_name, case=False, na=False)]
    
    if len(brand_df) == 0:
        return None
    
    total_reviews = len(brand_df)
    avg_rating = brand_df['Rating'].mean() if 'Rating' in brand_df.columns else 0
    
    if NLP_AVAILABLE:
        brand_df['sentiment'] = brand_df['Reviews'].apply(analyze_sentiment)
        avg_sentiment = brand_df['sentiment'].mean()
        positive_pct = (brand_df['sentiment'] > 0.1).sum() / total_reviews * 100
        negative_pct = (brand_df['sentiment'] < -0.1).sum() / total_reviews * 100
    else:
        avg_sentiment = (avg_rating - 3) / 2
        positive_pct = (brand_df['Rating'] >= 4).sum() / total_reviews * 100
        negative_pct = (brand_df['Rating'] <= 2).sum() / total_reviews * 100
    
    negative_reviews = brand_df[brand_df['Rating'] <= 2]['Reviews'].tolist() if 'Rating' in brand_df.columns else []
    complaints = extract_keywords(negative_reviews, top_n=5) if negative_reviews else []
    
    positive_reviews = brand_df[brand_df['Rating'] >= 4]['Reviews'].tolist() if 'Rating' in brand_df.columns else []
    strengths = extract_keywords(positive_reviews, top_n=5) if positive_reviews else []
    
    return {
        'total_reviews': total_reviews,
        'avg_rating': avg_rating,
        'avg_sentiment': avg_sentiment,
        'positive_pct': positive_pct,
        'negative_pct': negative_pct,
        'neutral_pct': 100 - positive_pct - negative_pct,
        'complaints': complaints,
        'strengths': strengths,
        'customer_satisfaction': (avg_sentiment + 1) * 50,
        'quality_score': avg_rating * 20 if avg_rating else 0,
    }

# ==================== STRATEGIC RECOMMENDATIONS ====================
def generate_strategic_recommendations(your_analysis, competitor_analysis, your_brand, competitor_brand):
    recommendations = {
        'swot': {'strengths': [], 'weaknesses': [], 'opportunities': [], 'threats': []},
        'action_items': [],
        'competitive_advantages': [],
        'risk_warnings': []
    }
    
    # STRENGTHS
    if your_analysis['customer_satisfaction'] > competitor_analysis['customer_satisfaction']:
        diff = your_analysis['customer_satisfaction'] - competitor_analysis['customer_satisfaction']
        recommendations['swot']['strengths'].append(
            f"Customer satisfaction {diff:.1f}% higher than {competitor_brand}"
        )
        recommendations['competitive_advantages'].append(
            f"Leverage superior satisfaction in marketing campaigns"
        )
    
    if your_analysis['avg_rating'] > competitor_analysis['avg_rating']:
        recommendations['swot']['strengths'].append(
            f"Rating ({your_analysis['avg_rating']:.1f}★) exceeds {competitor_brand} ({competitor_analysis['avg_rating']:.1f}★)"
        )
    
    for keyword, count in your_analysis['strengths'][:2]:
        recommendations['swot']['strengths'].append(
            f"'{keyword}' highly praised ({count} mentions)"
        )
    
    # WEAKNESSES
    if your_analysis['negative_pct'] > competitor_analysis['negative_pct']:
        diff = your_analysis['negative_pct'] - competitor_analysis['negative_pct']
        recommendations['swot']['weaknesses'].append(
            f"Negative reviews {diff:.1f}% higher than {competitor_brand}"
        )
        recommendations['action_items'].append(
            f"URGENT: Address top complaints immediately"
        )
    
    for keyword, count in your_analysis['complaints'][:3]:
        recommendations['swot']['weaknesses'].append(
            f"'{keyword}' recurring complaint ({count} mentions)"
        )
        recommendations['action_items'].append(
            f"Improve '{keyword}' to reduce dissatisfaction"
        )
    
    # OPPORTUNITIES
    for keyword, count in competitor_analysis['complaints'][:3]:
        recommendations['swot']['opportunities'].append(
            f"{competitor_brand} weakness: '{keyword}' ({count} complaints)"
        )
        recommendations['action_items'].append(
            f"Highlight YOUR superior '{keyword}' in advertising"
        )
    
    if competitor_analysis['negative_pct'] > 30:
        recommendations['swot']['opportunities'].append(
            f"{competitor_brand} has {competitor_analysis['negative_pct']:.1f}% negative reviews"
        )
    
    # THREATS
    if competitor_analysis['customer_satisfaction'] > your_analysis['customer_satisfaction']:
        diff = competitor_analysis['customer_satisfaction'] - your_analysis['customer_satisfaction']
        recommendations['swot']['threats'].append(
            f"{competitor_brand} satisfaction {diff:.1f}% higher"
        )
        recommendations['risk_warnings'].append(
            f"Risk of customer loss to {competitor_brand}"
        )
    
    for keyword, count in competitor_analysis['strengths'][:2]:
        recommendations['swot']['threats'].append(
            f"{competitor_brand} excels at '{keyword}'"
        )
        recommendations['action_items'].append(
            f"Match {competitor_brand}'s '{keyword}' performance"
        )
    
    # Strategic positioning
    your_score = your_analysis['customer_satisfaction'] + your_analysis['quality_score']
    comp_score = competitor_analysis['customer_satisfaction'] + competitor_analysis['quality_score']
    
    if your_score > comp_score:
        recommendations['action_items'].insert(0, 
            f"✅ WINNING: Maintain quality, expand aggressively"
        )
    else:
        gap = comp_score - your_score
        recommendations['action_items'].insert(0, 
            f"⚠️ GAP: {gap:.1f} points behind. Focus on quality NOW"
        )
    
    return recommendations

# ==================== SESSION STATE ====================
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

# Load data
if not st.session_state.data_loaded:
    with st.spinner('🔄 Loading dataset...'):
        df = load_and_process_data()
        if df is not None:
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.available_brands = sorted(df['Brand Name'].unique().tolist()) if 'Brand Name' in df.columns else []

# ==================== HEADER ====================
st.markdown(f"""
<div class="game-container">
    <div class="game-header">
        <h1 style="font-size: 3rem; margin: 0;">🎮 BUSINESS CONQUEST</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Exploit Customer Reviews for Strategic Advantage</p>
        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px;">
            <div>
                <div style="font-size: 0.9rem; opacity: 0.8;">LEVEL</div>
                <div style="font-size: 2rem; font-weight: bold;">{st.session_state.player_level}</div>
            </div>
            <div>
                <div style="font-size: 0.9rem; opacity: 0.8;">XP</div>
                <div style="font-size: 2rem; font-weight: bold;">{st.session_state.player_xp}</div>
            </div>
            <div>
                <div style="font-size: 0.9rem; opacity: 0.8;">MISSIONS</div>
                <div style="font-size: 2rem; font-weight: bold;">{len(st.session_state.completed_missions)}/10</div>
            </div>
        </div>
        <div class="progress-bar" style="margin-top: 20px;">
            <div class="progress-fill" style="width: {(st.session_state.player_xp % 1000) / 10}%"></div>
        </div>
    </div>
""", unsafe_allow_html=True)

if not st.session_state.data_loaded:
    st.error("⚠️ Cannot proceed without data.")
    st.stop()

# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 MISSIONS", "⚔️ BATTLE", "📊 INSIGHTS", "🏆 ACHIEVEMENTS"])

with tab1:
    st.markdown("<h2 style='text-align: center;'>🎯 SELECT YOUR BATTLE</h2>", unsafe_allow_html=True)
    
    common_brands = ['Samsung', 'Apple', 'OnePlus', 'Xiaomi', 'Motorola', 'Nokia']
    available = [b for b in common_brands if b in st.session_state.available_brands]
    
    if len(available) >= 2:
        missions = [
            {"id": 1, "title": "FLAGSHIP WAR", "target": available[0], "enemy": available[1], 
             "xp": 100, "icon": "📱", "color": "#6a11cb"},
            {"id": 2, "title": "BUDGET BATTLE", "target": available[2] if len(available) > 2 else available[0], 
             "enemy": available[3] if len(available) > 3 else available[1], "xp": 150, "icon": "💰", "color": "#2575fc"}
        ]
        
        cols = st.columns(2)
        for idx, m in enumerate(missions):
            with cols[idx]:
                completed = m["id"] in st.session_state.completed_missions
                st.markdown(f"""
                <div class="level-card" style="border-color: {m['color']};">
                    <div class="level-badge">{m['icon']} • {m['xp']} XP</div>
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
    if st.session_state.available_brands:
        c1, c2 = st.columns(2)
        with c1:
            target = st.selectbox("Your Brand:", st.session_state.available_brands)
        with c2:
            enemy = st.selectbox("Competitor:", [b for b in st.session_state.available_brands if b != target])
        
        if st.button("🚀 CREATE BATTLE", type="primary"):
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
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner('🔍 Analyzing reviews...'):
            your_data = analyze_brand_performance(st.session_state.df, battle['target'])
            comp_data = analyze_brand_performance(st.session_state.df, battle['enemy'])
        
        if your_data and comp_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"<h3 style='text-align: center;'>🎯 {battle['target']}</h3>", unsafe_allow_html=True)
                
                for name, val in [("Satisfaction", int(your_data['customer_satisfaction'])),
                                 ("Quality", int(your_data['quality_score'])),
                                 ("Positive %", int(your_data['positive_pct']))]:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div style="display: flex; justify-content: space-between;">
                            <span>{name}</span><span style="color: #00d4ff;">{val}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {val}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<h4>💪 Strengths:</h4>", unsafe_allow_html=True)
                for kw, cnt in your_data['strengths'][:3]:
                    st.markdown(f"<div class='stat-card'>✅ {kw} ({cnt})</div>", unsafe_allow_html=True)
                
                st.markdown("<h4>⚠️ Complaints:</h4>", unsafe_allow_html=True)
                for kw, cnt in your_data['complaints'][:3]:
                    st.markdown(f"<div class='stat-card'>❌ {kw} ({cnt})</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<h3 style='text-align: center;'>🎯 {battle['enemy']}</h3>", unsafe_allow_html=True)
                
                for name, val in [("Satisfaction", int(comp_data['customer_satisfaction'])),
                                 ("Quality", int(comp_data['quality_score'])),
                                 ("Negative %", int(comp_data['negative_pct']))]:
                    st.markdown(f"""
                    <div class="stat-card" style="border-color: #ff0080;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>{name}</span><span style="color: #ff0080;">{val}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {val}%; background: #ff0080;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<h4>💪 Their Strengths:</h4>", unsafe_allow_html=True)
                for kw, cnt in comp_data['strengths'][:3]:
                    st.markdown(f"<div class='stat-card' style='border-color: #ff0080;'>✅ {kw} ({cnt})</div>", unsafe_allow_html=True)
                
                st.markdown("<h4>🔍 Their Weaknesses:</h4>", unsafe_allow_html=True)
                for kw, cnt in comp_data['complaints'][:3]:
                    st.markdown(f"<div class='stat-card'>❌ {kw} ({cnt})</div>", unsafe_allow_html=True)
            
            # Win probability
            your_score = your_data['customer_satisfaction'] + your_data['quality_score']
            comp_score = comp_data['customer_satisfaction'] + comp_data['quality_score']
            win_pct = int((your_score / (your_score + comp_score)) * 100) if (your_score + comp_score) > 0 else 50
            
            st.markdown(f"""
            <div class="stat-card" style="border-color: gold; text-align: center;">
                <h3>📊 MARKET POSITION</h3>
                <div style="font-size: 1.5rem; color: gold;">
                    Competitive Advantage: {win_pct}%
                </div>
                <p style="font-size: 0.9rem; opacity: 0.8;">
                    {your_data['total_reviews']} reviews analyzed for {battle['target']}<br>
                    {comp_data['total_reviews']} reviews for {battle['enemy']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # RECOMMENDATIONS
            if st.session_state.show_recommendations:
                recs = st.session_state.recommendations
                
                st.markdown("<br><h2 style='text-align: center;'>💡 STRATEGIC RECOMMENDATIONS</h2>", unsafe_allow_html=True)
                
                # SWOT
                st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
                st.markdown("<h3>📊 SWOT ANALYSIS</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h4 style='color: #00d4ff;'>💪 STRENGTHS</h4>", unsafe_allow_html=True)
                    for s in recs['swot']['strengths']:
                        st.markdown(f"<div class='action-item'>✅ {s}</div>", unsafe_allow_html=True)
                    
                    st.markdown("<h4 style='color: #ff8c00;'>⚠️ WEAKNESSES</h4>", unsafe_allow_html=True)
                    for w in recs['swot']['weaknesses']:
                        st.markdown(f"<div class='action-item' style='border-color: #ff8c00;'>❌ {w}</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<h4 style='color: #00b09b;'>🎯 OPPORTUNITIES</h4>", unsafe_allow_html=True)
                    for o in recs['swot']['opportunities']:
                        st.markdown(f"<div class='action-item' style='border-color: #00b09b;'>💡 {o}</div>", unsafe_allow_html=True)
                    
                    st.markdown("<h4 style='color: #ff0080;'>🚨 THREATS</h4>", unsafe_allow_html=True)
                    for t in recs['swot']['threats']:
                        st.markdown(f"<div class='action-item' style='border-color: #ff0080;'>⚠️ {t}</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # ACTION ITEMS
                st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
                st.markdown("<h3>🎯 IMMEDIATE ACTION ITEMS</h3>", unsafe_allow_html=True)
                for idx, action in enumerate(recs['action_items'][:5], 1):
                    st.markdown(f"<div class='action-item'><b>{idx}.</b> {action}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # COMPETITIVE ADVANTAGES
                if recs['competitive_advantages']:
                    st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
                    st.markdown("<h3>⚡ COMPETITIVE ADVANTAGES</h3>", unsafe_allow_html=True)
                    for adv in recs['competitive_advantages']:
                        st.markdown(f"<div class='action-item' style='border-color: gold;'>🏆 {adv}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Complete mission button
                if st.button("✅ COMPLETE MISSION & EARN XP", type="primary", use_container_width=True):
                    xp = battle['xp_reward']
                    st.session_state.player_xp += xp
                    
                    if not battle.get("is_custom") and battle.get('mission_id') not in st.session_state.completed_missions:
                        st.session_state.completed_missions.append(battle['mission_id'])
                    
                    if st.session_state.player_xp >= st.session_state.player_level * 1000:
                        st.session_state.player_level += 1
                        st.balloons()
                    
                    st.success(f"🎉 Mission Complete! +{xp} XP")
                    st.session_state.current_battle = None
                    st.session_state.show_recommendations = False
                    st.rerun()
            
            else:
                # Generate recommendations button
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💡 GENERATE STRATEGY", type="primary", use_container_width=True):
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
            st.error("❌ Insufficient data for analysis")
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <div style="font-size: 4rem;">⚔️</div>
            <h2>NO ACTIVE BATTLE</h2>
            <p style="opacity: 0.7;">Select a mission from the Missions tab!</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 style='text-align: center;'>📊 BUSINESS INTELLIGENCE</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    total_reviews = len(st.session_state.df)
    total_brands = len(st.session_state.available_brands)
    
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
            <div style="font-size: 0.9em; opacity: 0.7;">MISSIONS DONE</div>
            <div style="font-size: 2em; color: #ff0080;">{len(st.session_state.completed_missions)}</div>                                                                                                                         
</div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 0.9em; opacity: 0.7;">SUCCESS RATE</div>
            <div style="font-size: 2em; color: gold;">
                {min(100, len(st.session_state.completed_missions) * 20)}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Brand Performance Chart
    st.markdown("<br><h3>📊 Top Brands Comparison</h3>", unsafe_allow_html=True)
    
    top_brands = st.session_state.df['Brand Name'].value_counts().head(5)
    brand_perf = []
    
    for brand in top_brands.index:
        analysis = analyze_brand_performance(st.session_state.df, brand)
        if analysis:
            brand_perf.append({
                'Brand': brand,
                'Avg Rating': analysis['avg_rating'],
                'Reviews': analysis['total_reviews'],
                'Satisfaction': analysis['customer_satisfaction']
            })
    
    if brand_perf:
        perf_df = pd.DataFrame(brand_perf)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=perf_df['Brand'],
            y=perf_df['Avg Rating'],
            name='Average Rating',
            marker_color='#00d4ff',
            text=perf_df['Avg Rating'].round(2),
            textposition='auto'
        ))
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            title_font_color="white",
            xaxis_title="Brand",
            yaxis_title="Average Rating (out of 5)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment Distribution
        st.markdown("<br><h3>😊 Sentiment Distribution</h3>", unsafe_allow_html=True)
        
        sentiment_data = []
        for brand in top_brands.index[:3]:
            analysis = analyze_brand_performance(st.session_state.df, brand)
            if analysis:
                sentiment_data.append({
                    'Brand': brand,
                    'Positive': analysis['positive_pct'],
                    'Neutral': analysis['neutral_pct'],
                    'Negative': analysis['negative_pct']
                })
        
        if sentiment_data:
            sent_df = pd.DataFrame(sentiment_data)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name='Positive', x=sent_df['Brand'], y=sent_df['Positive'], marker_color='#00d4ff'))
            fig2.add_trace(go.Bar(name='Neutral', x=sent_df['Brand'], y=sent_df['Neutral'], marker_color='#ff8c00'))
            fig2.add_trace(go.Bar(name='Negative', x=sent_df['Brand'], y=sent_df['Negative'], marker_color='#ff0080'))
            
            fig2.update_layout(
                barmode='stack',
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                title_font_color="white",
                xaxis_title="Brand",
                yaxis_title="Percentage (%)",
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.markdown("<h2 style='text-align: center;'>🏆 ACHIEVEMENTS</h2>", unsafe_allow_html=True)
    
    achievements = [
        {
            "name": "Data Explorer",
            "desc": "Successfully loaded dataset",
            "earned": st.session_state.data_loaded,
            "reward": "🎖️"
        },
        {
            "name": "First Victory",
            "desc": "Complete your first analysis",
            "earned": len(st.session_state.completed_missions) > 0,
            "reward": "🏆"
        },
        {
            "name": "Strategic Thinker",
            "desc": "Complete 3 missions",
            "earned": len(st.session_state.completed_missions) >= 3,
            "reward": "⭐"
        },
        {
            "name": "XP Hunter",
            "desc": "Earn 500 XP",
            "earned": st.session_state.player_xp >= 500,
            "reward": "💰"
        },
        {
            "name": "Master Analyst",
            "desc": "Reach Level 3",
            "earned": st.session_state.player_level >= 3,
            "reward": "👑"
        },
        {
            "name": "Competitive Genius",
            "desc": "Complete 5 missions",
            "earned": len(st.session_state.completed_missions) >= 5,
            "reward": "⚔️"
        }
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
                <div style="margin-top: 10px; font-weight: bold;">
                    {"✅ EARNED" if achievement['earned'] else "🔒 LOCKED"}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Progress to next rewards
    st.markdown("<br><h3 style='text-align: center;'>🎁 NEXT REWARDS</h3>", unsafe_allow_html=True)
    
    next_rewards = [
        {"level": 2, "reward": "Unlock Advanced Analytics", 
         "progress": min(100, (st.session_state.player_xp / 2000) * 100)},
        {"level": 3, "reward": "Export Report Feature", 
         "progress": min(100, (st.session_state.player_xp / 3000) * 100)},
        {"level": 5, "reward": "AI-Powered Insights", 
         "progress": min(100, (st.session_state.player_xp / 5000) * 100)}
    ]
    
    for reward in next_rewards:
        st.markdown(f"""
        <div class="stat-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span><b>Level {reward['level']}:</b> {reward['reward']}</span>
                <span>{reward['progress']:.0f}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {reward['progress']}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<br><br>
<div style="text-align: center; opacity: 0.7; font-size: 0.9em; padding-bottom: 30px;">
    <p>🎮 BUSINESS CONQUEST v2.0</p>
    <p>Capstone Project: Exploiting Business Insights from Customer Reviews</p>
    <p>Powered by VADER Sentiment Analysis & NLP</p>
</div>
</div>
""", unsafe_allow_html=True)
