import streamlit as st
import pandas as pd
import plotly.express as px
import random
from datetime import datetime

# ==================== GAMIFIED THEME CSS ====================
st.markdown("""
<style>
    /* GAME THEME */
    .game-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        min-height: 100vh;
        padding: 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* GAME HEADER */
    .game-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 15px;
        margin-bottom: 30px;
        border: 3px solid #00d4ff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    /* LEVEL CARDS */
    .level-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        cursor: pointer;
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
    
    /* XP & PROGRESS */
    .xp-badge {
        background: gold;
        color: black;
        padding: 3px 10px;
        border-radius: 10px;
        font-weight: bold;
        display: inline-block;
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
    
    /* BATTLE STATS */
    .stat-card {
        background: rgba(0, 0, 0, 0.3);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
        margin: 10px 0;
    }
    
    /* ACHIEVEMENTS */
    .achievement {
        background: rgba(255, 215, 0, 0.1);
        border: 2px solid gold;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    /* BUTTONS */
    .game-button {
        background: linear-gradient(45deg, #6a11cb, #2575fc);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
        text-align: center;
        display: block;
        width: 100%;
        margin: 10px 0;
    }
    
    .game-button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(106, 17, 203, 0.4);
    }
    
    /* BATTLE ARENA */
    .battle-arena {
        background: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #ff0080;
        margin: 20px 0;
    }
    
    /* ANIMATIONS */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .float {
        animation: float 3s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE GAME STATE ====================
if 'player_xp' not in st.session_state:
    st.session_state.player_xp = 0
if 'player_level' not in st.session_state:
    st.session_state.player_level = 1
if 'completed_missions' not in st.session_state:
    st.session_state.completed_missions = []
if 'current_battle' not in st.session_state:
    st.session_state.current_battle = None

# ==================== GAME HEADER ====================
st.markdown("""
<div class="game-container">
    <div class="game-header">
        <h1 style="font-size: 3rem; margin: 0;">🎮 BUSINESS CONQUEST</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Turn Customer Reviews into Victory Points!</p>
        
        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px;">
            <div>
                <div style="font-size: 0.9rem; opacity: 0.8;">PLAYER LEVEL</div>
                <div style="font-size: 2rem; font-weight: bold;">{}</div>
            </div>
            <div>
                <div style="font-size: 0.9rem; opacity: 0.8;">TOTAL XP</div>
                <div style="font-size: 2rem; font-weight: bold;">{} XP</div>
            </div>
            <div>
                <div style="font-size: 0.9rem; opacity: 0.8;">MISSIONS</div>
                <div style="font-size: 2rem; font-weight: bold;">{}/10</div>
            </div>
        </div>
        
        <div class="progress-bar" style="margin-top: 20px;">
            <div class="progress-fill" style="width: {}%"></div>
        </div>
    </div>
""".format(
    st.session_state.player_level,
    st.session_state.player_xp,
    len(st.session_state.completed_missions),
    (st.session_state.player_xp % 1000) / 10
), unsafe_allow_html=True)

# ==================== MAIN GAME CONTENT ====================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 MISSIONS", "⚔️ BATTLE", "📊 WAR ROOM", "🏆 ACHIEVEMENTS"])

with tab1:
    # MISSION SELECTION
    st.markdown("<h2 style='text-align: center;'>🎯 CHOOSE YOUR BATTLE</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.8;'>Complete missions to earn XP and unlock new strategies</p>", unsafe_allow_html=True)
    
    # Mission Cards
    missions = [
        {
            "id": 1,
            "title": "MOBILE WARS",
            "description": "Conquer the smartphone market",
            "target": "Samsung",
            "enemy": "Apple",
            "xp": 100,
            "difficulty": "Beginner",
            "icon": "📱",
            "color": "#6a11cb"
        },
        {
            "id": 2,
            "title": "AUTO BATTLE",
            "description": "Win the car industry race",
            "target": "Tata Motors",
            "enemy": "Maruti Suzuki",
            "xp": 150,
            "difficulty": "Intermediate",
            "icon": "🚗",
            "color": "#2575fc"
        },
        {
            "id": 3,
            "title": "FOOD FIGHT",
            "description": "Dominate the food delivery space",
            "target": "Dominos",
            "enemy": "Pizza Hut",
            "xp": 200,
            "difficulty": "Advanced",
            "icon": "🍕",
            "color": "#ff0080"
        },
        {
            "id": 4,
            "title": "TECH SHOWDOWN",
            "description": "Battle in the laptop market",
            "target": "Dell",
            "enemy": "HP",
            "xp": 250,
            "difficulty": "Expert",
            "icon": "💻",
            "color": "#00b09b"
        }
    ]
    
    # Display mission cards in 2x2 grid
    cols = st.columns(2)
    for idx, mission in enumerate(missions):
        with cols[idx % 2]:
            completed = mission["id"] in st.session_state.completed_missions
            
            st.markdown("""
            <div class="level-card" style="border-color: {};">
                <div class="level-badge" style="background: {};">{} • {} XP</div>
                <h3 style="margin: 10px 0;">{} {}</h3>
                <p style="opacity: 0.8; font-size: 0.9em;">{}</p>
                <div style="margin: 15px 0;">
                    <div style="font-size: 0.8em; opacity: 0.7;">TARGET: <b>{}</b></div>
                    <div style="font-size: 0.8em; opacity: 0.7;">ENEMY: <b>{}</b></div>
                </div>
                <div style="margin: 15px 0;">
                    <div style="font-size: 0.8em; background: rgba(255,255,255,0.1); padding: 5px; border-radius: 5px; display: inline-block;">
                        Difficulty: {}
                    </div>
                </div>
                {}
            </div>
            """.format(
                mission["color"],
                mission["color"],
                mission["icon"],
                mission["xp"],
                mission["icon"],
                mission["title"],
                mission["description"],
                mission["target"],
                mission["enemy"],
                mission["difficulty"],
                "✅ COMPLETED" if completed else f"""<button class="game-button" onclick="startMission({mission['id']})">LAUNCH MISSION</button>"""
            ), unsafe_allow_html=True)
    
    # Custom Mission
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>🎮 CREATE CUSTOM MISSION</h3>", unsafe_allow_html=True)
    
    custom_col1, custom_col2 = st.columns(2)
    with custom_col1:
        custom_target = st.text_input("Your Brand:", placeholder="Enter your brand name")
    with custom_col2:
        custom_enemy = st.text_input("Enemy Brand:", placeholder="Enter competitor brand")
    
    if st.button("🚀 CREATE CUSTOM BATTLE", type="primary", use_container_width=True):
        if custom_target and custom_enemy:
            st.session_state.current_battle = {
                "target": custom_target,
                "enemy": custom_enemy,
                "is_custom": True
            }
            st.success(f"Battle created: {custom_target} vs {custom_enemy}!")
            st.rerun()

with tab2:
    # BATTLE ARENA
    if st.session_state.current_battle:
        battle = st.session_state.current_battle
        
        st.markdown("""
        <div class="battle-arena">
            <h2 style='text-align: center;'>⚔️ BATTLE IN PROGRESS</h2>
            <div style='text-align: center; font-size: 1.5rem; margin: 20px 0;'>
                <span style='color: #00d4ff;'>{}</span> 
                <span style='margin: 0 20px;'>VS</span>
                <span style='color: #ff0080;'>{}</span>
            </div>
        </div>
        """.format(battle["target"], battle["enemy"]), unsafe_allow_html=True)
        
        # Battle Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='text-align: center;'>🎯 YOUR FORCES</h3>", unsafe_allow_html=True)
            
            # Simulated battle stats
            stats = {
                "Customer Satisfaction": random.randint(70, 90),
                "Price Advantage": random.randint(60, 85),
                "Feature Strength": random.randint(65, 95),
                "Service Quality": random.randint(50, 80)
            }
            
            for stat, value in stats.items():
                st.markdown(f"""
                <div class="stat-card">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{stat}</span>
                        <span style="color: #00d4ff; font-weight: bold;">{value}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {value}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h3 style='text-align: center;'>⚠️ ENEMY WEAKNESSES</h3>", unsafe_allow_html=True)
            
            weaknesses = [
                f"{battle['enemy']} has 35% battery complaints",
                f"Price too high - 42% negative mentions",
                f"Slow customer service response",
                f"Limited features compared to competitors"
            ]
            
            for weakness in weaknesses:
                st.markdown(f"""
                <div class="stat-card" style="border-color: #ff0080;">
                    <div style="color: #ff0080;">🔍 {weakness}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Battle Controls
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 LAUNCH ATTACK", type="primary", use_container_width=True):
                # Simulate battle outcome
                win_chance = random.randint(60, 90)
                if win_chance > 70:
                    st.session_state.player_xp += 100
                    if not battle.get("is_custom", False):
                        st.session_state.completed_missions.append(battle.get("mission_id", 0))
                    st.success(f"✅ VICTORY! +100 XP (Win chance: {win_chance}%)")
                else:
                    st.session_state.player_xp += 30
                    st.warning(f"⚠️ PARTIAL VICTORY! +30 XP (Win chance: {win_chance}%)")
                
                if st.session_state.player_xp >= st.session_state.player_level * 1000:
                    st.session_state.player_level += 1
                    st.balloons()
                
                st.session_state.current_battle = None
                st.rerun()
        
        with col2:
            if st.button("🔄 RETREAT & ANALYZE", use_container_width=True):
                st.info("Retreating to gather more intelligence...")
                st.session_state.player_xp += 20
                st.session_state.current_battle = None
                st.rerun()
        
        with col3:
            if st.button("🏁 END BATTLE", use_container_width=True):
                st.session_state.current_battle = None
                st.rerun()
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <div style="font-size: 4rem;" class="float">⚔️</div>
            <h2>NO ACTIVE BATTLE</h2>
            <p style="opacity: 0.7;">Select a mission from the Missions tab to begin battle!</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    # WAR ROOM - Analytics Dashboard
    st.markdown("<h2 style='text-align: center;'>📊 WAR ROOM - INTELLIGENCE DASHBOARD</h2>", unsafe_allow_html=True)
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 0.9em; opacity: 0.7;">BATTLES WON</div>
            <div style="font-size: 2em; font-weight: bold; color: #00d4ff;">{}</div>
        </div>
        """.format(len(st.session_state.completed_missions)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 0.9em; opacity: 0.7;">SUCCESS RATE</div>
            <div style="font-size: 2em; font-weight: bold; color: #00b09b;">85%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 0.9em; opacity: 0.7;">ENEMIES DEFEATED</div>
            <div style="font-size: 2em; font-weight: bold; color: #ff0080;">{}</div>
        </div>
        """.format(len(st.session_state.completed_missions)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div style="font-size: 0.9em; opacity: 0.7;">NEXT RANK</div>
            <div style="font-size: 2em; font-weight: bold; color: gold;">{} XP</div>
        </div>
        """.format((st.session_state.player_level * 1000) - st.session_state.player_xp), unsafe_allow_html=True)
    
    # XP Progress Chart
    st.markdown("<h3>📈 CONQUEST PROGRESS</h3>", unsafe_allow_html=True)
    
    # Simulated progress data
    progress_data = {
        "Week": ["Week 1", "Week 2", "Week 3", "Week 4"],
        "XP Earned": [150, 300, 220, 400],
        "Battles": [2, 3, 2, 4]
    }
    
    df = pd.DataFrame(progress_data)
    fig = px.line(df, x="Week", y="XP Earned", title="Weekly XP Progress",
                  markers=True, line_shape="spline")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font_color="white", title_font_color="white")
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent Battles
    st.markdown("<h3>🔄 RECENT BATTLES</h3>", unsafe_allow_html=True)
    
    recent_battles = [
        {"mission": "Mobile Wars", "result": "Victory", "xp": 100, "date": "2024-01-20"},
        {"mission": "Auto Battle", "result": "Victory", "xp": 150, "date": "2024-01-18"},
        {"mission": "Food Fight", "result": "Partial", "xp": 50, "date": "2024-01-15"}
    ]
    
    for battle in recent_battles:
        result_color = "#00d4ff" if battle["result"] == "Victory" else "#ff8c00"
        st.markdown(f"""
        <div class="stat-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{battle['mission']}</strong>
                    <div style="font-size: 0.8em; opacity: 0.7;">{battle['date']}</div>
                </div>
                <div style="text-align: right;">
                    <div style="color: {result_color}; font-weight: bold;">{battle['result']}</div>
                    <div class="xp-badge">+{battle['xp']} XP</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    # ACHIEVEMENTS
    st.markdown("<h2 style='text-align: center;'>🏆 ACHIEVEMENTS & REWARDS</h2>", unsafe_allow_html=True)
    
    achievements = [
        {"name": "First Blood", "desc": "Complete your first battle", "earned": len(st.session_state.completed_missions) > 0, "reward": "🎖️"},
        {"name": "Conqueror", "desc": "Win 5 battles", "earned": len(st.session_state.completed_missions) >= 5, "reward": "🏆"},
        {"name": "XP Master", "desc": "Earn 1000 XP", "earned": st.session_state.player_xp >= 1000, "reward": "⭐"},
        {"name": "Strategic Genius", "desc": "Reach Level 5", "earned": st.session_state.player_level >= 5, "reward": "👑"},
        {"name": "Custom Commander", "desc": "Create 3 custom battles", "earned": False, "reward": "⚔️"},
    ]
    
    cols = st.columns(3)
    for idx, achievement in enumerate(achievements):
        with cols[idx % 3]:
            earned_style = "border-color: gold; background: rgba(255,215,0,0.1);" if achievement["earned"] else "opacity: 0.5;"
            
            st.markdown(f"""
            <div class="level-card" style="{earned_style}">
                <div style="font-size: 2.5rem; text-align: center;">{achievement['reward']}</div>
                <h4 style="text-align: center; margin: 10px 0;">{achievement['name']}</h4>
                <p style="text-align: center; font-size: 0.9em; opacity: 0.8;">{achievement['desc']}</p>
                <div style="text-align: center; margin-top: 10px;">
                    {"✅ EARNED" if achievement['earned'] else "🔒 LOCKED"}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Next Rewards
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>🎁 NEXT REWARDS</h3>", unsafe_allow_html=True)
    
    next_rewards = [
        {"level": 2, "reward": "Unlock Advanced Analytics", "progress": min(100, (st.session_state.player_xp / 2000) * 100)},
        {"level": 3, "reward": "Get Strategy Templates", "progress": min(100, (st.session_state.player_xp / 3000) * 100)},
        {"level": 5, "reward": "Executive Dashboard Access", "progress": min(100, (st.session_state.player_xp / 5000) * 100)},
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
<div style="text-align: center; opacity: 0.7; font-size: 0.9em;">
    <p>🎮 BUSINESS CONQUEST v1.0 • Capstone Project: Exploiting Business in Review</p>
    <p>Earn XP by analyzing customer reviews and defeating competitors!</p>
</div>
</div>
""", unsafe_allow_html=True)

# ==================== JAVASCRIPT FOR BUTTONS ====================
st.markdown("""
<script>
function startMission(missionId) {
    // This would trigger a Streamlit rerun with the mission ID
    // For now, we'll show an alert
    alert("Starting Mission " + missionId + "! This would launch the battle in a real implementation.");
    
    // In real implementation, we would set Streamlit session state
    // and trigger a rerun
}
</script>
""", unsafe_allow_html=True)
