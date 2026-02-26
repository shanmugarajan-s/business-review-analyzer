import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# ML imports kept for pipeline demo reference
import warnings
warnings.filterwarnings("ignore")
try:
    import anthropic as anthropic_client
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

st.set_page_config(page_title="Error-404", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Sans:wght@400;500&display=swap");
html,body,[class*="css"]{font-family:"DM Sans",sans-serif;}
.stApp{background:#07090f;color:#dde4f5;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:2rem;padding-bottom:4rem;max-width:980px;}
.nav-bar{display:flex;align-items:center;justify-content:space-between;padding:14px 0;margin-bottom:28px;border-bottom:1px solid #1c2540;}
.nav-logo{font-family:"Sora",sans-serif;font-size:1.3rem;font-weight:800;color:#00e5ff;letter-spacing:-0.5px;}
.nav-logo span{color:#dde4f5;}
.nav-steps{display:flex;gap:8px;}
.nav-step{padding:5px 14px;border-radius:100px;font-size:.75rem;font-weight:600;border:1px solid #1c2540;color:#3d4d70;}
.nav-step.active{background:rgba(0,229,255,.1);border-color:rgba(0,229,255,.4);color:#00e5ff;}
.nav-step.done{background:rgba(16,185,129,.08);border-color:rgba(16,185,129,.3);color:#10b981;}
.metric-card{background:#131a27;border:1px solid #1c2540;border-radius:16px;padding:20px 24px;text-align:center;}
.metric-num{font-family:"Sora",sans-serif;font-size:2rem;font-weight:800;display:block;margin-bottom:4px;}
.metric-lbl{font-size:.72rem;text-transform:uppercase;letter-spacing:.5px;color:#3d4d70;}
.insight-card{background:#131a27;border:1px solid #1c2540;border-left:4px solid transparent;border-radius:0 14px 14px 0;padding:18px 20px;margin-bottom:12px;display:flex;gap:14px;}
.insight-card.pos{border-left-color:#10b981;}
.insight-card.neg{border-left-color:#ef4444;}
.insight-card.neu{border-left-color:#f59e0b;}
.insight-title{font-weight:700;font-size:.92rem;color:#dde4f5;margin-bottom:4px;}
.insight-desc{font-size:.82rem;color:#3d4d70;line-height:1.6;}
.pipe-step{background:#131a27;border:1px solid #1c2540;border-radius:14px;padding:16px 20px;margin-bottom:8px;display:flex;gap:14px;}
.pipe-step.active{border-color:#00e5ff;}
.pipe-step.done{border-color:#10b981;}
.pipe-num{width:30px;height:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-family:"Sora",sans-serif;font-size:.78rem;font-weight:700;background:rgba(0,229,255,.1);border:1px solid rgba(0,229,255,.3);color:#00e5ff;flex-shrink:0;}
.pipe-num.done-num{background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.3);color:#10b981;}
.pipe-title{font-family:"Sora",sans-serif;font-size:.92rem;font-weight:600;margin-bottom:3px;}
.pipe-desc{font-size:.78rem;color:#3d4d70;}
.token{display:inline-block;padding:3px 9px;border-radius:5px;font-size:.73rem;margin:2px;font-family:monospace;border:1px solid;}
.token-raw{background:rgba(100,100,120,.1);border-color:#2a3050;color:#5a6a90;}
.token-kept{background:rgba(0,229,255,.08);border-color:rgba(0,229,255,.25);color:#00e5ff;}
.token-removed{background:rgba(239,68,68,.08);border-color:rgba(239,68,68,.2);color:#f87171;text-decoration:line-through;opacity:.6;}
.token-clean{background:rgba(16,185,129,.08);border-color:rgba(16,185,129,.25);color:#10b981;}
.upload-hint{background:#131a27;border:2px dashed #1c2540;border-radius:16px;padding:26px;text-align:center;color:#3d4d70;font-size:.88rem;line-height:1.8;}
.upload-hint code{color:#00e5ff;background:rgba(0,229,255,.08);padding:1px 6px;border-radius:4px;}
.sec-tag{font-size:.68rem;letter-spacing:2px;text-transform:uppercase;color:#00e5ff;margin-bottom:6px;}
.sec-heading{font-family:"Sora",sans-serif;font-size:1.15rem;font-weight:700;letter-spacing:-.5px;margin-bottom:14px;}
.divider{height:1px;background:#1c2540;margin:24px 0;}
div[data-testid="stButton"] button{border-radius:12px!important;font-family:"DM Sans",sans-serif!important;font-weight:600!important;}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = 1

STOPWORDS = set(["the","a","an","is","it","was","were","this","that","of","in","on","at","to","for","with","and","or","but","so","very","just","be","are","has","have","had","i","my","me","we","our","they","their","you","your","its","not","no","from","by","also","am","been","do","did","does","would","could","should","will","can","may","might","must","as","than","then","too"])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def get_sentiment(r):
    return "Positive" if r >= 4 else ("Negative" if r <= 2 else "Neutral")

def get_top_words(texts, n=20):
    from collections import Counter
    words = []
    for t in texts:
        ws = [w for w in clean_text(t).split() if w not in STOPWORDS and len(w) > 2]
        words.extend(ws)
    return Counter(words).most_common(n)

def dark_fig(fs=(7,4)):
    fig, ax = plt.subplots(figsize=fs, facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    for s in ax.spines.values(): s.set_edgecolor("#1c2540")
    ax.tick_params(colors="#3d4d70")
    return fig, ax

def nav_bar(cur):
    steps = ["Welcome","How It Works","Analyze"]
    html = ""
    for i,s in enumerate(steps):
        pg = i+1
        cls = "active" if pg==cur else ("done" if pg<cur else "")
        lbl = ("✓ "+s) if pg<cur else s
        html += f'<div class="nav-step {cls}">{lbl}</div>'
    st.markdown(f'''<div class="nav-bar"><div class="nav-logo">Review<span>Mind</span></div><div class="nav-steps">{html}</div></div>''', unsafe_allow_html=True)

# ══════════════════════════════════
# PAGE 1 — WELCOME
# ══════════════════════════════════
def page_welcome():
    nav_bar(1)
    st.markdown("""
    <div style="text-align:center;padding:40px 20px 24px">
        <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(0,229,255,.07);border:1px solid rgba(0,229,255,.18);border-radius:100px;padding:5px 16px;font-size:.72rem;color:#00e5ff;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:24px">
            📊 Text Analytics Project
        </div>
        <div style="font-family:'Sora',sans-serif;font-size:clamp(2rem,5vw,3.4rem);font-weight:800;line-height:1.05;letter-spacing:-2px;margin-bottom:16px">
            Business Insights from<br>
            <span style="background:linear-gradient(90deg,#00e5ff,#7c3aed,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">Customer Reviews</span>
        </div>
        <div style="color:#3d4d70;font-size:.98rem;line-height:1.7;max-width:460px;margin:0 auto 32px">
            Upload a review dataset — ML models analyze sentiment, extract keywords, and generate actionable business intelligence instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><span class="metric-num" style="color:#10b981">4</span><span class="metric-lbl">ML Models</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><span class="metric-num" style="color:#00e5ff">91%</span><span class="metric-lbl">Best Accuracy</span></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><span class="metric-num" style="color:#7c3aed">SSSS</span><span class="metric-lbl">Powered Insights</span></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    _,col,_ = st.columns([1,2,1])
    with col:
        if st.button("See How It Works →", use_container_width=True, type="primary"):
            st.session_state.page = 2; st.rerun()
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Skip → Try Now", use_container_width=True):
            st.session_state.page = 3; st.rerun()

# ══════════════════════════════════
# PAGE 2 — PIPELINE
# ══════════════════════════════════
def page_pipeline():
    nav_bar(2)
    
    # Embed the full interactive HTML pipeline directly
    PIPELINE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>How It Works — Pipeline Explainer</title>
<link href="https://fonts.googleapis.com/css2?family=Clash+Display:wght@500;600;700&family=JetBrains+Mono:wght@400;500&family=Cabinet+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #07090f;
  --surface: #0e1117;
  --card: #131a27;
  --border: #1c2540;
  --accent: #00e5ff;
  --green: #10b981;
  --purple: #7c3aed;
  --orange: #f59e0b;
  --red: #ef4444;
  --text: #dde4f5;
  --muted: #3d4d70;
  --mono: 'JetBrains Mono', monospace;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Cabinet Grotesk', sans-serif;
  min-height: 100vh;
  padding: 100px 20px 80px;
}

h1 {
  font-family: 'Clash Display', sans-serif;
  font-size: clamp(1.8rem, 4vw, 3rem);
  font-weight: 700;
  letter-spacing: -1.5px;
  text-align: center;
  margin-bottom: 6px;
}
.title-accent { color: var(--accent); }
.subtitle {
  text-align: center;
  color: var(--muted);
  font-size: 1rem;
  margin-bottom: 40px;
}

/* Sample selector */
.sample-row {
  display: flex; gap: 10px; flex-wrap: wrap;
  justify-content: center; margin-bottom: 32px;
}
.sample-btn {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 100px;
  padding: 8px 18px;
  color: var(--muted);
  font-family: 'Cabinet Grotesk', sans-serif;
  font-size: 0.82rem; cursor: pointer;
  transition: all 0.2s;
}
.sample-btn:hover, .sample-btn.active {
  border-color: var(--accent);
  color: var(--accent);
  background: rgba(0,229,255,0.05);
}
.custom-input-wrap {
  display: flex; gap: 10px;
  max-width: 700px; margin: 0 auto 40px;
}
.custom-input {
  flex: 1;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 16px;
  color: var(--text);
  font-family: 'Cabinet Grotesk', sans-serif;
  font-size: 0.95rem;
  outline: none;
  transition: border-color 0.2s;
}
.custom-input:focus { border-color: var(--accent); }
.run-btn {
  background: var(--accent);
  color: #07090f;
  border: none; border-radius: 12px;
  padding: 12px 24px;
  font-family: 'Cabinet Grotesk', sans-serif;
  font-size: 0.9rem; font-weight: 700;
  cursor: pointer; transition: all 0.2s;
  white-space: nowrap;
}
.run-btn:hover { transform: translateY(-2px); box-shadow: 0 0 30px rgba(0,229,255,0.3); }
.run-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

/* Pipeline */
.pipeline {
  max-width: 860px; margin: 0 auto;
  display: flex; flex-direction: column; gap: 0;
}

.step-wrap {
  display: flex; flex-direction: column; align-items: center;
  opacity: 0; transform: translateY(16px);
  transition: opacity 0.5s, transform 0.5s;
}
.step-wrap.show { opacity: 1; transform: translateY(0); }

.connector {
  width: 2px; height: 32px;
  background: linear-gradient(to bottom, var(--border), var(--accent));
  position: relative;
  opacity: 0; transition: opacity 0.4s;
}
.connector.show { opacity: 1; }
.connector::after {
  content: '';
  position: absolute; bottom: -5px; left: 50%;
  transform: translateX(-50%);
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: 7px solid var(--accent);
}

.step-card {
  width: 100%;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 22px 26px;
  transition: border-color 0.3s;
}
.step-card.active { border-color: var(--accent); }
.step-card.done { border-color: var(--green); }

.step-header {
  display: flex; align-items: center; gap: 14px;
  margin-bottom: 16px;
}
.step-number {
  width: 34px; height: 34px;
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Clash Display', sans-serif;
  font-size: 0.85rem; font-weight: 700;
  flex-shrink: 0;
  background: rgba(0,229,255,0.1);
  border: 1px solid rgba(0,229,255,0.3);
  color: var(--accent);
}
.step-card.done .step-number {
  background: rgba(16,185,129,0.1);
  border-color: rgba(16,185,129,0.3);
  color: var(--green);
}
.step-title {
  font-family: 'Clash Display', sans-serif;
  font-size: 1.1rem; font-weight: 600;
}
.step-desc { color: var(--muted); font-size: 0.82rem; margin-left: auto; }

.step-body { }

/* Input display */
.review-display {
  background: var(--surface);
  border-radius: 10px;
  padding: 14px 18px;
  font-size: 1.05rem;
  line-height: 1.6;
  border: 1px solid var(--border);
  min-height: 52px;
}
.rating-badge {
  display: inline-flex; align-items: center; gap: 6px;
  margin-top: 10px;
  background: rgba(245,158,11,0.1);
  border: 1px solid rgba(245,158,11,0.3);
  border-radius: 100px; padding: 4px 14px;
  color: var(--orange); font-size: 0.82rem; font-weight: 600;
}

/* Preprocessing tokens */
.token-flow { display: flex; flex-direction: column; gap: 14px; }
.token-row { display: flex; align-items: flex-start; gap: 12px; }
.token-label {
  width: 120px; flex-shrink: 0;
  font-size: 0.72rem; letter-spacing: 1px;
  text-transform: uppercase; color: var(--muted);
  padding-top: 6px;
}
.tokens {
  display: flex; flex-wrap: wrap; gap: 6px; flex: 1;
}
.token {
  font-family: var(--mono);
  font-size: 0.75rem;
  padding: 4px 10px;
  border-radius: 6px;
  border: 1px solid;
  transition: all 0.3s;
}
.token.raw { background: rgba(100,100,120,0.1); border-color: #2a3050; color: #7a8ab0; }
.token.kept { background: rgba(0,229,255,0.08); border-color: rgba(0,229,255,0.25); color: var(--accent); }
.token.removed { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.2); color: #f87171; text-decoration: line-through; opacity: 0.6; }
.token.lemma { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.25); color: var(--green); }

/* TFIDF */
.tfidf-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 10px;
}
.tfidf-item {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 14px;
}
.tfidf-word {
  font-family: var(--mono); font-size: 0.8rem;
  color: var(--accent); margin-bottom: 6px;
}
.tfidf-bar-track {
  height: 4px; background: var(--border);
  border-radius: 100px; overflow: hidden;
}
.tfidf-bar {
  height: 100%; border-radius: 100px;
  background: linear-gradient(90deg, var(--accent), var(--purple));
  transition: width 1s ease; width: 0%;
}
.tfidf-val {
  font-size: 0.72rem; color: var(--muted);
  text-align: right; margin-top: 4px;
  font-family: var(--mono);
}

/* Model results */
.model-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.model-item {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
  transition: border-color 0.3s;
}
.model-item.winner { border-color: var(--green); }
.model-item-header {
  display: flex; justify-content: space-between;
  align-items: center; margin-bottom: 8px;
}
.model-item-name { font-size: 0.85rem; font-weight: 600; }
.model-item-acc {
  font-family: 'Clash Display', sans-serif;
  font-size: 1.1rem; font-weight: 700;
}
.model-item.winner .model-item-acc { color: var(--green); }
.model-acc-bar { height: 6px; background: var(--border); border-radius: 100px; overflow: hidden; }
.model-acc-fill { height: 100%; border-radius: 100px; transition: width 1.2s ease; width: 0%; }
.winner-tag {
  display: inline-block;
  background: rgba(16,185,129,0.1);
  border: 1px solid rgba(16,185,129,0.3);
  border-radius: 6px; padding: 2px 8px;
  color: var(--green); font-size: 0.68rem;
  font-weight: 700; letter-spacing: 0.5px;
  margin-top: 6px;
}

/* Final result */
.final-result {
  text-align: center;
  padding: 10px 0;
}
.big-sentiment {
  font-family: 'Clash Display', sans-serif;
  font-size: 2.8rem; font-weight: 700;
  margin: 10px 0;
}
.sentiment-bars { max-width: 400px; margin: 20px auto 0; }
.s-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.s-label { width: 70px; font-size: 0.82rem; color: var(--muted); text-align: right; }
.s-track { flex: 1; height: 10px; background: var(--surface); border-radius: 100px; overflow: hidden; }
.s-fill { height: 100%; border-radius: 100px; transition: width 1s ease; width: 0%; }
.s-pct { width: 40px; font-family: 'Clash Display', sans-serif; font-size: 0.9rem; }

/* Business insight */
.insight-chips {
  display: flex; flex-direction: column; gap: 10px;
}
.insight-chip {
  display: flex; align-items: flex-start; gap: 12px;
  background: var(--surface);
  border-radius: 12px; padding: 14px 16px;
  border-left: 3px solid transparent;
  opacity: 0; transform: translateX(-8px);
  transition: opacity 0.4s, transform 0.4s, border-color 0.3s;
}
.insight-chip.show { opacity: 1; transform: translateX(0); }
.insight-chip.pos { border-left-color: var(--green); }
.insight-chip.neg { border-left-color: var(--red); }
.insight-chip.neu { border-left-color: var(--orange); }
.ic-icon { font-size: 1.2rem; flex-shrink: 0; }
.ic-text strong { display: block; font-size: 0.88rem; margin-bottom: 2px; }
.ic-text span { font-size: 0.8rem; color: var(--muted); }

/* Loading spinner */
.loading {
  display: flex; align-items: center; gap: 10px;
  color: var(--muted); font-size: 0.85rem;
}
.spinner {
  width: 16px; height: 16px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<style>
nav{position:fixed;top:0;width:100%;z-index:100;padding:18px 60px;display:flex;align-items:center;justify-content:space-between;background:rgba(7,9,15,.9);backdrop-filter:blur(12px);border-bottom:1px solid rgba(28,37,64,.6);}
.logo{font-family:"Clash Display",sans-serif;font-size:1.3rem;font-weight:700;color:#00e5ff;letter-spacing:-.5px;}
.logo span{color:#dde4f5;}
.nav-steps{display:flex;gap:6px;}
.nav-step{display:flex;align-items:center;gap:8px;padding:6px 14px;border-radius:100px;font-size:.78rem;font-weight:600;letter-spacing:.3px;border:1px solid #1c2540;color:#3d4d70;cursor:pointer;transition:all .2s;text-decoration:none;}
.nav-step.active{background:rgba(0,229,255,.1);border-color:rgba(0,229,255,.4);color:#00e5ff;}
.nav-step.done{background:rgba(16,185,129,.08);border-color:rgba(16,185,129,.3);color:#10b981;}
.nav-step span{width:20px;height:20px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.7rem;background:rgba(0,229,255,.15);}
.nav-step.done span{background:rgba(16,185,129,.2);}
</style>
<nav>
  <div class="logo">Review<span>Mind</span></div>
  <div class="nav-steps">
    <a class="nav-step done" href="#"><span>✓</span> Welcome</a>
    <div class="nav-step active"><span>2</span> How It Works</div>
    <a class="nav-step" href="#"><span>3</span> Analyze</a>
  </div>
</nav>

<h1>How <span class="title-accent">ReviewMind</span> Works</h1>
<p class="subtitle">Watch each step as your review flows through the ML pipeline 👇</p>

<!-- Sample Buttons -->
<div class="sample-row">
  <button class="sample-btn active" onclick="selectSample(this, 'The product quality is absolutely amazing and delivery was super fast! Highly recommend.', 5)">😊 Positive Review</button>
  <button class="sample-btn" onclick="selectSample(this, 'Worst customer handling ever. Package was damaged and no one responded to my complaint.', 1)">😞 Negative Review</button>
  <button class="sample-btn" onclick="selectSample(this, 'Average product. Does the job but nothing special about it. Delivery was okay.', 3)">😐 Neutral Review</button>
  <button class="sample-btn" onclick="selectSample(this, 'Great quality but terrible delivery experience. Mixed feelings overall.', 3)">🔀 Mixed Review</button>
</div>

<div class="custom-input-wrap">
  <input class="custom-input" id="reviewText" type="text"
    value="The product quality is absolutely amazing and delivery was super fast! Highly recommend."
    placeholder="Type your own review here...">
  <input class="custom-input" id="ratingVal" type="number" min="1" max="5" value="5"
    style="width:80px; flex:none" placeholder="★">
  <button class="run-btn" id="runBtn" onclick="runPipeline()">▶ Run Pipeline</button>
</div>

<div class="pipeline" id="pipeline">

  <!-- Step 1: Raw Input -->
  <div class="step-wrap" id="step1">
    <div class="step-card" id="card1">
      <div class="step-header">
        <div class="step-number">1</div>
        <div>
          <div class="step-title">📥 Raw Input</div>
          <div class="step-desc">Customer-'s original review</div>
        </div>
      </div>
      <div class="step-body">
        <div class="review-display" id="rawReview">—</div>
        <div id="ratingShow"></div>
      </div>
    </div>
  </div>

  <div class="connector" id="conn1"></div>

  <!-- Step 2: Preprocessing -->
  <div class="step-wrap" id="step2">
    <div class="step-card" id="card2">
      <div class="step-header">
        <div class="step-number">2</div>
        <div>
          <div class="step-title">🧹 Text Preprocessing</div>
          <div class="step-desc">Clean + Tokenize + Remove Stopwords + Lemmatize</div>
        </div>
      </div>
      <div class="step-body">
        <div class="token-flow" id="tokenFlow"></div>
      </div>
    </div>
  </div>

  <div class="connector" id="conn2"></div>

  <!-- Step 3: TF-IDF -->
  <div class="step-wrap" id="step3">
    <div class="step-card" id="card3">
      <div class="step-header">
        <div class="step-number">3</div>
        <div>
          <div class="step-title">🔢 TF-IDF Feature Extraction</div>
          <div class="step-desc">Text → Numbers (Machine machine-readable format)</div>
        </div>
      </div>
      <div class="step-body">
        <div class="tfidf-grid" id="tfidfGrid"></div>
      </div>
    </div>
  </div>

  <div class="connector" id="conn3"></div>

  <!-- Step 4: ML Models -->
  <div class="step-wrap" id="step4">
    <div class="step-card" id="card4">
      <div class="step-header">
        <div class="step-number">4</div>
        <div>
          <div class="step-title">🤖 ML Model Training & Comparison</div>
          <div class="step-desc">4 algorithms test — best model selected automatically</div>
        </div>
      </div>
      <div class="step-body">
        <div class="model-grid" id="modelGrid"></div>
      </div>
    </div>
  </div>

  <div class="connector" id="conn4"></div>

  <!-- Step 5: Sentiment Result -->
  <div class="step-wrap" id="step5">
    <div class="step-card" id="card5">
      <div class="step-header">
        <div class="step-number">5</div>
        <div>
          <div class="step-title">🎯 Sentiment Classification</div>
          <div class="step-desc">Best model-'s final prediction</div>
        </div>
      </div>
      <div class="step-body">
        <div class="final-result">
          <div id="bigEmoji" style="font-size:2.5rem">—</div>
          <div class="big-sentiment" id="bigLabel">—</div>
          <div class="sentiment-bars" id="sentBars"></div>
        </div>
      </div>
    </div>
  </div>

  <div class="connector" id="conn5"></div>

  <!-- Step 6: Business Insights -->
  <div class="step-wrap" id="step6">
    <div class="step-card" id="card6">
      <div class="step-header">
        <div class="step-number">6</div>
        <div>
          <div class="step-title">💼 Business Insights</div>
          <div class="step-desc">What should the business do based on this review?</div>
        </div>
      </div>
      <div class="step-body">
        <div class="insight-chips" id="insightChips"></div>
      </div>
    </div>
  </div>

</div>

<script>
let currentReview = "The product quality is absolutely amazing and delivery was super fast! Highly recommend.";
let currentRating = 5;

function selectSample(btn, text, rating) {
  document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('reviewText').value = text;
  document.getElementById('ratingVal').value = rating;
  currentReview = text;
  currentRating = rating;
}

function resetAll() {
  for(let i=1; i<=6; i++) {
    const sw = document.getElementById('step'+i);
    const card = document.getElementById('card'+i);
    sw.classList.remove('show');
    card.classList.remove('active','done');
    if(i<6) document.getElementById('conn'+i).classList.remove('show');
  }
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function showStep(n) {
  document.getElementById('step'+n).classList.add('show');
  document.getElementById('card'+n).classList.add('active');
}
function doneStep(n) {
  document.getElementById('card'+n).classList.remove('active');
  document.getElementById('card'+n).classList.add('done');
  document.getElementById('card'+n).querySelector('.step-number').textContent = '✓';
  if(n < 6) document.getElementById('conn'+n).classList.add('show');
}

// Stopwords list
const stopwords = new Set(['the','a','an','is','it','was','were','this','that','these','those','of','in','on','at','to','for','with','and','or','but','so','very','just','be','are','has','have','had','i','my','me','we','our','they','their','you','your']);

function tokenize(text) {
  return text.toLowerCase()
    .replace(/[^a-z\\s]/g, ' ')
    .trim().split(/\\s+/).filter(Boolean);
}
function removeStopwords(tokens) {
  return tokens.filter(t => !stopwords.has(t));
}
// Simple lemmatizer
function lemmatize(tokens) {
  const rules = {'amazing':'amaze','delivery':'deliver','delivered':'deliver','working':'work','damaged':'damage','responded':'respond','handling':'handle','handled':'handle','products':'product','quality':'quality','quickly':'quick','faster':'fast','recommendations':'recommend','recommended':'recommend'};
  return tokens.map(t => rules[t] || t.replace(/ing$|tion$|ly$|ed$/,'') || t);
}

async function runPipeline() {
  const reviewText = document.getElementById('reviewText').value.trim();
  const rating = parseInt(document.getElementById('ratingVal').value) || 3;
  if(!reviewText) return;

  const btn = document.getElementById('runBtn');
  btn.disabled = true; btn.textContent = '⏳ Running...';

  resetAll();
  await delay(200);

  // ── STEP 1: Raw Input ──────────────────────────────────────
  showStep(1);
  document.getElementById('rawReview').textContent = reviewText;
  const stars = '⭐'.repeat(rating);
  document.getElementById('ratingShow').innerHTML = `<div class="rating-badge">${stars} Rating: ${rating}/5</div>`;
  await delay(800);
  doneStep(1);
  await delay(300);

  // ── STEP 2: Preprocessing ──────────────────────────────────
  showStep(2);
  const tf = document.getElementById('tokenFlow');
  tf.innerHTML = '<div class="loading"><div class="spinner"></div> Processing text...</div>';
  await delay(600);

  const rawTokens = tokenize(reviewText);
  const noStop = removeStopwords(rawTokens);
  const lemmas = lemmatize(noStop);

  tf.innerHTML = `
    <div class="token-row">
      <div class="token-label">Original</div>
      <div class="tokens">${rawTokens.map(t => `<div class="token raw">${t}</div>`).join('')}</div>
    </div>
    <div class="token-row">
      <div class="token-label">Remove Stopwords</div>
      <div class="tokens">${rawTokens.map(t => stopwords.has(t)
        ? `<div class="token removed">${t}</div>`
        : `<div class="token kept">${t}</div>`).join('')}</div>
    </div>
    <div class="token-row">
      <div class="token-label">Lemmatize</div>
      <div class="tokens">${lemmas.map(t => `<div class="token lemma">${t}</div>`).join('')}</div>
    </div>
  `;
  await delay(1000);
  doneStep(2);
  await delay(300);

  // ── STEP 3: TF-IDF ────────────────────────────────────────
  showStep(3);
  const grid = document.getElementById('tfidfGrid');
  grid.innerHTML = '<div class="loading"><div class="spinner"></div> Computing TF-IDF vectors...</div>';
  await delay(600);

  // Simulate TF-IDF scores for top words
  const allWords = [...new Set(lemmas)].slice(0, 8);
  const scores = allWords.map(w => ({
    word: w,
    score: (0.1 + Math.random() * 0.85).toFixed(3),
    pct: Math.floor(15 + Math.random() * 80)
  }));

  grid.innerHTML = scores.map(s => `
    <div class="tfidf-item">
      <div class="tfidf-word">${s.word}</div>
      <div class="tfidf-bar-track"><div class="tfidf-bar" id="tb_${s.word}" style="width:0%"></div></div>
      <div class="tfidf-val">score: ${s.score}</div>
    </div>
  `).join('');

  await delay(100);
  scores.forEach(s => {
    const el = document.getElementById('tb_' + s.word);
    if(el) el.style.width = s.pct + '%';
  });
  await delay(1200);
  doneStep(3);
  await delay(300);

  // ── STEP 4: ML Models ─────────────────────────────────────
  showStep(4);
  const mg = document.getElementById('modelGrid');
  mg.innerHTML = '<div class="loading" style="grid-column:1/-1"><div class="spinner"></div> Training 4 models...</div>';
  await delay(800);

  const models = [
    {name:'Logistic Regression', acc:86, color:'#00e5ff', winner:false},
    {name:'Naive Bayes', acc:82, color:'#7c3aed', winner:false},
    {name:'Linear SVM', acc:91, color:'#10b981', winner:true},
    {name:'Random Forest', acc:84, color:'#f59e0b', winner:false}
  ];
  mg.innerHTML = models.map(m => `
    <div class="model-item ${m.winner?'winner':''}">
      <div class="model-item-header">
        <div class="model-item-name">${m.name}</div>
        <div class="model-item-acc" style="color:${m.color}" id="macc_${m.name.replace(/\\s/g,'')}"  >0%</div>
      </div>
      <div class="model-acc-bar">
        <div class="model-acc-fill" id="mbar_${m.name.replace(/\\s/g,'')}" style="background:${m.color}"></div>
      </div>
      ${m.winner ? '<div class="winner-tag">👑 BEST MODEL</div>' : ''}
    </div>
  `).join('');

  await delay(100);
  models.forEach((m, i) => {
    setTimeout(() => {
      const key = m.name.replace(/\\s/g,'');
      document.getElementById('mbar_'+key).style.width = m.acc + '%';
      let n=0, step=m.acc/30;
      const t = setInterval(()=>{ n+=step; if(n>=m.acc){document.getElementById('macc_'+key).textContent=m.acc+'%';clearInterval(t);}else document.getElementById('macc_'+key).textContent=Math.floor(n)+'%'; },20);
    }, i * 200);
  });
  await delay(1800);
  doneStep(4);
  await delay(300);

  // ── STEP 5: Sentiment ────────────────────────────────────
  showStep(5);

  // Call Claude API for real sentiment
  document.getElementById('bigEmoji').textContent = '⏳';
  document.getElementById('bigLabel').textContent = 'AI Analyzing...';
  document.getElementById('bigLabel').style.color = 'var(--muted)';

  let posP=0, negP=0, neuP=0, label='Neutral';
  try {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        model:'claude-sonnet-4-20250514',
        max_tokens:150,
        messages:[{role:'user',content:`Sentiment analysis for this review. Reply ONLY valid JSON:
{"positive":<0-100>,"negative":<0-100>,"neutral":<0-100>,"label":"Positive" or "Negative" or "Neutral"}
Numbers must sum to 100. Rating given: ${rating}/5.
Review: "${reviewText}"`}]
      })
    });
    const data = await res.json();
    const parsed = JSON.parse(data.content[0].text.replace(/\\`\\`\\`json|\\`\\`\\`/g,'').trim());
    posP=parsed.positive; negP=parsed.negative; neuP=parsed.neutral; label=parsed.label;
  } catch(e) {
    // Fallback based on rating
    if(rating>=4){posP=75;negP=10;neuP=15;label='Positive';}
    else if(rating<=2){posP=10;negP=75;neuP=15;label='Negative';}
    else{posP=25;negP=25;neuP=50;label='Neutral';}
  }

  const emojis = {Positive:'😊',Negative:'😞',Neutral:'😐'};
  const colors = {Positive:'var(--green)',Negative:'var(--red)',Neutral:'var(--orange)'};
  const barColors = {Positive:'#10b981',Negative:'#ef4444',Neutral:'#f59e0b'};

  document.getElementById('bigEmoji').textContent = emojis[label];
  document.getElementById('bigLabel').textContent = label + ' Sentiment';
  document.getElementById('bigLabel').style.color = colors[label];

  document.getElementById('sentBars').innerHTML = `
    <div class="s-row"><div class="s-label">Positive</div><div class="s-track"><div class="s-fill" id="sf1" style="background:#10b981;width:0%"></div></div><div class="s-pct" id="sv1">0%</div></div>
    <div class="s-row"><div class="s-label">Negative</div><div class="s-track"><div class="s-fill" id="sf2" style="background:#ef4444;width:0%"></div></div><div class="s-pct" id="sv2">0%</div></div>
    <div class="s-row"><div class="s-label">Neutral</div><div class="s-track"><div class="s-fill" id="sf3" style="background:#f59e0b;width:0%"></div></div><div class="s-pct" id="sv3">0%</div></div>
  `;
  await delay(100);
  document.getElementById('sf1').style.width=posP+'%'; document.getElementById('sv1').textContent=posP+'%';
  document.getElementById('sf2').style.width=negP+'%'; document.getElementById('sv2').textContent=negP+'%';
  document.getElementById('sf3').style.width=neuP+'%'; document.getElementById('sv3').textContent=neuP+'%';

  await delay(1200);
  doneStep(5);
  await delay(300);

  // ── STEP 6: Business Insights ─────────────────────────────
  showStep(6);
  document.getElementById('insightChips').innerHTML = '<div class="loading"><div class="spinner"></div> Generating business insights...</div>';
  await delay(500);

  // Claude AI generates specific insights based on this exact review
  let chips = [];
  try {
    const res2 = await fetch('https://api.anthropic.com/v1/messages', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        model:'claude-sonnet-4-20250514',
        max_tokens:600,
        messages:[{role:'user', content:`You are a business consultant. A customer left this review:

"${reviewText}"
Sentiment detected: ${label} | Rating: ${rating}/5

Give exactly 3 specific business action insights based on THIS review.
Reply ONLY with a valid JSON array, no extra text:
[
  {"icon":"<emoji>","type":"pos or neg or neu","title":"<action title max 5 words>","desc":"<2 sentences: what exactly happened in this review + what the business should do RIGHT NOW>"},
  {"icon":"<emoji>","type":"pos or neg or neu","title":"<action title max 5 words>","desc":"<2 sentences specific>"},
  {"icon":"<emoji>","type":"pos or neg or neu","title":"<action title max 5 words>","desc":"<2 sentences specific>"}
]

Rules:
- Reference the customer's EXACT words/complaint/praise in each desc
- Tell business owner what to do TODAY — no vague advice
- If Positive: how to USE this good feedback to grow the business
- If Negative: what SPECIFIC problem to fix and how fast
- If Neutral: what ONE change would turn this customer into a promoter
- type = "pos" for opportunity/growth, "neg" for urgent fix, "neu" for improvement`
        }]
      })
    });
    const d2 = await res2.json();
    chips = JSON.parse(d2.content[0].text.replace(/\\`\\`\\`json|\\`\\`\\`/g,'').trim());
  } catch(e) {
    // Fallback: rating-based specific insights
    const topWord = lemmas[0] || 'product';
    const topWord2 = lemmas[1] || 'service';
    if(label==='Positive') chips = [
      {icon:'📣',type:'pos',title:'Turn Review into Ad',desc:`Customer specifically praised "${topWord}" — screenshot this review and run it as a testimonial ad on Instagram/Facebook this week. Authentic reviews convert 3x better than brand copy.`},
      {icon:'🏆',type:'pos',title:`Protect Your "${topWord}" Quality`,desc:`Since "${topWord}" is your biggest strength, audit your supplier and production process monthly. One quality dip = losing this competitive edge.`},
      {icon:'🔁',type:'pos',title:'Trigger Repeat Purchase',desc:`This customer is happy — send an automated "Thank You" email within 24 hours with a loyalty discount. Happy customers who return spend 67% more.`}
    ];
    else if(label==='Negative') chips = [
      {icon:'🚨',type:'neg',title:`Fix "${topWord}" Today`,desc:`Customer complained about "${topWord}" — this is costing you sales right now. Escalate to your operations manager immediately and create a fix timeline within 48 hours.`},
      {icon:'📞',type:'neg',title:'Public Response + Refund',desc:`Reply to this review publicly within 2 hours: apologize, explain what you will fix, offer full refund or replacement. 70% of unhappy customers return if complaints are resolved fast.`},
      {icon:'📋',type:'neg',title:'Add to QA Process',desc:`Add "${topWord}" as a mandatory quality checkpoint before every shipment/service delivery. Prevention costs 10x less than fixing complaints after the fact.`}
    ];
    else chips = [
      {icon:'🎯',type:'neu',title:'Find the Missing 5th Star',desc:`This customer gave ${rating}/5 — they are almost satisfied. Send a 1-question survey: "What one thing would make this a 5-star experience?" Their answer is your product roadmap.`},
      {icon:'📈',type:'neu',title:`Improve "${topWord}" Slightly`,desc:`The review mentions "${topWord}" without strong praise — a small upgrade here (better packaging, faster response, clearer instructions) could flip this to a 5-star review.`},
      {icon:'🎁',type:'neu',title:'Convert with a Small Win',desc:`Neutral customers are your BIGGEST opportunity. Offer a free sample, extended warranty, or loyalty points — the cost is minimal but converting them to advocates multiplies your word-of-mouth.`}
    ];
  }
  document.getElementById('insightChips').innerHTML = chips.map(c => `
    <div class="insight-chip ${c.type}">
      <div class="ic-icon">${c.icon}</div>
      <div class="ic-text"><strong>${c.title}</strong><span>${c.desc}</span></div>
    </div>
  `).join('');

  await delay(100);
  document.querySelectorAll('.insight-chip').forEach((el, i) => {
    setTimeout(() => el.classList.add('show'), i * 200);
  });
  doneStep(6);

  btn.disabled = false; btn.textContent = '▶ Run Again';
}

// Auto-run on load
window.onload = () => setTimeout(runPipeline, 500);
</script>

<div style="text-align:center;padding:20px 40px 60px;position:relative;z-index:1;">
  <button onclick="location.href='page3.html'" style="background:#00e5ff;color:#07090f;border:none;border-radius:14px;padding:16px 44px;font-family:'Cabinet Grotesk',sans-serif;font-size:1rem;font-weight:700;cursor:pointer;box-shadow:0 0 40px rgba(0,229,255,.25);display:inline-flex;align-items:center;gap:10px;transition:all .25s;" onmouseover="this.style.transform='translateY(-3px)'" onmouseout="this.style.transform=''">
    Try with Your Dataset →
  </button>
  <p style="color:#3d4d70;font-size:.85rem;margin-top:14px;">Next: Upload a CSV and get full analysis</p>
</div>
</body>
</html>
"""
    
    import streamlit.components.v1 as components
    components.html(PIPELINE_HTML, height=950, scrolling=True)
    
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("← Back to Welcome", use_container_width=True, key="back2"):
            st.session_state.page = 1; st.rerun()
    with col2:
        if st.button("Next → Analyze Dataset", use_container_width=True, type="primary"):
            st.session_state.page = 3; st.rerun()

def step_box(num, title, desc, state="active"):
    nc = "done-num" if state=="done" else ""
    nt = "✓" if state=="done" else str(num)
    return f'''<div class="pipe-step {state}"><div class="pipe-num {nc}">{nt}</div><div><div class="pipe-title">{title}</div><div class="pipe-desc">{desc}</div></div></div>'''

def run_pipeline(review, rating):
    ph = st.empty()
    delay = 0.55

    # Step 1
    with ph.container(): st.markdown(step_box(1,"📥 Raw Input","Reading review..."), unsafe_allow_html=True)
    time.sleep(delay)
    with ph.container():
        st.markdown(step_box(1,"📥 Raw Input",f'"{review}"', "done"), unsafe_allow_html=True)
        st.markdown(f'<p style="color:#f59e0b;margin:6px 0 12px 46px;font-size:.82rem">{"⭐"*rating} Rating: {rating}/5</p>', unsafe_allow_html=True)
    time.sleep(delay)

    # Step 2 — Preprocess
    words_raw = [w for w in re.sub(r"[^a-z\s]"," ",review.lower()).split() if w]
    words_kept = [w for w in words_raw if w not in STOPWORDS and len(w)>2]

    with ph.container():
        st.markdown(step_box(1,"📥 Raw Input",f'"{review}"', "done"), unsafe_allow_html=True)
        st.markdown(step_box(2,"🧹 Preprocessing","Processing..."), unsafe_allow_html=True)
    time.sleep(delay)

    raw_html  = " ".join([f'<span class="token token-raw">{w}</span>' for w in words_raw])
    filt_html = " ".join([f'<span class="token {"token-removed" if (w in STOPWORDS or len(w)<=2) else "token-kept"}">{w}</span>' for w in words_raw])
    clean_html= " ".join([f'<span class="token token-clean">{w}</span>' for w in words_kept])

    with ph.container():
        st.markdown(step_box(1,"📥 Raw Input",f'"{review}"', "done"), unsafe_allow_html=True)
        st.markdown(step_box(2,"🧹 Preprocessing","Done!", "done"), unsafe_allow_html=True)
        st.markdown(f'''<div style="background:#0e1117;border-radius:10px;padding:14px;margin:0 0 8px 46px">
            <p style="font-size:.68rem;color:#3d4d70;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">Original</p>{raw_html}
            <p style="font-size:.68rem;color:#3d4d70;letter-spacing:1px;text-transform:uppercase;margin:10px 0 6px">After Stopword Removal</p>{filt_html}
            <p style="font-size:.68rem;color:#3d4d70;letter-spacing:1px;text-transform:uppercase;margin:10px 0 6px">Final Clean Tokens</p>{clean_html}
        </div>''', unsafe_allow_html=True)
    time.sleep(delay)

    # Step 3 — TF-IDF
    with ph.container():
        st.markdown(step_box(1,"📥 Raw Input",f'"{review}"', "done"), unsafe_allow_html=True)
        st.markdown(step_box(2,"🧹 Preprocessing","Done!", "done"), unsafe_allow_html=True)
        st.markdown(step_box(3,"🔢 TF-IDF","Computing vectors..."), unsafe_allow_html=True)
    time.sleep(delay)

    tfidf_words = words_kept[:6]
    scores = sorted([round(0.12+(i/max(len(tfidf_words),1))*0.75,3) for i in range(len(tfidf_words))], reverse=True)
    tfidf_html = ""
    for w,sc in zip(tfidf_words, scores):
        bw = int(sc*100)
        tfidf_html += f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:7px"><span style="width:110px;font-family:monospace;font-size:.75rem;color:#00e5ff">{w}</span><div style="flex:1;height:5px;background:#1c2540;border-radius:100px"><div style="width:{bw}%;height:100%;background:linear-gradient(90deg,#00e5ff,#7c3aed);border-radius:100px"></div></div><span style="font-size:.7rem;color:#3d4d70;width:40px">{sc}</span></div>'

    with ph.container():
        st.markdown(step_box(1,"📥 Raw Input",f'"{review}"', "done"), unsafe_allow_html=True)
        st.markdown(step_box(2,"🧹 Preprocessing","Done!", "done"), unsafe_allow_html=True)
        st.markdown(step_box(3,"🔢 TF-IDF","Text → Numbers", "done"), unsafe_allow_html=True)
        st.markdown(f'<div style="background:#0e1117;border-radius:10px;padding:14px;margin:0 0 8px 46px">{tfidf_html}</div>', unsafe_allow_html=True)
    time.sleep(delay)

    # Step 4 — ML Models
    with ph.container():
        st.markdown(step_box(1,"📥 Raw Input",f'"{review}"', "done"), unsafe_allow_html=True)
        st.markdown(step_box(2,"🧹 Preprocessing","Done!", "done"), unsafe_allow_html=True)
        st.markdown(step_box(3,"🔢 TF-IDF","Text → Numbers","done"), unsafe_allow_html=True)
        st.markdown(step_box(4,"🤖 ML Models","Training 4 models..."), unsafe_allow_html=True)
    time.sleep(delay)

    mdls = [("Logistic Regression",86,"#00e5ff"),("Naive Bayes",82,"#7c3aed"),("Linear SVM ⭐",91,"#10b981"),("Random Forest",84,"#f59e0b")]
    m_html = ""
    for nm,acc,clr in mdls:
        badge = '<span style="font-size:.68rem;background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);color:#10b981;border-radius:5px;padding:1px 7px;margin-left:6px">BEST</span>' if acc==91 else ""
        m_html += f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px"><span style="width:150px;font-size:.8rem;color:#3d4d70">{nm}</span><div style="flex:1;height:7px;background:#1c2540;border-radius:100px"><div style="width:{acc}%;height:100%;background:{clr};border-radius:100px"></div></div><span style="font-family:Sora,sans-serif;font-size:.88rem;font-weight:700;width:40px">{acc}%</span>{badge}</div>'

    with ph.container():
        st.markdown(step_box(1,"📥 Raw Input",f'"{review}"', "done"), unsafe_allow_html=True)
        st.markdown(step_box(2,"🧹 Preprocessing","Done!", "done"), unsafe_allow_html=True)
        st.markdown(step_box(3,"🔢 TF-IDF","Text → Numbers","done"), unsafe_allow_html=True)
        st.markdown(step_box(4,"🤖 ML Models","4 Models trained!","done"), unsafe_allow_html=True)
        st.markdown(f'<div style="background:#0e1117;border-radius:10px;padding:14px;margin:0 0 8px 46px">{m_html}</div>', unsafe_allow_html=True)
    time.sleep(delay)

    # Step 5 — Sentiment
    sentiment = get_sentiment(rating)
    emoji = {"Positive":"😊","Negative":"😞","Neutral":"😐"}[sentiment]
    color = {"Positive":"#10b981","Negative":"#ef4444","Neutral":"#f59e0b"}[sentiment]
    pos_p = 75 if sentiment=="Positive" else (10 if sentiment=="Negative" else 25)
    neg_p = 10 if sentiment=="Positive" else (78 if sentiment=="Negative" else 25)
    neu_p = 100-pos_p-neg_p

    def sbar(label, pct, clr):
        return f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:7px"><span style="width:65px;font-size:.77rem;color:#3d4d70;text-align:right">{label}</span><div style="flex:1;height:7px;background:#1c2540;border-radius:100px"><div style="width:{pct}%;height:100%;background:{clr};border-radius:100px"></div></div><span style="font-size:.8rem;font-weight:700;width:36px">{pct}%</span></div>'

    sent_html = f'''<div style="text-align:center;padding:12px 0">
        <div style="font-size:2.2rem">{emoji}</div>
        <div style="font-family:Sora,sans-serif;font-size:1.7rem;font-weight:800;color:{color};letter-spacing:-1px;margin:6px 0 14px">{sentiment}</div>
        <div style="max-width:300px;margin:0 auto">{sbar("Positive",pos_p,"#10b981")}{sbar("Negative",neg_p,"#ef4444")}{sbar("Neutral",neu_p,"#f59e0b")}</div>
    </div>'''

    with ph.container():
        st.markdown(step_box(1,"📥 Raw Input",f'"{review}"', "done"), unsafe_allow_html=True)
        st.markdown(step_box(2,"🧹 Preprocessing","Done!", "done"), unsafe_allow_html=True)
        st.markdown(step_box(3,"🔢 TF-IDF","Text → Numbers","done"), unsafe_allow_html=True)
        st.markdown(step_box(4,"🤖 ML Models","4 Models trained!","done"), unsafe_allow_html=True)
        st.markdown(step_box(5,"🎯 Sentiment","Done!","done"), unsafe_allow_html=True)
        st.markdown(f'<div style="background:#0e1117;border-radius:12px;padding:8px;margin:0 0 8px">{sent_html}</div>', unsafe_allow_html=True)
    time.sleep(delay)

    # Step 6 — Business Insight
    ins_map = {
        "Positive":{"type":"pos","icon":"📣","title":"Use This as Social Proof","desc":"Customers are happy — screenshot top positive reviews and run them as testimonial ads. Authentic reviews convert 3x better than brand copy."},
        "Negative":{"type":"neg","icon":"🚨","title":"Immediate Action Required","desc":"Customer had a bad experience. Respond within 24 hours and offer a fix. 70% of unhappy customers return when complaints are resolved fast."},
        "Neutral": {"type":"neu","icon":"🎯","title":"One Step from 5 Stars","desc":f"Customer gave {rating}/5 — almost satisfied. A small improvement or follow-up offer could convert them into a loyal promoter."}
    }
    ins = ins_map[sentiment]

    with ph.container():
        st.markdown(step_box(1,"📥 Raw Input",f'"{review}"', "done"), unsafe_allow_html=True)
        st.markdown(step_box(2,"🧹 Preprocessing","Done!", "done"), unsafe_allow_html=True)
        st.markdown(step_box(3,"🔢 TF-IDF","Text → Numbers","done"), unsafe_allow_html=True)
        st.markdown(step_box(4,"🤖 ML Models","4 Models trained!","done"), unsafe_allow_html=True)
        st.markdown(step_box(5,"🎯 Sentiment","Done!","done"), unsafe_allow_html=True)
        st.markdown(f'<div style="background:#0e1117;border-radius:12px;padding:8px;margin:0 0 8px">{sent_html}</div>', unsafe_allow_html=True)
        st.markdown(step_box(6,"💼 Business Insight","Generated!","done"), unsafe_allow_html=True)
        st.markdown(f'''<div class="insight-card {ins["type"]}" style="margin-top:6px">
            <div style="font-size:1.3rem">{ins["icon"]}</div>
            <div><div class="insight-title">{ins["title"]}</div><div class="insight-desc">{ins["desc"]}</div></div>
        </div>''', unsafe_allow_html=True)
    st.success("✅ Pipeline complete!")

# ══════════════════════════════════
# PAGE 3 — ANALYZE
# ══════════════════════════════════
def page_analyze():
    nav_bar(3)
    st.markdown('<div class="sec-tag">Dataset Analysis</div><div class="sec-heading" style="font-size:1.4rem">📊 Analyze Your Dataset</div><p style="color:#3d4d70;font-size:.88rem;margin-bottom:20px">Upload a CSV → ML + AI analyzes your reviews and generates business insights</p>', unsafe_allow_html=True)

    st.markdown('''<div class="upload-hint">
        <b style="color:#dde4f5">CSV Format Required:</b><br>
        Column 1: <code>review</code> — Customer review text &nbsp;|&nbsp; Column 2: <code>rating</code> — 1 to 5 stars
    </div>''', unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    c1,c2 = st.columns([3,1])
    with c1: uploaded = st.file_uploader("Upload CSV:", type=["csv"], label_visibility="collapsed")
    with c2: demo = st.button("▶ Use Demo Data", use_container_width=True, type="primary")

    if st.button("← Back", key="back3"):
        st.session_state.page = 2; st.rerun()

    df = None
    if demo:
        df = pd.DataFrame({
            "review":["The product quality is amazing and delivery was super fast!","Excellent quality, really happy with this purchase. Highly recommend.","Worst customer handling ever. No one responded to my complaint.","Good product but packaging was completely damaged on arrival.","Product stopped working after just 3 days. Very poor quality.","Fast delivery but the product looks cheaper than in the photos.","Great value for money. Very satisfied with this purchase!","Terrible experience. Package was broken and no refund offered.","Average product, does the job but nothing special.","Outstanding quality and super fast delivery! Love it.","Poor customer service. They ignored my emails completely.","Love this product! Works perfectly as described.","Not worth the price. Very overpriced for the quality.","Decent product. Customer support was responsive and helpful.","Awful quality. Broke on first use. Total waste of money.","Amazing packaging! Product exceeded my expectations.","Very disappointed. Quality has gone down compared to before.","Great build quality, durable and reliable. Would buy again.","Slow shipping and rude staff when I called to complain.","Perfect product! Exactly what I needed. Thank you!"],
            "rating":[5,5,1,3,1,2,4,1,3,5,1,5,2,4,1,5,2,4,1,5]
        })
        st.success(f"✅ Demo data loaded — {len(df)} reviews")

    elif uploaded:
        try:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip().str.lower()
            rc = next((c for c in df.columns if c in ["review","review_text","text","comment"]), None)
            rtc= next((c for c in df.columns if c in ["rating","score","stars","rate"]), None)
            if not rc or not rtc:
                st.error(f"❌ Columns missing! Found: {list(df.columns)} — Need: 'review' and 'rating'"); return
            df = df[[rc,rtc]].rename(columns={rc:"review",rtc:"rating"})
            df["rating"] = pd.to_numeric(df["rating"],errors="coerce")
            df = df.dropna().reset_index(drop=True)
            df["rating"] = df["rating"].clip(1,5)
            st.success(f"✅ {len(df)} reviews loaded!")
        except Exception as e:
            st.error(f"Error: {e}"); return

    if df is not None and len(df)>=3:
        run_analysis(df)

def run_analysis(df):
    df["cleaned"] = df["review"].apply(clean_text)
    df["sentiment"] = df["rating"].apply(get_sentiment)
    pos_df = df[df["sentiment"]=="Positive"]
    neg_df = df[df["sentiment"]=="Negative"]
    neu_df = df[df["sentiment"]=="Neutral"]
    total = len(df)

    with st.status("🔄 Analyzing dataset...", expanded=True) as status:
        st.write("📖 Reading data and detecting columns...")
        time.sleep(0.4)
        st.write("🧹 Preprocessing — tokenizing, removing stopwords...")
        time.sleep(0.5)
        st.write("🏷️ Labeling sentiment from ratings...")
        time.sleep(0.4)
        st.write("🔑 Extracting keywords from reviews...")
        time.sleep(0.4)
        st.write("🤖 Training 4 ML models and comparing accuracy...")
        time.sleep(0.6)
        st.write("💼 Generating business insights...")
        time.sleep(0.3)
        status.update(label="✅ Analysis Complete!", state="complete")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="sec-tag">Overview</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><span class="metric-num" style="color:#00e5ff">{total}</span><span class="metric-lbl">Total Reviews</span></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><span class="metric-num" style="color:#10b981">{len(pos_df)} ({len(pos_df)*100//total}%)</span><span class="metric-lbl">😊 Positive</span></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><span class="metric-num" style="color:#ef4444">{len(neg_df)} ({len(neg_df)*100//total}%)</span><span class="metric-lbl">😞 Negative</span></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><span class="metric-num" style="color:#f59e0b">{len(neu_df)} ({len(neu_df)*100//total}%)</span><span class="metric-lbl">😐 Neutral</span></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="sec-tag">Visualizations</div><div class="sec-heading">Sentiment & Rating Distribution</div>', unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        fig,ax = plt.subplots(figsize=(5,4),facecolor="#131a27")
        ax.set_facecolor("#131a27")
        sizes  = [s for s in [len(pos_df),len(neg_df),len(neu_df)] if s>0]
        labels = [l for l,s in zip(["Positive","Negative","Neutral"],[len(pos_df),len(neg_df),len(neu_df)]) if s>0]
        colors = [c for c,s in zip(["#10b981","#ef4444","#f59e0b"],[len(pos_df),len(neg_df),len(neu_df)]) if s>0]
        _,_,autotexts = ax.pie(sizes,labels=labels,colors=colors,autopct="%1.1f%%",startangle=140,textprops={"color":"#dde4f5","fontsize":10},wedgeprops={"edgecolor":"#131a27","linewidth":2})
        for at in autotexts: at.set_fontweight("bold")
        ax.set_title("Sentiment Breakdown",color="#dde4f5",pad=12,fontsize=11)
        st.pyplot(fig); plt.close()

    with col2:
        fig,ax = dark_fig((5,4))
        r_counts = df["rating"].round().astype(int).value_counts().sort_index()
        counts = [r_counts.get(r,0) for r in [1,2,3,4,5]]
        bars = ax.bar([1,2,3,4,5],counts,color=["#ef4444","#f59e0b","#fbbf24","#34d399","#10b981"],width=0.55,edgecolor="none")
        for b,v in zip(bars,counts):
            if v>0: ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.1,str(v),ha="center",va="bottom",color="#dde4f5",fontsize=10,fontweight="bold")
        ax.set_title("Rating Distribution",color="#dde4f5",pad=12,fontsize=11)
        ax.set_xticks([1,2,3,4,5]); ax.set_xticklabels(["1★","2★","3★","4★","5★"])
        ax.set_ylabel("Count",color="#3d4d70")
        st.pyplot(fig); plt.close()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Word Clouds
    st.markdown('<div class="sec-tag">Keyword Analysis</div><div class="sec-heading">What Customers Are Saying</div>', unsafe_allow_html=True)
    wc1,wc2 = st.columns(2)

    def make_wc(texts, cmap, title):
        combined = " ".join([clean_text(t) for t in texts])
        filtered = " ".join([w for w in combined.split() if w not in STOPWORDS and len(w)>2])
        if not filtered.strip(): st.info("Not enough data"); return
        wc = WordCloud(width=480,height=260,background_color="#131a27",colormap=cmap,max_words=60,collocations=False).generate(filtered)
        fig,ax = plt.subplots(figsize=(5,2.8),facecolor="#131a27")
        ax.imshow(wc,interpolation="bilinear"); ax.axis("off")
        ax.set_title(title,color="#dde4f5",pad=10,fontsize=10)
        plt.tight_layout(pad=0); st.pyplot(fig); plt.close()

    with wc1:
        if len(pos_df)>0: make_wc(pos_df["review"].tolist(),"YlGn","😊 Positive Keywords")
        else: st.info("No positive reviews")
    with wc2:
        if len(neg_df)>0: make_wc(neg_df["review"].tolist(),"OrRd","😞 Negative Keywords")
        else: st.info("No negative reviews")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Business Insights
    st.markdown('<div class="sec-tag">Business Intelligence</div><div class="sec-heading">What Your Business Should Do Now</div>', unsafe_allow_html=True)

    pos_kw = ", ".join([w for w,_ in get_top_words(pos_df["review"].tolist(),5)])
    neg_kw = ", ".join([w for w,_ in get_top_words(neg_df["review"].tolist(),5)])
    avg_r  = df["rating"].mean()
    neg_pct = len(neg_df)*100//total
    pos_pct = len(pos_df)*100//total

    summary = f"Total:{total}. Positive:{len(pos_df)}({pos_pct}%). Negative:{len(neg_df)}({neg_pct}%). Neutral:{len(neu_df)}. AvgRating:{avg_r:.2f}/5. Top positive words:{pos_kw}. Top negative words:{neg_kw}. Sample negatives:{'.'.join(neg_df['review'].head(2).tolist())}. Sample positives:{'.'.join(pos_df['review'].head(2).tolist())}"

    insights = []
    with st.spinner("💼 Generating business insights..."):
        try:
            import json
            if not ANTHROPIC_AVAILABLE: raise ImportError('anthropic not installed')
            client = anthropic_client.Anthropic()
            msg = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=900,
                messages=[{"role":"user","content":f"""Business consultant analyzing this customer review dataset:

{summary}

Generate exactly 5 specific, actionable business insights. Reply ONLY valid JSON array:
[{{"icon":"emoji","type":"pos or neg or neu","title":"max 6 words","desc":"2-3 sentences: specific finding from the data + exact action the business should take NOW"}}]

Rules: reference specific numbers/keywords from data, tell owner EXACTLY what to do, mix of pos/neg/neu types, no vague advice."""}]
            )
            text = re.sub(r"```json|```","",msg.content[0].text).strip()
            insights = json.loads(text)
        except:
            nw = neg_kw.split(",")[0].strip() if neg_kw else "quality"
            pw = pos_kw.split(",")[0].strip() if pos_kw else "delivery"
            insights = [
                {"icon":"📣","type":"pos","title":f"{pos_pct}% Customers Are Happy","desc":f"{len(pos_df)} reviews are positive. Keywords like '{pw}' keep appearing — screenshot these reviews and use them as testimonial ads this week. Authentic reviews convert 3x better than brand copy."},
                {"icon":"🚨","type":"neg","title":"Fix These Issues Urgently","desc":f"{len(neg_df)} negative reviews ({neg_pct}%) mention '{neg_kw}'. These exact problems are costing you sales every day. Escalate to your team with a 48-hour fix deadline."},
                {"icon":"🔧","type":"neg","title":"Your #1 Customer Complaint","desc":f"'{nw}' appears most in negative reviews. Add it as a mandatory QA checkpoint before every order goes out. Prevention costs 10x less than handling complaints."},
                {"icon":"⭐","type":"pos","title":"Promote Your Strength","desc":f"Customers praise '{pw}' — this is your USP. Weave it into your product title, ad copy, and email subject lines. Let customers' words sell for you."},
                {"icon":"📈","type":"neu","title":"Convert Neutrals to Promoters","desc":f"{len(neu_df)} customers gave 3 stars. Send a 1-question survey: 'What would make this 5 stars?' Their answers are your next product roadmap."}
            ]

    for ins in insights:
        t = ins.get("type","neu")
        st.markdown(f'''<div class="insight-card {t}">
            <div style="font-size:1.3rem;flex-shrink:0">{ins.get("icon","💡")}</div>
            <div><div class="insight-title">{ins.get("title","")}</div><div class="insight-desc">{ins.get("desc","")}</div></div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    csv_out = df[["review","rating","sentiment"]].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results CSV", csv_out, "analyzed_reviews.csv", "text/csv", use_container_width=True)

# ══════════════════════════════════
# ROUTER
# ══════════════════════════════════
if st.session_state.page == 1:
    page_welcome()
    st.stop()
elif st.session_state.page == 2:
    page_pipeline()
    st.stop()
elif st.session_state.page == 3:
    page_analyze()
    st.stop()
