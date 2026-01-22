import streamlit as st
import pandas as pd

st.set_page_config(page_title="Business Review Analyzer", layout="wide")

st.title("📈 Exploiting business in review")
st.markdown("## Capstone Project - Text Analytics & Business Intelligence")

# Sidebar with uploader
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    st.markdown("---")
    if st.button("📊 Load Sample Data"):
        st.session_state.use_sample = True
        st.rerun()

# Main app logic
if 'use_sample' in st.session_state and st.session_state.use_sample:
    # Create sample data
    df = pd.DataFrame({
        'Review': ['Excellent product!', 'Not good quality', 'Average experience'],
        'Rating': [5, 2, 3],
        'Brand': ['Brand A', 'Brand B', 'Brand A'],
        'Date': ['2024-01-15', '2024-01-16', '2024-01-17']
    })
    
    st.success("✅ Sample Data Loaded!")
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Reviews", len(df))
    with col2: st.metric("Avg Rating", f"{df['Rating'].mean():.2f}")
    with col3: st.metric("Brands", df['Brand'].nunique())
    
    # Show data
    st.subheader("📋 Review Data")
    st.dataframe(df)
    
    # Chart
    st.subheader("📈 Rating Distribution")
    st.bar_chart(df['Rating'].value_counts())
    
elif uploaded_file is not None:
    # Load uploaded file
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Uploaded {len(df)} reviews!")
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Reviews", len(df))
    with col2: st.metric("Avg Rating", f"{df['Rating'].mean():.2f}")
    with col3: st.metric("Brands", df['Brand'].nunique())
    
    # Show data
    st.subheader("📋 Review Data")
    st.dataframe(df.head(10))
    
else:
    # Welcome screen
    st.markdown("### Welcome to Business Review Analyzer 🎉")
    st.markdown("#### How to Use This Application:")
    
    st.markdown("""
    1. **Prepare your data** in CSV format with these columns:
       - Review: Customer review text
       - Rating: Numeric rating (1-5)
       - Brand: Brand name
       - Date: Review date (YYYY-MM-DD)
    
    2. **Upload your file** using the sidebar
    
    3. **Get instant insights** including:
       - Sentiment analysis
       - Brand performance comparison
       - Trend analysis
       - Actionable business recommendations
    """)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to upload a CSV file or load sample data")

# Footer
st.markdown("---")
st.caption("🎓 Capstone Project: Text Analytics | Exploiting Business Intelligence from Reviews")
