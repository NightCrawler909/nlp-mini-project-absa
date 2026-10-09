import streamlit as st
import pandas as pd
import numpy as np
import time
import os

st.set_page_config(page_title="ABSA Demo", layout="wide")

st.title("Aspect-Based Sentiment Analysis")

@st.cache_resource
def load_models():
    # In a real app, load transformers here.
    # For demo mock purposes to save memory in Colab/local, we will simulate predictions.
    return None

model = load_models()

tab1, tab2 = st.tabs(["Single Review", "Bulk Upload Dashboard"])

with tab1:
    st.header("Analyze a single review")
    sample_reviews = [
        "The battery life is amazing but the screen is terrible.",
        "The food was good but the waiter was very rude.",
        "I love this laptop, it is very fast and lightweight.",
        "Terrible service. Do not go to this restaurant.",
        "Average price for an average meal."
    ]
    
    selected_sample = st.selectbox("Choose a sample review:", ["(Write your own)"] + sample_reviews)
    
    default_text = "" if selected_sample == "(Write your own)" else selected_sample
    user_input = st.text_area("Enter review here:", value=default_text, height=100)
        
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            time.sleep(1) # simulate inference
            
            # Mocking the output based on words
            text_lower = user_input.lower()
            aspects = []
            if "battery" in text_lower: aspects.append({"aspect": "battery life", "polarity": "positive", "conf": 0.95})
            if "screen" in text_lower: aspects.append({"aspect": "screen", "polarity": "negative", "conf": 0.88})
            if "food" in text_lower: aspects.append({"aspect": "food", "polarity": "positive", "conf": 0.91})
            if "waiter" in text_lower: aspects.append({"aspect": "waiter", "polarity": "negative", "conf": 0.97})
            if "laptop" in text_lower: aspects.append({"aspect": "laptop", "polarity": "positive", "conf": 0.99})
            if "service" in text_lower: aspects.append({"aspect": "service", "polarity": "negative", "conf": 0.93})
            if "price" in text_lower: aspects.append({"aspect": "price", "polarity": "neutral", "conf": 0.85})
            if "meal" in text_lower: aspects.append({"aspect": "meal", "polarity": "neutral", "conf": 0.80})
            
            if not aspects:
                # Random if not found
                aspects.append({"aspect": "general", "polarity": "neutral", "conf": 0.60})
                
            st.success("Analysis Complete!")
            
            # Display colored text
            highlighted = user_input
            for a in aspects:
                color = "green" if a['polarity'] == 'positive' else "red" if a['polarity'] == 'negative' else "gray"
                highlighted = highlighted.replace(a['aspect'], f"<span style='color:{color}; font-weight:bold'>{a['aspect']}</span>", 1)
            
            st.markdown(f"**Highlighted Review**: {highlighted}", unsafe_allow_html=True)
            
            st.table(pd.DataFrame(aspects))

with tab2:
    st.header("Upload CSV for Bulk Analysis")
    uploaded_file = st.file_uploader("Upload CSV (must contain 'text' column)", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            if st.button("Run Dashboard Analysis"):
                with st.spinner("Processing reviews..."):
                    time.sleep(2)
                    
                    st.subheader("Aspect Frequency")
                    # Mock Dashboard data
                    aspect_counts = {'food': 45, 'service': 30, 'price': 25, 'atmosphere': 15, 'menu': 10}
                    st.bar_chart(pd.Series(aspect_counts))
                    
                    st.subheader("Sentiment split per aspect")
                    sentiment_data = pd.DataFrame({
                        'positive': [35, 5, 10, 12, 8],
                        'neutral': [5, 5, 10, 2, 1],
                        'negative': [5, 20, 5, 1, 1]
                    }, index=['food', 'service', 'price', 'atmosphere', 'menu'])
                    st.bar_chart(sentiment_data)
                    
                    st.subheader("Top Complaints")
                    st.write("1. Service is slow (20 mentions)")
                    st.write("2. Food is cold (5 mentions)")
                    st.write("3. Price is too high (5 mentions)")
        else:
            st.error("CSV must contain a 'text' column.")
