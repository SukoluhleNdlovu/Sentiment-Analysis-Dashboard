import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime
import re
 
# Streamlit page setup
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E86AB;
    }
    .sentiment-positive { color: #28a745; font-weight: bold; }
    .sentiment-negative { color: #dc3545; font-weight: bold; }
    .sentiment-neutral { color: #6c757d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)
 
# Sentiment Analyzer Class
class SentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.models = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "siebert/sentiment-roberta-large-english",
            "distilbert-base-uncased-finetuned-sst-2-english"
        ]
        self.current_model = None
        self.api_url = None
 
    def test_model(self, model_name, test_text="I love this product!"):
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        payload = {"inputs": test_text}
        try:
            response = requests.post(api_url, headers=self.headers, json=payload)
            return response.status_code == 200, response.json()
        except:
            return False, None
 
    def find_working_model(self):
        if self.current_model:
            return True
        for model in self.models:
            success, _ = self.test_model(model)
            if success:
                self.current_model = model
                self.api_url = f"https://api-inference.huggingface.co/models/{model}"
                return True
        return False
 
    def analyze_sentiment(self, text):
        if not self.find_working_model():
            return None
        try:
            payload = {"inputs": text}
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code != 200:
                return None
            result = response.json()
            sentiments = result[0] if isinstance(result[0], list) else result
            processed_result = {}
            for item in sentiments:
                label = item['label'].upper()
                score = item['score']
                if 'NEGATIVE' in label or 'LABEL_0' in label:
                    processed_result['negative'] = score
                elif 'POSITIVE' in label or 'LABEL_2' in label:
                    processed_result['positive'] = score
                elif 'NEUTRAL' in label or 'LABEL_1' in label:
                    processed_result['neutral'] = score
            processed_result.setdefault('positive', 0.0)
            processed_result.setdefault('negative', 0.0)
            processed_result.setdefault('neutral', 0.0)
            if sum(processed_result.values()) == 0:
                processed_result['neutral'] = 1.0
            primary_sentiment = max(processed_result, key=processed_result.get)
            confidence = processed_result[primary_sentiment]
            return {
                'sentiment': primary_sentiment,
                'confidence': confidence,
                'scores': processed_result,
                'text': text
            }
        except:
            return None
 
# Simple Visualizations
def create_sentiment_chart(results):
    counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for r in results:
        counts[r['sentiment']] += 1
    return px.pie(names=list(counts.keys()), values=list(counts.values()),
                  title="Sentiment Distribution",
                  color_discrete_map={'positive':'#28a745','negative':'#dc3545','neutral':'#6c757d'})
 
def create_confidence_chart(results):
    df = pd.DataFrame({"confidence": [r['confidence'] for r in results], "sentiment": [r['sentiment'] for r in results]})
    return px.histogram(df, x="confidence", color="sentiment",
                        title="Confidence Score Distribution",
                        color_discrete_map={'positive':'#28a745','negative':'#dc3545','neutral':'#6c757d'})
 
# Streamlit Interface
st.markdown('<h1 class="main-header">📊 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
api_key = st.sidebar.text_input("Enter your Hugging Face API Key", type="password")
if not api_key:
    st.warning("Please provide your Hugging Face API key in the sidebar.")
    st.stop()
 
analyzer = SentimentAnalyzer(api_key)
text_input = st.text_area("Enter text to analyze:", height=150)
if st.button("Analyze"):
    result = analyzer.analyze_sentiment(text_input)
    if result:
        st.success(f"Sentiment: {result['sentiment'].title()} ({result['confidence']:.2f} confidence)")
        st.json(result['scores'])
        st.session_state.results = st.session_state.get("results", []) + [result]
    else:
        st.error("Failed to get sentiment result.")
 
# Analytics Section
if "results" in st.session_state and st.session_state.results:
    st.subheader("📈 Sentiment Analytics")
    col1, col2 = st.columns(2)
    col1.plotly_chart(create_sentiment_chart(st.session_state.results), use_container_width=True)
    col2.plotly_chart(create_confidence_chart(st.session_state.results), use_container_width=True)
    st.dataframe(pd.DataFrame(st.session_state.results))