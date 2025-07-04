import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import csv
from io import StringIO, BytesIO
import time
from datetime import datetime
import re
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Enhanced Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
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
    .sentiment-positive { color: #28a745; font-weight: bold; font-size: 1.2em; }
    .sentiment-negative { color: #dc3545; font-weight: bold; font-size: 1.2em; }
    .sentiment-neutral { color: #6c757d; font-weight: bold; font-size: 1.2em; }
</style>
""",
    unsafe_allow_html=True,
)

class SentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.models = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "finiteautomata/bertweet-base-sentiment-analysis",
            "siebert/sentiment-roberta-large-english",
        ]
        self.current_model = None
        self.api_url = None
        self.neutral_threshold = 0.3  # Difference between pos/neg to consider neutral
        self.neutral_min_score = 0.4  # Minimum neutral score to consider

    def normalize_sentiment(self, label):
        """Normalize sentiment labels to standard format"""
        label = str(label).upper().strip()
        if any(pos in label for pos in ["POSITIVE", "POS", "LABEL_2", "2"]):
            return "positive"
        elif any(neg in label for neg in ["NEGATIVE", "NEG", "LABEL_0", "0"]):
            return "negative"
        elif any(neu in label for neu in ["NEUTRAL", "NEU", "LABEL_1", "1"]):
            return "neutral"
        else:
            return "neutral"  # Default to neutral if unclear

    def contains_neutral_phrases(self, text):
        """Check for common neutral phrases"""
        neutral_phrases = [
            "it's okay", "not bad", "not great", "so so", 
            "nothing special", "average", "it's fine", "no strong opinion"
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in neutral_phrases)

    def find_working_model(self):
        """Find the first working model"""
        if self.current_model:
            return True

        for model in self.models:
            try:
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                response = requests.post(
                    api_url, 
                    headers=self.headers, 
                    json={"inputs": "This is a test sentence."}, 
                    timeout=10
                )

                if response.status_code == 200:
                    self.current_model = model
                    self.api_url = api_url
                    return True
            except:
                continue
        return False

    def analyze_sentiment(self, text):
        """Analyze sentiment of text with improved neutral detection"""
        try:
            if not self.find_working_model() or not self.api_url:
                st.warning("Falling back to rule-based analysis (API not available).")
                return self.fallback_analysis(text)

            # Check for neutral phrases first
            if self.contains_neutral_phrases(text):
                return {
                    "text": text[:500] + "..." if len(text) > 500 else text,
                    "sentiment": "neutral",
                    "confidence": 0.8,
                    "scores": {"positive": 0.1, "negative": 0.1, "neutral": 0.8},
                    "model": "neutral-phrase-detection"
                }

            text = re.sub(r"\s+", " ", text.strip())
            max_chunk_length = 400
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]

            aggregate_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            count = 0

            for chunk in chunks:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": chunk},
                    timeout=20
                )

                if response.status_code == 200:
                    result = response.json()
                    processed = self.process_response(result, chunk)
                    if processed and "scores" in processed:
                        for k in aggregate_scores:
                            aggregate_scores[k] += processed["scores"].get(k, 0.0)
                        count += 1
                    else:
                        fallback = self.fallback_analysis(chunk)
                        for k in aggregate_scores:
                            aggregate_scores[k] += fallback["scores"].get(k, 0.0)
                        count += 1
                elif response.status_code == 503:
                    time.sleep(5)
                    return self.analyze_sentiment(text)
                else:
                    return self.fallback_analysis(text)

            if count > 0:
                avg_scores = {k: v / count for k, v in aggregate_scores.items()}
                
                # Enhanced neutral detection
                pos_neg_diff = abs(avg_scores["positive"] - avg_scores["negative"])
                if (pos_neg_diff < self.neutral_threshold and 
                    max(avg_scores["positive"], avg_scores["negative"]) < 0.7 and
                    avg_scores["neutral"] > self.neutral_min_score):
                    primary_sentiment = "neutral"
                    confidence = avg_scores["neutral"]
                else:
                    primary_sentiment = max(avg_scores, key=avg_scores.get)
                    confidence = avg_scores[primary_sentiment]

                return {
                    "text": text[:500] + "..." if len(text) > 500 else text,
                    "sentiment": primary_sentiment,
                    "confidence": confidence,
                    "scores": avg_scores,
                    "model": self.current_model,
                }
            else:
                return self.fallback_analysis(text)

        except Exception as e:
            st.error(f"Exception during sentiment analysis: {str(e)}")
            return self.fallback_analysis(text)

    def process_response(self, result, original_text):
        """Process API response with improved neutral handling"""
        try:
            if isinstance(result, list) and len(result) > 0:
                sentiments = result[0] if isinstance(result[0], list) else result

                scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

                for item in sentiments:
                    if "label" in item and "score" in item:
                        normalized = self.normalize_sentiment(item["label"])
                        scores[normalized] += float(item["score"])

                # Normalize scores
                total = sum(scores.values())
                if total > 0:
                    scores = {k: v / total for k, v in scores.items()}

                # Enhanced neutral detection
                pos_neg_diff = abs(scores["positive"] - scores["negative"])
                if (pos_neg_diff < self.neutral_threshold and 
                    max(scores["positive"], scores["negative"]) < 0.7 and
                    scores["neutral"] > self.neutral_min_score):
                    primary_sentiment = "neutral"
                else:
                    primary_sentiment = max(scores, key=scores.get)

                return {
                    "text": original_text,
                    "sentiment": primary_sentiment,
                    "confidence": scores[primary_sentiment],
                    "scores": scores,
                    "model": self.current_model,
                }
            return None
        except Exception as e:
            return self.fallback_analysis(original_text)

    def fallback_analysis(self, text):
        """Improved rule-based fallback with better neutral detection"""
        positive_words = ["good", "great", "excellent", "amazing", "love", "perfect", "awesome"]
        negative_words = ["bad", "terrible", "awful", "hate", "worst", "horrible", "disgusting"]
        neutral_indicators = ["okay", "average", "fine", "decent", "adequate", "sufficient", "neutral"]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        neu_count = sum(1 for word in neutral_indicators if word in text_lower) * 2  # Boost neutral indicators

        # Calculate sentiment weights
        sentiment_weights = {
            "positive": pos_count,
            "negative": neg_count,
            "neutral": neu_count
        }

        # Determine primary sentiment
        primary_sentiment = max(sentiment_weights, key=sentiment_weights.get)
        
        # Calculate confidence
        total = sum(sentiment_weights.values())
        if total == 0:
            primary_sentiment = "neutral"
            confidence = 0.5
        else:
            confidence = sentiment_weights[primary_sentiment] / total

        # Ensure confidence isn't too extreme for neutral
        if primary_sentiment == "neutral":
            confidence = min(max(confidence, 0.4), 0.8)

        scores = {
            "positive": pos_count / max(total, 1),
            "negative": neg_count / max(total, 1),
            "neutral": neu_count / max(total, 1)
        }
        
        # Normalize scores to sum to 1
        total_scores = sum(scores.values())
        if total_scores > 0:
            scores = {k: v/total_scores for k, v in scores.items()}

        return {
            "text": text,
            "sentiment": primary_sentiment,
            "confidence": confidence,
            "scores": scores,
            "model": "fallback",
        }

    def batch_analyze(self, texts):
        """Analyze multiple texts"""
        results = []
        progress = st.progress(0)

        for i, text in enumerate(texts):
            progress.progress((i + 1) / len(texts))
            result = self.analyze_sentiment(text)
            if result:
                results.append(result)
            time.sleep(0.1)  # Rate limiting

        return results

# [Rest of your code remains the same - the visualization, export, and main functions]
# [Include all the remaining functions exactly as you had them]
# [Make sure to keep all the UI and display code unchanged]

if __name__ == "__main__":
    main()
