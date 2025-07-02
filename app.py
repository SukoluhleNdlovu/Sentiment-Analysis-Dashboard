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

# Set page configuration
st.set_page_config(
    page_title="Enhanced Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2em;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2em;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
        font-size: 1.2em;
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 0.25rem;
        height: 20px;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        border-radius: 0.25rem;
        height: 100%;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedSentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        # Improved model selection with better sentiment models
        self.models = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",  # Best for general sentiment
            "siebert/sentiment-roberta-large-english",           # High accuracy
            "distilbert-base-uncased-finetuned-sst-2-english",  # Fast and reliable
            "nlptown/bert-base-multilingual-uncased-sentiment",  # Multilingual support
            "j-hartmann/emotion-english-distilroberta-base"      # Emotion-based sentiment
        ]
        self.current_model = None
        self.api_url = None
        self.model_info = {}
    
    def normalize_sentiment_labels(self, label):
        """Improved sentiment label normalization with better pattern matching"""
        if not label or not isinstance(label, str):
            return 'neutral'
        
        label = label.upper().strip()
        
        # More comprehensive label mapping
        label_mappings = {
            # Standard positive labels
            'POSITIVE': 'positive', 'POS': 'positive', 'LABEL_1': 'positive', 
            'LABEL_POSITIVE': 'positive', '1': 'positive', 'GOOD': 'positive',
            
            # Standard negative labels  
            'NEGATIVE': 'negative', 'NEG': 'negative', 'LABEL_0': 'negative',
            'LABEL_NEGATIVE': 'negative', '0': 'negative', 'BAD': 'negative',
            
            # Standard neutral labels
            'NEUTRAL': 'neutral', 'NEU': 'neutral', 'LABEL_2': 'neutral',
            'LABEL_NEUTRAL': 'neutral', '2': 'neutral', 'MIXED': 'neutral',
            
            # Emotion-based labels (map to sentiment)
            'JOY': 'positive', 'HAPPINESS': 'positive', 'LOVE': 'positive',
            'ANGER': 'negative', 'SADNESS': 'negative', 'FEAR': 'negative',
            'DISGUST': 'negative', 'SURPRISE': 'neutral', 'TRUST': 'positive'
        }
        
        # Direct mapping first
        if label in label_mappings:
            return label_mappings[label]
        
        # Pattern-based matching for complex labels
        if any(pos in label for pos in ['POS', 'GOOD', 'HAPPY', 'JOY', 'LOVE']):
            return 'positive'
        elif any(neg in label for neg in ['NEG', 'BAD', 'SAD', 'ANGER', 'HATE', 'FEAR']):
            return 'negative'
        elif any(neu in label for neu in ['NEU', 'NEUTRAL', 'MIXED', 'OBJECTIVE']):
            return 'neutral'
        
        # If we still can't classify, return neutral as fallback
        return 'neutral'
    
    def test_model(self, model_name, test_texts=None):
        """Enhanced model testing with multiple test cases"""
        if test_texts is None:
            test_texts = [
                "I love this product! It's amazing!",
                "This is terrible and I hate it.",
                "It's okay, nothing special.",
                "The weather is sunny today."
            ]
        
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        try:
            # Test with a simple positive sentiment
            payload = {"inputs": test_texts[0]}
            response = requests.post(api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Validate the response format
                if self.validate_response_format(result):
                    return True, result
                else:
                    return False, "Invalid response format"
            else:
                return False, f"Status {response.status_code}: {response.text}"
                
        except Exception as e:
            return False, str(e)
    
    def validate_response_format(self, response):
        """Validate that the API response has the expected sentiment analysis format"""
        try:
            if isinstance(response, list) and len(response) > 0:
                first_item = response[0]
                if isinstance(first_item, list):
                    # Format: [[{"label": "POSITIVE", "score": 0.9}]]
                    return all('label' in item and 'score' in item for item in first_item)
                elif isinstance(first_item, dict):
                    # Format: [{"label": "POSITIVE", "score": 0.9}]
                    return 'label' in first_item and 'score' in first_item
            return False
        except:
            return False
    
    def find_working_model(self):
        """Enhanced model selection with better testing"""
        if self.current_model:
            return True
            
        st.info("🔍 Testing models for optimal sentiment analysis...")
        
        model_scores = {}
        
        for model in self.models:
            with st.spinner(f"Testing {model}..."):
                success, result = self.test_model(model)
                
                if success:
                    # Score the model based on response quality
                    score = self.score_model_response(result)
                    model_scores[model] = score
                    st.success(f"✅ {model}: Score {score:.2f}")
                else:
                    st.warning(f"❌ {model}: {result}")
        
        # Select the best scoring model
        if model_scores:
            best_model = max(model_scores.keys(), key=lambda k: model_scores[k])
            self.current_model = best_model
            self.api_url = f"https://api-inference.huggingface.co/models/{best_model}"
            st.info(f"🎯 Selected best model: {best_model}")
            return True
        
        st.error("❌ No working models found. Please check your API key or try again later.")
        return False
    
    def score_model_response(self, response):
        """Score model response quality for model selection"""
        try:
            # Extract sentiment scores
            if isinstance(response, list) and len(response) > 0:
                if isinstance(response[0], list):
                    sentiments = response[0]
                else:
                    sentiments = response
                
                # Score based on confidence and label clarity
                max_score = max(item['score'] for item in sentiments)
                label_clarity = len(set(item['label'] for item in sentiments))
                
                return max_score * 0.7 + (label_clarity / 3) * 0.3
            
            return 0.0
        except:
            return 0.0
    
    def analyze_sentiment(self, text):
        """Enhanced sentiment analysis with better error handling and normalization"""
        # Ensure we have a working model
        if not self.find_working_model():
            return None
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        if not cleaned_text:
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'text': text,
                'model': 'preprocessing',
                'warning': 'Empty or invalid text'
            }
            
        try:
            payload = {"inputs": cleaned_text}
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                processed_result = self.process_api_response(result, text)
                
                if processed_result:
                    # Enhanced sentiment determination
                    processed_result = self.enhance_sentiment_detection(processed_result)
                    return processed_result
                else:
                    return self.fallback_sentiment_analysis(text)
                    
            elif response.status_code == 503:
                st.warning("⏳ Model is loading, please wait 10-20 seconds and try again")
                time.sleep(2)
                return self.analyze_sentiment(text)  # Retry once
            elif response.status_code == 429:
                st.warning("⏸️ Rate limit reached, please wait a moment and try again")
                time.sleep(5)
                return None
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return self.fallback_sentiment_analysis(text)
                
        except Exception as e:
            st.error(f"Error analyzing sentiment: {str(e)}")
            return self.fallback_sentiment_analysis(text)
    
    def preprocess_text(self, text):
        """Clean and preprocess text for better analysis"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Limit text length (most models have token limits)
        if len(text) > 500:
            text = text[:500] + "..."
        
        return text.strip()
    
    def process_api_response(self, result, original_text):
        """Fixed API response processing with better label handling"""
        try:
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    sentiments = result[0]
                else:
                    sentiments = result
                
                # Debug: Print raw response for troubleshooting
                if st.session_state.get('debug_mode', False):
                    st.write("Debug - Raw API Response:", sentiments)
                
                processed_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
                
                # Process each sentiment score
                for item in sentiments:
                    if 'label' in item and 'score' in item:
                        raw_label = item['label']
                        normalized_label = self.normalize_sentiment_labels(raw_label)
                        score = float(item['score'])
                        
                        # Debug information
                        if st.session_state.get('debug_mode', False):
                            st.write(f"Raw label: {raw_label} -> Normalized: {normalized_label}, Score: {score}")
                        
                        # Use the highest score for each sentiment category
                        if normalized_label in processed_scores:
                            processed_scores[normalized_label] = max(processed_scores[normalized_label], score)
                
                # If no scores were assigned, try a different approach
                if all(score == 0.0 for score in processed_scores.values()):
                    # Map scores directly without normalization first
                    for item in sentiments:
                        if 'label' in item and 'score' in item:
                            raw_label = item['label'].upper()
                            score = float(item['score'])
                            
                            # Direct mapping based on common patterns
                            if 'POSITIVE' in raw_label or 'POS' in raw_label or raw_label == 'LABEL_1':
                                processed_scores['positive'] = score
                            elif 'NEGATIVE' in raw_label or 'NEG' in raw_label or raw_label == 'LABEL_0':
                                processed_scores['negative'] = score
                            elif 'NEUTRAL' in raw_label or raw_label == 'LABEL_2':
                                processed_scores['neutral'] = score
                
                # Normalize scores to sum to 1.0
                total_score = sum(processed_scores.values())
                if total_score > 0:
                    processed_scores = {k: v/total_score for k, v in processed_scores.items()}
                else:
                    # Fallback: equal distribution
                    processed_scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                
                # Determine primary sentiment
                primary_sentiment = max(processed_scores.keys(), key=lambda k: processed_scores[k])
                confidence = processed_scores[primary_sentiment]
                
                return {
                    'sentiment': primary_sentiment,
                    'confidence': confidence,
                    'scores': processed_scores,
                    'text': original_text,
                    'model': self.current_model,
                    'raw_response': result
                }
            
            return None
            
        except Exception as e:
            st.error(f"Error processing API response: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.write("Debug - Exception details:", str(e))
                st.write("Debug - Raw result:", result)
            return None
    
    def enhance_sentiment_detection(self, result):
        """Apply additional logic to improve sentiment detection accuracy"""
        text = result['text'].lower()
        scores = result['scores'].copy()
        
        # Rule-based adjustments for common patterns
        positive_boosters = ['love', 'excellent', 'amazing', 'fantastic', 'perfect', 'wonderful', 'awesome', 'brilliant', 'outstanding', 'great']
        negative_boosters = ['hate', 'terrible', 'awful', 'horrible', 'worst', 'disgusting', 'pathetic', 'useless', 'disappointing']
        
        # Count sentiment indicators
        positive_count = sum(1 for word in positive_boosters if word in text)
        negative_count = sum(1 for word in negative_boosters if word in text)
        
        # Adjust scores based on strong indicators
        boost_factor = 0.15  # Increased boost factor
        if positive_count > negative_count and positive_count > 0:
            scores['positive'] = min(scores['positive'] + (boost_factor * positive_count), 0.95)
            scores['neutral'] = max(scores['neutral'] - (boost_factor * positive_count / 2), 0.05)
        elif negative_count > positive_count and negative_count > 0:
            scores['negative'] = min(scores['negative'] + (boost_factor * negative_count), 0.95)
            scores['neutral'] = max(scores['neutral'] - (boost_factor * negative_count / 2), 0.05)
        
        # Handle exclamation marks (usually indicate strong sentiment)
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            if scores['positive'] > scores['negative']:
                scores['positive'] = min(scores['positive'] + (0.05 * exclamation_count), 0.95)
            elif scores['negative'] > scores['positive']:
                scores['negative'] = min(scores['negative'] + (0.05 * exclamation_count), 0.95)
        
        # Handle negations more carefully
        negation_pattern = r'\b(not|no|never|neither|nobody|nothing|nowhere|hardly|scarcely|barely|don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|can\'t|couldn\'t)\b'
        negations = len(re.findall(negation_pattern, text))
        
        if negations > 0:
            # Only swap if there's a clear sentiment difference
            if abs(scores['positive'] - scores['negative']) > 0.2:
                if scores['positive'] > scores['negative']:
                    # Reduce positive, increase negative
                    scores['negative'] = min(scores['positive'], 0.9)
                    scores['positive'] = max(scores['negative'] / 2, 0.1)
                elif scores['negative'] > scores['positive']:
                    # Reduce negative, increase positive  
                    scores['positive'] = min(scores['negative'], 0.9)
                    scores['negative'] = max(scores['positive'] / 2, 0.1)
        
        # Renormalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        # Update primary sentiment
        primary_sentiment = max(scores.keys(), key=lambda k: scores[k])
        
        result['sentiment'] = primary_sentiment
        result['confidence'] = scores[primary_sentiment]
        result['scores'] = scores
        
        return result
    
    def fallback_sentiment_analysis(self, text):
        """Improved rule-based fallback sentiment analysis"""
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'perfect', 'best', 'awesome', 'brilliant', 'outstanding', 'superb', 'marvelous']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'poor', 'disgusting', 'pathetic', 'useless', 'annoying', 'frustrating']
        
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        # Consider exclamation marks
        exclamation_boost = text.count('!') * 0.1
        
        if positive_score > negative_score:
            sentiment = 'positive'
            confidence = min(0.7 + (positive_score - negative_score) * 0.1 + exclamation_boost, 0.95)
        elif negative_score > positive_score:
            sentiment = 'negative'
            confidence = min(0.7 + (negative_score - positive_score) * 0.1 + exclamation_boost, 0.95)
        else:
            sentiment = 'neutral'
            confidence = 0.6
        
        # Create normalized scores
        if sentiment == 'positive':
            scores = {'positive': confidence, 'negative': (1-confidence)*0.3, 'neutral': (1-confidence)*0.7}
        elif sentiment == 'negative':
            scores = {'negative': confidence, 'positive': (1-confidence)*0.3, 'neutral': (1-confidence)*0.7}
        else:
            scores = {'neutral': confidence, 'positive': (1-confidence)/2, 'negative': (1-confidence)/2}
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': scores,
            'text': text,
            'model': 'fallback_rule_based',
            'warning': 'Using fallback analysis'
        }
    
import re
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def batch_analyze(self, texts, progress_bar=None):
    """Enhanced batch analysis with better progress tracking and error handling"""
    if not texts:
        return []
    
    results = []
    total = len(texts)
    failed_count = 0
    
    # Ensure we have a working model before starting batch processing
    if not self.find_working_model():
        st.error("❌ No working model available for batch analysis")
        return []
    
    for i, text in enumerate(texts):
        try:
            if progress_bar:
                progress_bar.progress(
                    (i + 1) / total, 
                    f"Processing {i + 1}/{total} texts (Failed: {failed_count})"
                )
            
            # Skip empty or invalid texts
            if not text or not isinstance(text, str) or len(text.strip()) == 0:
                failed_count += 1
                fallback_result = self.fallback_sentiment_analysis("Empty text")
                fallback_result['text'] = text or "Empty"
                fallback_result['warning'] = 'Empty or invalid text'
                results.append(fallback_result)
                continue
            
            result = self.analyze_sentiment(text)
            if result:
                results.append(result)
            else:
                failed_count += 1
                # Add a fallback result
                fallback_result = self.fallback_sentiment_analysis(text)
                fallback_result['warning'] = 'Analysis failed, using fallback'
                results.append(fallback_result)
            
            # Adaptive delay based on API response and batch position
            if i % 10 == 0 and i > 0:
                time.sleep(1.0)  # Longer pause every 10 requests
            else:
                time.sleep(0.2)  # Short pause between requests
                
        except Exception as e:
            failed_count += 1
            st.warning(f"Error processing text {i+1}: {str(e)}")
            fallback_result = self.fallback_sentiment_analysis(text if text else "Error")
            fallback_result['warning'] = f'Processing error: {str(e)}'
            results.append(fallback_result)
    
    if progress_bar:
        progress_bar.progress(1.0, f"Completed! Processed {total} texts (Failed: {failed_count})")
    
    return results

def create_enhanced_sentiment_display(result):
    """Create an enhanced visual display for sentiment results with better error handling"""
    try:
        if not result or not isinstance(result, dict):
            return "<div>Invalid result data</div>"
        
        sentiment = result.get('sentiment', 'unknown')
        confidence = result.get('confidence', 0.0)
        scores = result.get('scores', {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34})
        
        # Ensure confidence is a valid number
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.0
        
        # Color mapping
        color_map = {
            'positive': '#28a745',
            'negative': '#dc3545',
            'neutral': '#6c757d',
            'unknown': '#6c757d'
        }
        
        main_color = color_map.get(sentiment, '#6c757d')
        
        # Create confidence visualization
        confidence_html = f"""
        <div style="margin: 10px 0; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span><strong>Confidence:</strong></span>
                <span style="color: {main_color}; font-weight: bold;">{confidence:.1%}</span>
            </div>
            <div style="background-color: #e9ecef; border-radius: 10px; height: 20px; overflow: hidden;">
                <div style="width: {confidence*100:.1f}%; height: 100%; background-color: {main_color}; border-radius: 10px; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """
        
        # Create score breakdown
        score_html = "<div style='margin: 10px 0;'><strong>Score Breakdown:</strong><br>"
        
        for sent, score in scores.items():
            try:
                score_value = float(score)
            except (ValueError, TypeError):
                score_value = 0.0
                
            color = color_map.get(sent, '#6c757d')
            is_primary = sent == sentiment
            
            score_html += f"""
            <div style='margin: 5px 0; padding: 8px; 
                       background-color: {color}{'30' if is_primary else '15'}; 
                       border-left: 4px solid {color}; 
                       border-radius: 3px;
                       {'font-weight: bold;' if is_primary else ''}'>
                {sent.title()}: {score_value:.1%}
            </div>
            """
        score_html += "</div>"
        
        # Add warning if present
        warning_html = ""
        if 'warning' in result:
            warning_html = f"""
            <div style='margin: 10px 0; padding: 8px; background-color: #fff3cd; 
                       border-left: 4px solid #ffc107; border-radius: 3px;'>
                <small><strong>Note:</strong> {result['warning']}</small>
            </div>
            """
        
        return confidence_html + score_html + warning_html
        
    except Exception as e:
        return f"<div style='color: red;'>Error displaying result: {str(e)}</div>"

def extract_enhanced_keywords(text, sentiment_result):
    """Enhanced keyword extraction with better sentiment correlation and error handling"""
    try:
        if not text or not isinstance(text, str):
            return []
        
        # Expanded keyword lists
        positive_keywords = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'perfect', 'best',
            'awesome', 'brilliant', 'outstanding', 'superb', 'magnificent', 'incredible', 'fabulous',
            'terrific', 'marvelous', 'phenomenal', 'exceptional', 'impressive', 'delightful', 'splendid',
            'remarkable', 'extraordinary', 'stunning', 'beautiful', 'gorgeous', 'lovely', 'charming'
        ]
        
        negative_keywords = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'poor',
            'disgusting', 'pathetic', 'useless', 'dreadful', 'appalling', 'atrocious', 'abysmal',
            'deplorable', 'horrendous', 'ghastly', 'hideous', 'repulsive', 'revolting', 'nasty',
            'ugly', 'annoying', 'frustrating', 'irritating', 'disturbing', 'unpleasant'
        ]
        
        neutral_keywords = [
            'okay', 'fine', 'average', 'normal', 'standard', 'typical', 'regular', 'common',
            'ordinary', 'moderate', 'adequate', 'acceptable', 'reasonable', 'fair', 'decent',
            'usual', 'conventional', 'traditional', 'basic', 'simple', 'plain'
        ]
        
        # Extract words from text (improved regex)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        found_keywords = []
        
        sentiment = sentiment_result.get('sentiment', 'neutral') if sentiment_result else 'neutral'
        
        # Find relevant keywords based on sentiment
        if sentiment == 'positive':
            found_keywords = [word for word in words if word in positive_keywords]
        elif sentiment == 'negative':
            found_keywords = [word for word in words if word in negative_keywords]
        else:
            found_keywords = [word for word in words if word in neutral_keywords]
        
        # Also include cross-sentiment keywords if they appear (but limit them)
        all_sentiment_words = positive_keywords + negative_keywords + neutral_keywords
        additional_keywords = [word for word in words if word in all_sentiment_words and word not in found_keywords]
        
        # Combine and limit results, remove duplicates
        all_keywords = list(dict.fromkeys(found_keywords + additional_keywords[:3]))  # Preserve order while removing duplicates
        return all_keywords[:5]  # Return max 5 keywords
        
    except Exception as e:
        st.warning(f"Error extracting keywords: {str(e)}")
        return []

def create_sentiment_chart(results):
    """Enhanced sentiment distribution chart with better error handling"""
    try:
        if not results or not isinstance(results, list):
            return None
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_confidence = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        # Count sentiments and accumulate confidence scores
        for result in results:
            if isinstance(result, dict) and 'sentiment' in result:
                sentiment = result.get('sentiment', 'neutral')
                if sentiment in sentiment_counts:
                    sentiment_counts[sentiment] += 1
                    confidence = result.get('confidence', 0)
                    try:
                        total_confidence[sentiment] += float(confidence)
                    except (ValueError, TypeError):
                        pass
        
        # Check if we have any data
        if sum(sentiment_counts.values()) == 0:
            return None
        
        # Calculate average confidence per sentiment
        avg_confidence = {}
        for sentiment in sentiment_counts:
            if sentiment_counts[sentiment] > 0:
                avg_confidence[sentiment] = total_confidence[sentiment] / sentiment_counts[sentiment]
            else:
                avg_confidence[sentiment] = 0
        
        # Create enhanced pie chart
        fig = px.pie(
            values=list(sentiment_counts.values()),
            names=list(sentiment_counts.keys()),
            title="Sentiment Distribution",
            color_discrete_map={
                'positive': '#28a745',
                'negative': '#dc3545',
                'neutral': '#6c757d'
            },
            hole=0.4  # Donut chart for better visual appeal
        )
        
        # Enhanced hover template with confidence info
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>' +
                         'Count: %{value}<br>' +
                         'Percentage: %{percent}<br>' +
                         'Avg Confidence: ' + 
                         '<extra></extra>'
        )
        
        # Update layout
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(size=12),
            title_font_size=16
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating sentiment chart: {str(e)}")
        return None

def create_confidence_chart(results):
    """Enhanced confidence score visualization with better error handling"""
    try:
        if not results or not isinstance(results, list):
            return None
        
        # Prepare data for visualization
        confidence_data = []
        
        for i, result in enumerate(results):
            if isinstance(result, dict):
                try:
                    confidence = float(result.get('confidence', 0))
                    sentiment = result.get('sentiment', 'unknown')
                    model = result.get('model', 'unknown')
                    
                    confidence_data.append({
                        'confidence': confidence,
                        'sentiment': sentiment,
                        'model': model,
                        'index': i + 1
                    })
                except (ValueError, TypeError):
                    continue
        
        if not confidence_data:
            return None
        
        df = pd.DataFrame(confidence_data)
        
        # Create box plot for confidence distribution by sentiment
        fig = px.box(
            df,
            x='sentiment',
            y='confidence',
            color='sentiment',
            title="Confidence Score Distribution by Sentiment",
            color_discrete_map={
                'positive': '#28a745',
                'negative': '#dc3545',
                'neutral': '#6c757d',
                'unknown': '#6c757d'
            },
            points="outliers"  # Show outlier points
        )
        
        # Add mean line
        for sentiment in df['sentiment'].unique():
            sentiment_data = df[df['sentiment'] == sentiment]
            mean_confidence = sentiment_data['confidence'].mean()
            
            fig.add_hline(
                y=mean_confidence,
                line_dash="dash",
                line_color="red",
                opacity=0.7,
                annotation_text=f"Mean: {mean_confidence:.2f}"
            )
        
        fig.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="Confidence Score",
            showlegend=False,
            yaxis=dict(range=[0, 1]),  # Set y-axis range from 0 to 1
            font=dict(size=12),
            title_font_size=16
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating confidence chart: {str(e)}")
        return None

def create_detailed_results_table(results):
    """Create a detailed table view of all results"""
    try:
        if not results or not isinstance(results, list):
            return None
        
        # Prepare data for table
        table_data = []
        
        for i, result in enumerate(results):
            if isinstance(result, dict):
                text = result.get('text', '')[:100]  # Truncate long text
                if len(result.get('text', '')) > 100:
                    text += "..."
                
                table_data.append({
                    'Index': i + 1,
                    'Text': text,
                    'Sentiment': result.get('sentiment', 'unknown').title(),
                    'Confidence': f"{result.get('confidence', 0):.1%}",
                    'Positive': f"{result.get('scores', {}).get('positive', 0):.1%}",
                    'Negative': f"{result.get('scores', {}).get('negative', 0):.1%}",
                    'Neutral': f"{result.get('scores', {}).get('neutral', 0):.1%}",
                    'Model': result.get('model', 'unknown'),
                    'Warning': result.get('warning', '')
                })
        
        if table_data:
            return pd.DataFrame(table_data)
        
        return None
        
    except Exception as e:
        st.error(f"Error creating results table: {str(e)}")
        return None

def export_results(results, format_type):
    """Enhanced export functionality with more details"""
    if format_type == "CSV":
        # Create a more comprehensive DataFrame
        export_data = []
        for result in results:
            export_data.append({
                'text': result['text'],
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'positive_score': result['scores']['positive'],
                'negative_score': result['scores']['negative'],
                'neutral_score': result['scores']['neutral'],
                'model': result['model'],
                'warning': result.get('warning', '')
            })
        df = pd.DataFrame(export_data)
        return df.to_csv(index=False)
    
    elif format_type == "JSON":
        return json.dumps(results, indent=2, default=str)
    
    elif format_type == "TXT":
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"Result {i}:")
            output.append(f"Text: {result['text']}")
            output.append(f"Sentiment: {result['sentiment'].title()}")
            output.append(f"Confidence: {result['confidence']:.3f}")
            output.append(f"Detailed Scores:")
            for sentiment, score in result['scores'].items():
                output.append(f"  {sentiment.title()}: {score:.3f}")
            output.append(f"Model: {result['model']}")
            if result.get('warning'):
                output.append(f"Warning: {result['warning']}")
            output.append("-" * 60)
        return "\n".join(output)

# Main Application
def main():
    st.markdown('<h1 class="main-header">📊 Enhanced Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for API configuration
    st.sidebar.header("🔧 Configuration")
    api_key = st.sidebar.text_input("Hugging Face API Key", type="password", help="Enter your Hugging Face API key")
    
    # Add model selection option
    st.sidebar.subheader("Model Preferences")
    auto_select = st.sidebar.checkbox("Auto-select best model", value=True)
    
    if not api_key:
        st.warning("Please enter your Hugging Face API key in the sidebar to begin.")
        st.info("You can get your API key from: https://huggingface.co/settings/tokens")
        return
    
    # Initialize the enhanced analyzer
    analyzer = EnhancedSentimentAnalyzer(api_key)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Text Analysis", "📁 Batch Analysis", "📊 Analytics", "📥 Export Results"])
    
    # Initialize session state for results
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    with tab1:
        st.header("Single Text Analysis")
        
        # Text input methods
        input_method = st.radio("Choose input method:", ["Direct Input", "File Upload"])
        
        if input_method == "Direct Input":
            user_text = st.text_area("Enter text to analyze:", height=150, placeholder="Type or paste your text here...")
            
            # Add example texts
            example_col1, example_col2, example_col3 = st.columns(3)
            with example_col1:
                if st.button("Try Positive Example"):
                    st.session_state.example_text = "I absolutely love this product! It's amazing and works perfectly. Best purchase I've made this year!"
            with example_col2:
                if st.button("Try Negative Example"):
                    st.session_state.example_text = "This is terrible quality. I hate it and want my money back. Worst experience ever!"
            with example_col3:
                if st.button("Try Neutral Example"):
                    st.session_state.example_text = "The product is okay. It works as expected, nothing special but does the job."
            
            # Use example text if set
            if 'example_text' in st.session_state:
                user_text = st.session_state.example_text
                del st.session_state.example_text
            
            if st.button("Analyze Sentiment", type="primary"):
                if user_text.strip():
                    with st.spinner("Analyzing sentiment..."):
                        result = analyzer.analyze_sentiment(user_text)
                        
                        if result:
                            # Enhanced display
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Main sentiment display
                                sentiment_class = f"sentiment-{result['sentiment']}"
                                st.markdown(f'<div class="metric-card"><h3>Primary Sentiment</h3><p class="{sentiment_class}">{result["sentiment"].title()}</p></div>', unsafe_allow_html=True)
                                
                                # Enhanced confidence and scores display
                                st.markdown(create_enhanced_sentiment_display(result), unsafe_allow_html=True)
                                
                                # Keywords
                                keywords = extract_enhanced_keywords(user_text, result)
                                if keywords:
                                    st.markdown(f'<div class="metric-card"><h3>Key Sentiment Words</h3><p>{", ".join(keywords)}</p></div>', unsafe_allow_html=True)
                                
                                # Model info
                                if result.get('warning'):
                                    st.warning(f"⚠️ {result['warning']}")
                                st.info(f"Model used: {result['model']}")
                            
                            with col2:
                                scores_df = pd.DataFrame([result['scores']]).T
                                scores_df.columns = ['Score']
                                scores_df.index.name = 'Sentiment'
                                
                                fig = px.bar(
                                    x=scores_df.index,
                                    y=scores_df['Score'],
                                    color=scores_df.index,
                                    color_discrete_map={
                                        'positive': '#28a745',
                                        'negative': '#dc3545',
                                        'neutral': '#6c757d'
                                    },
                                    title="Sentiment Scores"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.session_state.results.append(result)
    
    with tab2:
        st.header("Batch Processing")
        
        batch_method = st.radio("Choose batch input method:", ["Multiple Texts", "CSV Upload"])
        
        if batch_method == "Multiple Texts":
            batch_text = st.text_area("Enter multiple texts (one per line):", height=200, placeholder="Enter each text on a new line...")
            
            # Add batch examples
            if st.button("Load Example Batch"):
                example_batch = """I love this new smartphone! The camera quality is amazing.
This restaurant has terrible service and cold food.
The weather today is okay, nothing special.
Amazing customer support! They solved my problem quickly.
The movie was boring and too long. Waste of time.
Standard delivery service, arrived on time as expected."""
                st.session_state.batch_example = example_batch
            
            # Use example batch if set
            if 'batch_example' in st.session_state:
                batch_text = st.session_state.batch_example
                del st.session_state.batch_example
            
            if st.button("Analyze Batch", type="primary"):
                if batch_text.strip():
                    texts = [text.strip() for text in batch_text.split('\n') if text.strip()]
                    
                    if texts:
                        progress_bar = st.progress(0, "Starting batch analysis...")
                        
                        batch_results = analyzer.batch_analyze(texts, progress_bar)
                        
                        if batch_results:
                            st.success(f"Successfully analyzed {len(batch_results)} texts!")
                            
                            # Enhanced batch results display
                            st.subheader("Batch Analysis Results")
                            
                            # Summary statistics
                            positive_count = sum(1 for r in batch_results if r['sentiment'] == 'positive')
                            negative_count = sum(1 for r in batch_results if r['sentiment'] == 'negative')
                            neutral_count = sum(1 for r in batch_results if r['sentiment'] == 'neutral')
                            avg_confidence = sum(r['confidence'] for r in batch_results) / len(batch_results)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Positive", positive_count, f"{positive_count/len(batch_results)*100:.1f}%")
                            with col2:
                                st.metric("Negative", negative_count, f"{negative_count/len(batch_results)*100:.1f}%")
                            with col3:
                                st.metric("Neutral", neutral_count, f"{neutral_count/len(batch_results)*100:.1f}%")
                            with col4:
                                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                            
                            # Individual results
                            for i, result in enumerate(batch_results):
                                confidence_emoji = "🔥" if result['confidence'] > 0.8 else "👍" if result['confidence'] > 0.6 else "🤔"
                                sentiment_emoji = "😊" if result['sentiment'] == 'positive' else "😞" if result['sentiment'] == 'negative' else "😐"
                                
                                with st.expander(f"{sentiment_emoji} Text {i+1}: {result['sentiment'].title()} {confidence_emoji} ({result['confidence']:.1%})"):
                                    st.write(f"**Text:** {result['text']}")
                                    st.write(f"**Sentiment:** {result['sentiment'].title()}")
                                    st.write(f"**Confidence:** {result['confidence']:.1%}")
                                    
                                    # Mini score chart
                                    scores = result['scores']
                                    score_text = " | ".join([f"{k.title()}: {v:.1%}" for k, v in scores.items()])
                                    st.write(f"**Scores:** {score_text}")
                                    
                                    if result.get('warning'):
                                        st.warning(result['warning'])
                            
                            # Add to session state
                            st.session_state.results.extend(batch_results)
                        else:
                            st.error("Failed to analyze texts. Please try again.")
                else:
                    st.warning("Please enter some texts to analyze.")
        
        else:  # CSV Upload
            uploaded_csv = st.file_uploader("Upload CSV file with 'text' column", type=['csv'])
            
            if uploaded_csv is not None:
                try:
                    df = pd.read_csv(uploaded_csv)
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())
                    
                    # Show column information
                    st.info(f"Columns found: {', '.join(df.columns.tolist())}")
                    
                    # Allow column selection if 'text' is not present
                    text_column = 'text'
                    if 'text' not in df.columns:
                        text_column = st.selectbox("Select the text column:", df.columns.tolist())
                    
                    if text_column in df.columns:
                        # Show sample data
                        st.write(f"Sample data from '{text_column}' column:")
                        st.write(df[text_column].head().tolist())
                        
                        if st.button("Analyze CSV Data", type="primary"):
                            texts = df[text_column].dropna().tolist()
                            
                            if texts:
                                progress_bar = st.progress(0, "Processing CSV data...")
                                
                                batch_results = analyzer.batch_analyze(texts, progress_bar)
                                
                                if batch_results:
                                    st.success(f"Successfully analyzed {len(batch_results)} texts from CSV!")
                                    
                                    # Create enhanced results DataFrame
                                    results_df = pd.DataFrame([
                                        {
                                            'original_text': r['text'],
                                            'sentiment': r['sentiment'],
                                            'confidence': r['confidence'],
                                            'positive_score': r['scores']['positive'],
                                            'negative_score': r['scores']['negative'],
                                            'neutral_score': r['scores']['neutral'],
                                            'model': r['model']
                                        }
                                        for r in batch_results
                                    ])
                                    
                                    st.subheader("CSV Analysis Results")
                                    st.dataframe(results_df)
                                    
                                    # Quick stats
                                    sentiment_counts = results_df['sentiment'].value_counts()
                                    st.write("**Sentiment Distribution:**")
                                    for sentiment, count in sentiment_counts.items():
                                        percentage = count / len(results_df) * 100
                                        st.write(f"- {sentiment.title()}: {count} ({percentage:.1f}%)")
                                    
                                    st.session_state.results.extend(batch_results)
                                else:
                                    st.error("Failed to analyze CSV data.")
                            else:
                                st.error(f"No valid text data found in column '{text_column}'.")
                    else:
                        st.error(f"Column '{text_column}' not found in the CSV file.")
                        
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        if st.session_state.results:
            # Enhanced summary metrics
            total_texts = len(st.session_state.results)
            positive_count = sum(1 for r in st.session_state.results if r['sentiment'] == 'positive')
            negative_count = sum(1 for r in st.session_state.results if r['sentiment'] == 'negative')
            neutral_count = sum(1 for r in st.session_state.results if r['sentiment'] == 'neutral')
            avg_confidence = sum(r['confidence'] for r in st.session_state.results) / total_texts
            
            # Calculate confidence ranges
            high_confidence = sum(1 for r in st.session_state.results if r['confidence'] > 0.8)
            medium_confidence = sum(1 for r in st.session_state.results if 0.6 <= r['confidence'] <= 0.8)
            low_confidence = sum(1 for r in st.session_state.results if r['confidence'] < 0.6)
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Texts", total_texts)
            with col2:
                st.metric("Positive 😊", positive_count, f"{positive_count/total_texts*100:.1f}%")
            with col3:
                st.metric("Negative 😞", negative_count, f"{negative_count/total_texts*100:.1f}%")
            with col4:
                st.metric("Neutral 😐", neutral_count, f"{neutral_count/total_texts*100:.1f}%")
            with col5:
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            # Confidence breakdown
            st.subheader("Confidence Analysis")
            conf_col1, conf_col2, conf_col3 = st.columns(3)
            
            with conf_col1:
                st.metric("High Confidence", high_confidence, f"{high_confidence/total_texts*100:.1f}%")
                st.caption("(>80%)")
            with conf_col2:
                st.metric("Medium Confidence", medium_confidence, f"{medium_confidence/total_texts*100:.1f}%")
                st.caption("(60-80%)")
            with conf_col3:
                st.metric("Low Confidence", low_confidence, f"{low_confidence/total_texts*100:.1f}%")
                st.caption("(<60%)")
            
            # Enhanced Charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                sentiment_chart = create_sentiment_chart(st.session_state.results)
                if sentiment_chart:
                    st.plotly_chart(sentiment_chart, use_container_width=True)
            
            with chart_col2:
                confidence_chart = create_confidence_chart(st.session_state.results)
                if confidence_chart:
                    st.plotly_chart(confidence_chart, use_container_width=True)
            
            # Model performance analysis
            st.subheader("Model Performance")
            model_stats = {}
            for result in st.session_state.results:
                model = result['model']
                if model not in model_stats:
                    model_stats[model] = {'count': 0, 'total_confidence': 0}
                model_stats[model]['count'] += 1
                model_stats[model]['total_confidence'] += result['confidence']
            
            model_df = pd.DataFrame([
                {
                    'Model': model,
                    'Usage Count': stats['count'],
                    'Avg Confidence': stats['total_confidence'] / stats['count'],
                    'Usage %': stats['count'] / total_texts * 100
                }
                for model, stats in model_stats.items()
            ])
            
            st.dataframe(model_df, use_container_width=True)
            
            # Detailed Results Table with filtering
            st.subheader("Detailed Results")
            
            # Add filters
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                sentiment_filter = st.multiselect(
                    "Filter by Sentiment:",
                    ['positive', 'negative', 'neutral'],
                    default=['positive', 'negative', 'neutral']
                )
            
            with filter_col2:
                min_confidence = st.slider("Minimum Confidence:", 0.0, 1.0, 0.0)
            
            with filter_col3:
                max_results = st.number_input("Max Results to Show:", 1, len(st.session_state.results), min(50, len(st.session_state.results)))
            
            # Apply filters
            filtered_results = [
                r for r in st.session_state.results 
                if r['sentiment'] in sentiment_filter and r['confidence'] >= min_confidence
            ][:max_results]
            
            if filtered_results:
                results_df = pd.DataFrame([
                    {
                        'Text': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
                        'Sentiment': r['sentiment'].title(),
                        'Confidence': f"{r['confidence']:.1%}",
                        'Positive': f"{r['scores']['positive']:.1%}",
                        'Negative': f"{r['scores']['negative']:.1%}",
                        'Neutral': f"{r['scores']['neutral']:.1%}",
                        'Model': r['model']
                    }
                    for r in filtered_results
                ])
                
                st.dataframe(results_df, use_container_width=True)
            else:
                st.info("No results match the current filters.")
            
        else:
            st.info("No analysis results yet. Analyze some texts in the other tabs to see analytics here.")
    
    with tab4:
        st.header("Export Results")
        
        if st.session_state.results:
            col1, col2 = st.columns(2)
            
            with col1:
                export_format = st.selectbox("Choose export format:", ["CSV", "JSON", "TXT"])
                
                # Export options
                include_scores = st.checkbox("Include detailed scores", value=True)
                include_model_info = st.checkbox("Include model information", value=True)
            
            with col2:
                if st.button("Generate Export", type="primary"):
                    # Prepare data based on options
                    export_data = st.session_state.results.copy()
                    
                    if not include_scores:
                        for item in export_data:
                            if 'scores' in item:
                                del item['scores']
                    
                    if not include_model_info:
                        for item in export_data:
                            if 'model' in item:
                                del item['model']
                    
                    exported_data = export_results(export_data, export_format)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    if export_format == "CSV":
                        st.download_button(
                            label="📥 Download CSV",
                            data=exported_data,
                            file_name=f"sentiment_analysis_{timestamp}.csv",
                            mime="text/csv"
                        )
                    elif export_format == "JSON":
                        st.download_button(
                            label="📥 Download JSON",
                            data=exported_data,
                            file_name=f"sentiment_analysis_{timestamp}.json",
                            mime="application/json"
                        )
                    else:  # TXT
                        st.download_button(
                            label="📥 Download TXT",
                            data=exported_data,
                            file_name=f"sentiment_analysis_{timestamp}.txt",
                            mime="text/plain"
                        )
            
            # Export statistics
            st.subheader("Export Statistics")
            st.write(f"**Total results to export:** {len(st.session_state.results)}")
            st.write(f"**Sentiment breakdown:**")
            sentiment_counts = {}
            for result in st.session_state.results:
                sentiment = result['sentiment']
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            for sentiment, count in sentiment_counts.items():
                percentage = count / len(st.session_state.results) * 100
                st.write(f"- {sentiment.title()}: {count} ({percentage:.1f}%)")
            
            # Preview export
            st.subheader("Export Preview")
            if export_format == "CSV":
                preview_df = pd.DataFrame(st.session_state.results[:3])
                st.dataframe(preview_df)
            elif export_format == "JSON":
                st.json(st.session_state.results[:2])  # Show first 2 results
            else:  # TXT
                preview = export_results(st.session_state.results[:2], "TXT")
                st.text(preview)
            
            # Management options
            st.subheader("Data Management")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear All Results", type="secondary"):
                    st.session_state.results = []
                    st.success("All results cleared!")
                    st.rerun()
            
            with col2:
                if st.button("📊 Export Summary Report"):
                    # Generate a summary report
                    total = len(st.session_state.results)
                    if total > 0:
                        summary = f"""
SENTIMENT ANALYSIS SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
- Total texts analyzed: {total}
- Positive sentiment: {sum(1 for r in st.session_state.results if r['sentiment'] == 'positive')} ({sum(1 for r in st.session_state.results if r['sentiment'] == 'positive')/total*100:.1f}%)
- Negative sentiment: {sum(1 for r in st.session_state.results if r['sentiment'] == 'negative')} ({sum(1 for r in st.session_state.results if r['sentiment'] == 'negative')/total*100:.1f}%)
- Neutral sentiment: {sum(1 for r in st.session_state.results if r['sentiment'] == 'neutral')} ({sum(1 for r in st.session_state.results if r['sentiment'] == 'neutral')/total*100:.1f}%)

CONFIDENCE ANALYSIS:
- Average confidence: {sum(r['confidence'] for r in st.session_state.results)/total:.3f}
- High confidence (>0.8): {sum(1 for r in st.session_state.results if r['confidence'] > 0.8)} ({sum(1 for r in st.session_state.results if r['confidence'] > 0.8)/total*100:.1f}%)
- Medium confidence (0.6-0.8): {sum(1 for r in st.session_state.results if 0.6 <= r['confidence'] <= 0.8)} ({sum(1 for r in st.session_state.results if 0.6 <= r['confidence'] <= 0.8)/total*100:.1f}%)
- Low confidence (<0.6): {sum(1 for r in st.session_state.results if r['confidence'] < 0.6)} ({sum(1 for r in st.session_state.results if r['confidence'] < 0.6)/total*100:.1f}%)

MODEL USAGE:
"""
                        model_usage = {}
                        for result in st.session_state.results:
                            model = result['model']
                            model_usage[model] = model_usage.get(model, 0) + 1
                        
                        for model, count in model_usage.items():
                            summary += f"- {model}: {count} ({count/total*100:.1f}%)\n"
                        
                        st.download_button(
                            label="📥 Download Summary Report",
                            data=summary,
                            file_name=f"sentiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
        
        else:
            st.info("No results to export. Analyze some texts first!")
    
    st.markdown("**Built with Streamlit and Hugging Face API** | [Model Documentation](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)")

if __name__ == "__main__":
    main()

def main():
    if st.button("Analyze", type="primary"):
        if text_input.strip():
            with st.spinner("Analyzing sentiment..."):
                result = analyzer.analyze_sentiment(text_input)
                
                if result:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Analysis Results")
                        st.write(f"**Overall Sentiment:** {result['sentiment'].title()}")
                        st.write(f"**Confidence:** {result['confidence']:.2%}")
                        
                        if 'warning' in result:
                            st.warning(f"⚠️ {result['warning']}")
                        st.info(f"Model used: {result['model']}")
                    
                    with col2:
                        # Visual score representation
                        scores_df = pd.DataFrame([result['scores']]).T
                        scores_df.columns = ['Score']
                        scores_df.index.name = 'Sentiment'
                        
                        fig = px.bar(
                            x=scores_df.index,
                            y=scores_df['Score'],
                            color=scores_df.index,
                            color_discrete_map={
                                'positive': '#28a745',
                                'negative': '#dc3545',
                                'neutral': '#6c757d'
                            },
                            title="Sentiment Scores"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Add to results for later analysis
                    if 'results' not in st.session_state:
                        st.session_state.results = []
                    st.session_state.results.append(result)
                    
                else:
                    st.error("Failed to analyze sentiment. Please try again.")
        else:
            st.warning("Please enter some text to analyze.")

    # File Upload section
    uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
    
    if uploaded_file is not None:
        text_content = str(uploaded_file.read(), "utf-8")
        st.text_area("File content:", value=text_content[:500] + ("..." if len(text_content) > 500 else ""), height=150, disabled=True)
        
        if st.button("Analyze File Content", type="primary"):
            with st.spinner("Analyzing sentiment..."):
                result = analyzer.analyze_sentiment(text_content)
                
                if result:
                    # Same enhanced display as direct input
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        sentiment_class = f"sentiment-{result['sentiment']}"
                        st.markdown(f'<div class="metric-card"><h3>Primary Sentiment</h3><p class="{sentiment_class}">{result["sentiment"].title()}</p></div>', unsafe_allow_html=True)
                        
                        st.markdown(create_enhanced_sentiment_display(result), unsafe_allow_html=True)
                        
                        keywords = extract_enhanced_keywords(text_content, result)
                        if keywords:
                            st.markdown(f'<div class="metric-card"><h3>Key Sentiment Words</h3><p>{", ".join(keywords)}</p></div>', unsafe_allow_html=True)
                        
                        if result.get('warning'):
                            st.warning(f"⚠️ {result['warning']}")
                        st.info(f"Model used: {result['model']}")
                    
                    with col2:
                        scores_df = pd.DataFrame([result['scores']]).T
                        scores_df.columns = ['Score']
                        scores_df.index.name = 'Sentiment'
                        
                        fig = px.bar(
                            x=scores_df.index,
                            y=scores_df['Score'],
                            color=scores_df.index,
                            color_discrete_map={
                                'positive': '#28a745',
                                'negative': '#dc3545',
                                'neutral': '#6c757d'
                            },
                            title="Sentiment Scores"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if 'results' not in st.session_state:
                        st.session_state.results = []
                    st.session_state.results.append(result)
                else:
                    st.error("Failed to analyze sentiment. Please try again.")
    with tab2:
        st.header("Batch Processing")
        
        batch_method = st.radio("Choose batch input method:", ["Multiple Texts", "CSV Upload"])
        
        if batch_method == "Multiple Texts":
            batch_text = st.text_area("Enter multiple texts (one per line):", height=200, placeholder="Enter each text on a new line...")
            
            # Add batch examples
            if st.button("Load Example Batch"):
                example_batch = """I love this new smartphone! The camera quality is amazing.
                                   This restaurant has terrible service and cold food.
                                   The weather today is okay, nothing special.
                                   Amazing customer support! They solved my problem quickly.
                                   The movie was boring and too long. Waste of time.
                                   Standard delivery service, arrived on time as expected."""
                st.session_state.batch_example = example_batch
            
            # Use example batch if set
            if 'batch_example' in st.session_state:
                batch_text = st.session_state.batch_example
                del st.session_state.batch_example
            
            if st.button("Analyze Batch", type="primary"):
                if batch_text.strip():
                    texts = [text.strip() for text in batch_text.split('\n') if text.strip()]
                    
                    if texts:
                        progress_bar = st.progress(0, "Starting batch analysis...")
                        
                        batch_results = analyzer.batch_analyze(texts, progress_bar)
                        
                        if batch_results:
                            st.success(f"Successfully analyzed {len(batch_results)} texts!")
                            
                            # Enhanced batch results display
                            st.subheader("Batch Analysis Results")
                            
                            # Summary statistics
                            positive_count = sum(1 for r in batch_results if r['sentiment'] == 'positive')
                            negative_count = sum(1 for r in batch_results if r['sentiment'] == 'negative')
                            neutral_count = sum(1 for r in batch_results if r['sentiment'] == 'neutral')
                            avg_confidence = sum(r['confidence'] for r in batch_results) / len(batch_results)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Positive", positive_count, f"{positive_count/len(batch_results)*100:.1f}%")
                            with col2:
                                st.metric("Negative", negative_count, f"{negative_count/len(batch_results)*100:.1f}%")
                            with col3:
                                st.metric("Neutral", neutral_count, f"{neutral_count/len(batch_results)*100:.1f}%")
                            with col4:
                                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                            
                            # Individual results
                            for i, result in enumerate(batch_results):
                                confidence_emoji = "🔥" if result['confidence'] > 0.8 else "👍" if result['confidence'] > 0.6 else "🤔"
                                sentiment_emoji = "😊" if result['sentiment'] == 'positive' else "😞" if result['sentiment'] == 'negative' else "😐"
                                
                                with st.expander(f"{sentiment_emoji} Text {i+1}: {result['sentiment'].title()} {confidence_emoji} ({result['confidence']:.1%})"):
                                    st.write(f"**Text:** {result['text']}")
                                    st.write(f"**Sentiment:** {result['sentiment'].title()}")
                                    st.write(f"**Confidence:** {result['confidence']:.1%}")
                                    
                                    # Mini score chart
                                    scores = result['scores']
                                    score_text = " | ".join([f"{k.title()}: {v:.1%}" for k, v in scores.items()])
                                    st.write(f"**Scores:** {score_text}")
                                    
                                    if result.get('warning'):
                                        st.warning(result['warning'])
                            
                            # Add to session state
                            st.session_state.results.extend(batch_results)
                        else:
                            st.error("Failed to analyze texts. Please try again.")
                else:
                    st.warning("Please enter some texts to analyze.")
        
        else:  # CSV Upload
            uploaded_csv = st.file_uploader("Upload CSV file with 'text' column", type=['csv'])
            
            if uploaded_csv is not None:
                try:
                    df = pd.read_csv(uploaded_csv)
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())
                    
                    # Show column information
                    st.info(f"Columns found: {', '.join(df.columns.tolist())}")
                    
                    # Allow column selection if 'text' is not present
                    text_column = 'text'
                    if 'text' not in df.columns:
                        text_column = st.selectbox("Select the text column:", df.columns.tolist())
                    
                    if text_column in df.columns:
                        # Show sample data
                        st.write(f"Sample data from '{text_column}' column:")
                        st.write(df[text_column].head().tolist())
                        
                        if st.button("Analyze CSV Data", type="primary"):
                            texts = df[text_column].dropna().tolist()
                            
                            if texts:
                                progress_bar = st.progress(0, "Processing CSV data...")
                                
                                batch_results = analyzer.batch_analyze(texts, progress_bar)
                                
                                if batch_results:
                                    st.success(f"Successfully analyzed {len(batch_results)} texts from CSV!")
                                    
                                    # Create enhanced results DataFrame
                                    results_df = pd.DataFrame([
                                        {
                                            'original_text': r['text'],
                                            'sentiment': r['sentiment'],
                                            'confidence': r['confidence'],
                                            'positive_score': r['scores']['positive'],
                                            'negative_score': r['scores']['negative'],
                                            'neutral_score': r['scores']['neutral'],
                                            'model': r['model']
                                        }
                                        for r in batch_results
                                    ])
                                    
                                    st.subheader("CSV Analysis Results")
                                    st.dataframe(results_df)
                                    
                                    # Quick stats
                                    sentiment_counts = results_df['sentiment'].value_counts()
                                    st.write("**Sentiment Distribution:**")
                                    for sentiment, count in sentiment_counts.items():
                                        percentage = count / len(results_df) * 100
                                        st.write(f"- {sentiment.title()}: {count} ({percentage:.1f}%)")
                                    
                                    st.session_state.results.extend(batch_results)
                                else:
                                    st.error("Failed to analyze CSV data.")
                            else:
                                st.error(f"No valid text data found in column '{text_column}'.")
                    else:
                        st.error(f"Column '{text_column}' not found in the CSV file.")
                        
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        if st.session_state.results:
            # Enhanced summary metrics
            total_texts = len(st.session_state.results)
            positive_count = sum(1 for r in st.session_state.results if r['sentiment'] == 'positive')
            negative_count = sum(1 for r in st.session_state.results if r['sentiment'] == 'negative')
            neutral_count = sum(1 for r in st.session_state.results if r['sentiment'] == 'neutral')
            avg_confidence = sum(r['confidence'] for r in st.session_state.results) / total_texts
            
            # Calculate confidence ranges
            high_confidence = sum(1 for r in st.session_state.results if r['confidence'] > 0.8)
            medium_confidence = sum(1 for r in st.session_state.results if 0.6 <= r['confidence'] <= 0.8)
            low_confidence = sum(1 for r in st.session_state.results if r['confidence'] < 0.6)
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Texts", total_texts)
            with col2:
                st.metric("Positive ", positive_count, f"{positive_count/total_texts*100:.1f}%")
            with col3:
                st.metric("Negative ", negative_count, f"{negative_count/total_texts*100:.1f}%")
            with col4:
                st.metric("Neutral", neutral_count, f"{neutral_count/total_texts*100:.1f}%")
            with col5:
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            # Confidence breakdown
            st.subheader("Confidence Analysis")
            conf_col1, conf_col2, conf_col3 = st.columns(3)
            
            with conf_col1:
                st.metric("High Confidence", high_confidence, f"{high_confidence/total_texts*100:.1f}%")
                st.caption("(>80%)")
            with conf_col2:
                st.metric("Medium Confidence", medium_confidence, f"{medium_confidence/total_texts*100:.1f}%")
                st.caption("(60-80%)")
            with conf_col3:
                st.metric("Low Confidence", low_confidence, f"{low_confidence/total_texts*100:.1f}%")
                st.caption("(<60%)")
            
            # Enhanced Charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                sentiment_chart = create_sentiment_chart(st.session_state.results)
                if sentiment_chart:
                    st.plotly_chart(sentiment_chart, use_container_width=True)
            
            with chart_col2:
                confidence_chart = create_confidence_chart(st.session_state.results)
                if confidence_chart:
                    st.plotly_chart(confidence_chart, use_container_width=True)
            
            # Model performance analysis
            st.subheader("Model Performance")
            model_stats = {}
            for result in st.session_state.results:
                model = result['model']
                if model not in model_stats:
                    model_stats[model] = {'count': 0, 'total_confidence': 0}
                model_stats[model]['count'] += 1
                model_stats[model]['total_confidence'] += result['confidence']
            
            model_df = pd.DataFrame([
                {
                    'Model': model,
                    'Usage Count': stats['count'],
                    'Avg Confidence': stats['total_confidence'] / stats['count'],
                    'Usage %': stats['count'] / total_texts * 100
                }
                for model, stats in model_stats.items()
            ])
            
            st.dataframe(model_df, use_container_width=True)
            
            # Detailed Results Table with filtering
            st.subheader("Detailed Results")
            
            # Add filters
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                sentiment_filter = st.multiselect(
                    "Filter by Sentiment:",
                    ['positive', 'negative', 'neutral'],
                    default=['positive', 'negative', 'neutral']
                )
            
            with filter_col2:
                min_confidence = st.slider("Minimum Confidence:", 0.0, 1.0, 0.0)
            
            with filter_col3:
                max_results = st.number_input("Max Results to Show:", 1, len(st.session_state.results), min(50, len(st.session_state.results)))
            
            # Apply filters
            filtered_results = [
                r for r in st.session_state.results 
                if r['sentiment'] in sentiment_filter and r['confidence'] >= min_confidence
            ][:max_results]
            
            if filtered_results:
                results_df = pd.DataFrame([
                    {
                        'Text': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
                        'Sentiment': r['sentiment'].title(),
                        'Confidence': f"{r['confidence']:.1%}",
                        'Positive': f"{r['scores']['positive']:.1%}",
                        'Negative': f"{r['scores']['negative']:.1%}",
                        'Neutral': f"{r['scores']['neutral']:.1%}",
                        'Model': r['model']
                    }
                    for r in filtered_results
                ])
                
                st.dataframe(results_df, use_container_width=True)
            else:
                st.info("No results match the current filters.")
            
        else:
            st.info("No analysis results yet. Analyze some texts in the other tabs to see analytics here.")
    
    with tab4:
        st.header("Export Results")
        
        if st.session_state.results:
            col1, col2 = st.columns(2)
            
            with col1:
                export_format = st.selectbox("Choose export format:", ["CSV", "JSON", "TXT"])
                
                # Export options
                include_scores = st.checkbox("Include detailed scores", value=True)
                include_model_info = st.checkbox("Include model information", value=True)
            
            with col2:
                if st.button("Generate Export", type="primary"):
                    # Prepare data based on options
                    export_data = st.session_state.results.copy()
                    
                    if not include_scores:
                        for item in export_data:
                            if 'scores' in item:
                                del item['scores']
                    
                    if not include_model_info:
                        for item in export_data:
                            if 'model' in item:
                                del item['model']
                    
                    exported_data = export_results(export_data, export_format)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    if export_format == "CSV":
                        st.download_button(
                            label="📥 Download CSV",
                            data=exported_data,
                            file_name=f"sentiment_analysis_{timestamp}.csv",
                            mime="text/csv"
                        )
                    elif export_format == "JSON":
                        st.download_button(
                            label="📥 Download JSON",
                            data=exported_data,
                            file_name=f"sentiment_analysis_{timestamp}.json",
                            mime="application/json"
                        )
                    else:  # TXT
                        st.download_button(
                            label="📥 Download TXT",
                            data=exported_data,
                            file_name=f"sentiment_analysis_{timestamp}.txt",
                            mime="text/plain"
                        )
            
            # Export statistics
            st.subheader("Export Statistics")
            st.write(f"**Total results to export:** {len(st.session_state.results)}")
            st.write(f"**Sentiment breakdown:**")
            sentiment_counts = {}
            for result in st.session_state.results:
                sentiment = result['sentiment']
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            for sentiment, count in sentiment_counts.items():
                percentage = count / len(st.session_state.results) * 100
                st.write(f"- {sentiment.title()}: {count} ({percentage:.1f}%)")
            
            # Preview export
            st.subheader("Export Preview")
            if export_format == "CSV":
                preview_df = pd.DataFrame(st.session_state.results[:3])
                st.dataframe(preview_df)
            elif export_format == "JSON":
                st.json(st.session_state.results[:2])  # Show first 2 results
            else:  # TXT
                preview = export_results(st.session_state.results[:2], "TXT")
                st.text(preview)
            
            # Management options
            st.subheader("Data Management")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear All Results", type="secondary"):
                    st.session_state.results = []
                    st.success("All results cleared!")
                    st.rerun()
            
            with col2:
                if st.button("📊 Export Summary Report"):
                    # Generate a summary report
                    total = len(st.session_state.results)
                    if total > 0:
                        summary = f"""
SENTIMENT ANALYSIS SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
- Total texts analyzed: {total}
- Positive sentiment: {sum(1 for r in st.session_state.results if r['sentiment'] == 'positive')} ({sum(1 for r in st.session_state.results if r['sentiment'] == 'positive')/total*100:.1f}%)
- Negative sentiment: {sum(1 for r in st.session_state.results if r['sentiment'] == 'negative')} ({sum(1 for r in st.session_state.results if r['sentiment'] == 'negative')/total*100:.1f}%)
- Neutral sentiment: {sum(1 for r in st.session_state.results if r['sentiment'] == 'neutral')} ({sum(1 for r in st.session_state.results if r['sentiment'] == 'neutral')/total*100:.1f}%)

CONFIDENCE ANALYSIS:
- Average confidence: {sum(r['confidence'] for r in st.session_state.results)/total:.3f}
- High confidence (>0.8): {sum(1 for r in st.session_state.results if r['confidence'] > 0.8)} ({sum(1 for r in st.session_state.results if r['confidence'] > 0.8)/total*100:.1f}%)
- Medium confidence (0.6-0.8): {sum(1 for r in st.session_state.results if 0.6 <= r['confidence'] <= 0.8)} ({sum(1 for r in st.session_state.results if 0.6 <= r['confidence'] <= 0.8)/total*100:.1f}%)
- Low confidence (<0.6): {sum(1 for r in st.session_state.results if r['confidence'] < 0.6)} ({sum(1 for r in st.session_state.results if r['confidence'] < 0.6)/total*100:.1f}%)

MODEL USAGE:
"""
                        model_usage = {}
                        for result in st.session_state.results:
                            model = result['model']
                            model_usage[model] = model_usage.get(model, 0) + 1
                        
                        for model, count in model_usage.items():
                            summary += f"- {model}: {count} ({count/total*100:.1f}%)\n"
                        
                        st.download_button(
                            label="📥 Download Summary Report",
                            data=summary,
                            file_name=f"sentiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
        
        else:
            st.info("No results to export. Analyze some texts first!")
    
    # Enhanced Footer 
    st.markdown("**Built with Streamlit and Hugging Face API** | [Model Documentation](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)")

if __name__ == "__main__":
    main()
