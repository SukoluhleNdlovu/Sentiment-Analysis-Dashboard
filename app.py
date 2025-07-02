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
        """Normalize different sentiment label formats to standard format"""
        label = label.upper().strip()
        
        # Handle different label formats
        positive_indicators = ['POSITIVE', 'POS', 'LABEL_1', '1', 'GOOD', 'HAPPY', 'JOY']
        negative_indicators = ['NEGATIVE', 'NEG', 'LABEL_0', '0', 'BAD', 'SAD', 'ANGER']
        neutral_indicators = ['NEUTRAL', 'NEU', 'LABEL_2', '2', 'MIXED', 'OBJECTIVE']
        
        for indicator in positive_indicators:
            if indicator in label:
                return 'positive'
        
        for indicator in negative_indicators:
            if indicator in label:
                return 'negative'
        
        for indicator in neutral_indicators:
            if indicator in label:
                return 'neutral'
        
        # Default classification based on score if available
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
        """Enhanced API response processing"""
        try:
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    sentiments = result[0]
                else:
                    sentiments = result
                
                processed_scores = {}
                
                # Process each sentiment score
                for item in sentiments:
                    if 'label' in item and 'score' in item:
                        normalized_label = self.normalize_sentiment_labels(item['label'])
                        score = float(item['score'])
                        
                        # Aggregate scores for the same sentiment
                        if normalized_label in processed_scores:
                            processed_scores[normalized_label] = max(processed_scores[normalized_label], score)
                        else:
                            processed_scores[normalized_label] = score
                
                # Ensure all three sentiments are present
                for sentiment in ['positive', 'negative', 'neutral']:
                    if sentiment not in processed_scores:
                        processed_scores[sentiment] = 0.0
                
                # Normalize scores to sum to 1.0
                total_score = sum(processed_scores.values())
                if total_score > 0:
                    processed_scores = {k: v/total_score for k, v in processed_scores.items()}
                else:
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
            return None
    
    def enhance_sentiment_detection(self, result):
        """Apply additional logic to improve sentiment detection accuracy"""
        text = result['text'].lower()
        scores = result['scores'].copy()
        
        # Rule-based adjustments for common patterns
        positive_boosters = ['love', 'excellent', 'amazing', 'fantastic', 'perfect', 'wonderful']
        negative_boosters = ['hate', 'terrible', 'awful', 'horrible', 'worst', 'disgusting']
        
        # Count sentiment indicators
        positive_count = sum(1 for word in positive_boosters if word in text)
        negative_count = sum(1 for word in negative_boosters if word in text)
        
        # Adjust scores based on strong indicators
        if positive_count > negative_count and positive_count > 0:
            scores['positive'] = min(scores['positive'] + 0.1, 1.0)
        elif negative_count > positive_count and negative_count > 0:
            scores['negative'] = min(scores['negative'] + 0.1, 1.0)
        
        # Handle negations
        negation_pattern = r'\b(not|no|never|neither|nobody|nothing|nowhere|hardly|scarcely|barely)\b'
        if re.search(negation_pattern, text):
            # Swap positive and negative if negation is present near sentiment words
            if scores['positive'] > scores['negative']:
                scores['positive'], scores['negative'] = scores['negative'], scores['positive']
        
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
        """Simple rule-based fallback sentiment analysis"""
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'perfect', 'best', 'awesome', 'brilliant', 'outstanding']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'poor', 'disgusting', 'pathetic', 'useless']
        
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        if positive_score > negative_score:
            sentiment = 'positive'
            confidence = min(0.6 + (positive_score - negative_score) * 0.1, 0.9)
        elif negative_score > positive_score:
            sentiment = 'negative'
            confidence = min(0.6 + (negative_score - positive_score) * 0.1, 0.9)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        # Create normalized scores
        if sentiment == 'positive':
            scores = {'positive': confidence, 'negative': (1-confidence)/2, 'neutral': (1-confidence)/2}
        elif sentiment == 'negative':
            scores = {'negative': confidence, 'positive': (1-confidence)/2, 'neutral': (1-confidence)/2}
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
    
    def batch_analyze(self, texts, progress_bar=None):
        """Enhanced batch analysis with better progress tracking"""
        results = []
        total = len(texts)
        failed_count = 0
        
        for i, text in enumerate(texts):
            if progress_bar:
                progress_bar.progress((i + 1) / total, f"Processing {i + 1}/{total} (Failed: {failed_count})")
            
            result = self.analyze_sentiment(text)
            if result:
                results.append(result)
            else:
                failed_count += 1
                # Add a fallback result
                fallback_result = self.fallback_sentiment_analysis(text)
                results.append(fallback_result)
            
            # Adaptive delay based on API response
            time.sleep(0.1 if i % 10 != 0 else 0.5)
        
        return results

def create_enhanced_sentiment_display(result):
    """Create an enhanced visual display for sentiment results"""
    sentiment = result['sentiment']
    confidence = result['confidence']
    scores = result['scores']
    
    # Create confidence visualization
    confidence_html = f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span><strong>Confidence:</strong></span>
            <span>{confidence:.1%}</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence*100}%; background-color: {'#28a745' if sentiment == 'positive' else '#dc3545' if sentiment == 'negative' else '#6c757d'};"></div>
        </div>
    </div>
    """
    
    # Create score breakdown
    score_html = "<div style='margin: 10px 0;'><strong>Score Breakdown:</strong><br>"
    for sent, score in scores.items():
        color = '#28a745' if sent == 'positive' else '#dc3545' if sent == 'negative' else '#6c757d'
        score_html += f"<div style='margin: 5px 0; padding: 5px; background-color: {color}20; border-left: 3px solid {color};'>{sent.title()}: {score:.1%}</div>"
    score_html += "</div>"
    
    return confidence_html + score_html

def extract_enhanced_keywords(text, sentiment_result):
    """Enhanced keyword extraction with better sentiment correlation"""
    # Expanded keyword lists
    positive_keywords = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'perfect', 'best',
        'awesome', 'brilliant', 'outstanding', 'superb', 'magnificent', 'incredible', 'fabulous',
        'terrific', 'marvelous', 'phenomenal', 'exceptional', 'impressive', 'delightful'
    ]
    
    negative_keywords = [
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'poor',
        'disgusting', 'pathetic', 'useless', 'dreadful', 'appalling', 'atrocious', 'abysmal',
        'deplorable', 'horrendous', 'ghastly', 'hideous', 'repulsive', 'revolting'
    ]
    
    neutral_keywords = [
        'okay', 'fine', 'average', 'normal', 'standard', 'typical', 'regular', 'common',
        'ordinary', 'moderate', 'adequate', 'acceptable', 'reasonable', 'fair'
    ]
    
    # Extract words from text
    words = re.findall(r'\b\w+\b', text.lower())
    found_keywords = []
    
    sentiment = sentiment_result['sentiment']
    
    # Find relevant keywords based on sentiment
    if sentiment == 'positive':
        found_keywords = [word for word in words if word in positive_keywords]
    elif sentiment == 'negative':
        found_keywords = [word for word in words if word in negative_keywords]
    else:
        found_keywords = [word for word in words if word in neutral_keywords]
    
    # Also include cross-sentiment keywords if they appear
    all_sentiment_words = positive_keywords + negative_keywords + neutral_keywords
    additional_keywords = [word for word in words if word in all_sentiment_words and word not in found_keywords]
    
    # Combine and limit results
    all_keywords = found_keywords + additional_keywords[:3]
    return list(set(all_keywords))[:5]

def create_sentiment_chart(results):
    """Enhanced sentiment distribution chart"""
    if not results:
        return None
    
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for result in results:
        sentiment_counts[result['sentiment']] += 1
    
    # Create a more detailed pie chart
    fig = px.pie(
        values=list(sentiment_counts.values()),
        names=list(sentiment_counts.keys()),
        title="Sentiment Distribution",
        color_discrete_map={
            'positive': '#28a745',
            'negative': '#dc3545',
            'neutral': '#6c757d'
        },
        hole=0.3  # Donut chart for better visual appeal
    )
    
    # Add percentage labels
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    return fig

def create_confidence_chart(results):
    """Enhanced confidence score visualization"""
    if not results:
        return None
    
    # Create a more detailed confidence analysis
    confidence_data = []
    
    for result in results:
        confidence_data.append({
            'confidence': result['confidence'],
            'sentiment': result['sentiment'],
            'model': result.get('model', 'unknown')
        })
    
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
            'neutral': '#6c757d'
        }
    )
    
    fig.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Confidence Score",
        showlegend=False
    )
    
    return fig

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
    
    # Enhanced Footer with tips
    st.markdown("---")
    st.markdown("### 💡 Tips for Better Sentiment Analysis")
    
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    
    with tip_col1:
        st.markdown("""
        **Text Quality:**
        - Use clear, complete sentences
        - Avoid excessive punctuation
        - Check for typos and grammar
        """)
    
    with tip_col2:
        st.markdown("""
        **Confidence Interpretation:**
        - >80%: High confidence
        - 60-80%: Medium confidence  
        - <60%: Low confidence
        """)
    
    with tip_col3:
        st.markdown("""
        **Best Practices:**
        - Analyze similar text types together
        - Review low-confidence results
        - Use batch processing for efficiency
        """)
    
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
    
    # Enhanced Footer with tips
    st.markdown("---")
    st.markdown("### 💡 Tips for Better Sentiment Analysis")
    
    st.markdown("**Built with Streamlit and Hugging Face API** | [Model Documentation](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)")

if __name__ == "__main__":
    main()
