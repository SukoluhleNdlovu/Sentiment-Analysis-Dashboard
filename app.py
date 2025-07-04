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
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .sentiment-positive { color: #28a745; font-weight: bold; font-size: 1.2em; }
    .sentiment-negative { color: #dc3545; font-weight: bold; font-size: 1.2em; }
    .sentiment-neutral { color: #6c757d; font-weight: bold; font-size: 1.2em; }
</style>
""", unsafe_allow_html=True)

class SentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        # Updated models with better neutral detection
        self.models = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "siebert/sentiment-roberta-large-english",
            "distilbert-base-uncased-finetuned-sst-2-english"
        ]
        self.current_model = None
        self.api_url = None
    
    def normalize_sentiment(self, label):
        """Enhanced sentiment normalization with better neutral detection"""
        if not label:
            return 'neutral'
        
        label = str(label).upper().strip()
        
        # Handle different label formats
        if any(pos in label for pos in ['POSITIVE', 'POS', 'LABEL_2', '2']):
            return 'positive'
        elif any(neg in label for neg in ['NEGATIVE', 'NEG', 'LABEL_0', '0']):
            return 'negative'
        elif any(neu in label for neu in ['NEUTRAL', 'NEU', 'LABEL_1', '1']):
            return 'neutral'
        else:
            # Default to neutral for unknown labels
            return 'neutral'
    
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
                    json={"inputs": "This is a test"}, 
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Check if the model returns proper sentiment results
                    if isinstance(result, list) and len(result) > 0:
                        self.current_model = model
                        self.api_url = api_url
                        return True
            except Exception as e:
                # Log error but continue to next model
                st.sidebar.warning(f"Model {model} failed: {str(e)}")
                continue
        return False
    
    def analyze_sentiment(self, text):
        """Enhanced sentiment analysis with better neutral detection"""
        if not text or not text.strip():
            return None
            
        if not self.find_working_model():
            return self.fallback_analysis(text)
        
        # Clean and prepare text
        text = re.sub(r'\s+', ' ', text.strip())[:500]
        
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json={"inputs": text}, 
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                return self.process_response(result, text)
            elif response.status_code == 503:
                # Model is loading
                with st.spinner("Model loading, please wait..."):
                    time.sleep(3)
                    return self.analyze_sentiment(text)
            else:
                st.warning(f"API returned status {response.status_code}")
                return self.fallback_analysis(text)
                
        except Exception as e:
            st.error(f"Error analyzing text: {str(e)}")
            return self.fallback_analysis(text)
    
    def process_response(self, result, original_text):
        """Enhanced response processing with better neutral detection"""
        try:
            if not result:
                return self.fallback_analysis(original_text)
                
            if isinstance(result, list) and len(result) > 0:
                sentiments = result[0] if isinstance(result[0], list) else result
                
                # Initialize scores
                scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
                
                # Process sentiment scores
                for item in sentiments:
                    if isinstance(item, dict) and 'label' in item and 'score' in item:
                        try:
                            normalized = self.normalize_sentiment(item['label'])
                            score = float(item['score'])
                            scores[normalized] = max(scores[normalized], score)
                        except (ValueError, TypeError):
                            continue
                
                # If no neutral score was found, calculate it
                if scores['neutral'] == 0.0 and (scores['positive'] > 0 or scores['negative'] > 0):
                    # Calculate neutral as the remaining probability
                    total_polar = scores['positive'] + scores['negative']
                    if total_polar < 1.0:
                        scores['neutral'] = 1.0 - total_polar
                
                # Normalize scores to sum to 1
                total = sum(scores.values())
                if total > 0:
                    scores = {k: v/total for k, v in scores.items()}
                else:
                    # If all scores are 0, default to neutral
                    scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
                
                # Determine primary sentiment with neutral threshold
                primary_sentiment = self.determine_primary_sentiment(scores)
                
                return {
                    'text': original_text,
                    'sentiment': primary_sentiment,
                    'confidence': scores[primary_sentiment],
                    'scores': scores,
                    'model': self.current_model
                }
            return self.fallback_analysis(original_text)
        except Exception as e:
            st.error(f"Error processing response: {str(e)}")
            return self.fallback_analysis(original_text)
    
    def determine_primary_sentiment(self, scores):
        """Determine primary sentiment with neutral threshold"""
        # Define thresholds for neutral detection
        NEUTRAL_THRESHOLD = 0.4  # If neutral score is above this, consider neutral
        CONFIDENCE_THRESHOLD = 0.6  # If max polar sentiment is below this, consider neutral
        
        max_sentiment = max(scores.keys(), key=lambda k: scores[k])
        
        # If neutral has highest score and above threshold
        if max_sentiment == 'neutral' and scores['neutral'] >= NEUTRAL_THRESHOLD:
            return 'neutral'
        
        # If the difference between positive and negative is small, consider neutral
        pos_neg_diff = abs(scores['positive'] - scores['negative'])
        if pos_neg_diff < 0.1 and scores['neutral'] >= 0.3:
            return 'neutral'
        
        # If max polar sentiment is below confidence threshold, consider neutral
        max_polar_score = max(scores['positive'], scores['negative'])
        if max_polar_score < CONFIDENCE_THRESHOLD and scores['neutral'] >= 0.25:
            return 'neutral'
        
        return max_sentiment
    
    def fallback_analysis(self, text):
        """Enhanced rule-based fallback with better neutral detection"""
        # Expanded word lists
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'awesome',
            'fantastic', 'wonderful', 'outstanding', 'brilliant', 'superb', 'marvelous',
            'delighted', 'pleased', 'happy', 'satisfied', 'impressed', 'recommend'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting',
            'disappointing', 'frustrated', 'annoying', 'useless', 'pathetic', 'dreadful',
            'unacceptable', 'regret', 'disappointed', 'complaint', 'problem', 'issues'
        ]
        
        neutral_words = [
            'okay', 'ok', 'fine', 'average', 'normal', 'standard', 'typical',
            'regular', 'moderate', 'decent', 'acceptable', 'fair', 'adequate',
            'reasonable', 'nothing', 'whatever', 'maybe', 'perhaps', 'probably'
        ]
        
        text_lower = text.lower()
        
        # Count sentiment indicators
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        neu_count = sum(1 for word in neutral_words if word in text_lower)
        
        # Enhanced neutral detection
        total_words = len(text_lower.split())
        sentiment_word_ratio = (pos_count + neg_count + neu_count) / max(total_words, 1)
        
        # Determine sentiment
        if neu_count > 0 and (pos_count == 0 and neg_count == 0):
            # Explicitly neutral words with no polar sentiment
            sentiment = 'neutral'
            confidence = min(0.7, 0.5 + neu_count * 0.1)
        elif pos_count == 0 and neg_count == 0 and neu_count == 0:
            # No sentiment indicators - likely neutral
            sentiment = 'neutral'
            confidence = 0.6
        elif sentiment_word_ratio < 0.1:
            # Very few sentiment words relative to text length
            sentiment = 'neutral'
            confidence = 0.55
        elif pos_count > neg_count and pos_count > neu_count:
            sentiment = 'positive'
            confidence = min(0.8, 0.6 + (pos_count - max(neg_count, neu_count)) * 0.1)
        elif neg_count > pos_count and neg_count > neu_count:
            sentiment = 'negative'
            confidence = min(0.8, 0.6 + (neg_count - max(pos_count, neu_count)) * 0.1)
        else:
            # Equal or mixed sentiment
            sentiment = 'neutral'
            confidence = 0.5
        
        # Create balanced scores
        if sentiment == 'neutral':
            scores = {
                'neutral': confidence,
                'positive': (1 - confidence) * 0.4,
                'negative': (1 - confidence) * 0.6
            }
        else:
            scores = {
                sentiment: confidence,
                'neutral': (1 - confidence) * 0.6,
                ('positive' if sentiment == 'negative' else 'negative'): (1 - confidence) * 0.4
            }
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': scores,
            'model': 'enhanced_fallback'
        }

    def batch_analyze(self, texts):
        """Analyze multiple texts with better error handling"""
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, text in enumerate(texts):
            try:
                status_text.text(f"Analyzing text {i+1}/{len(texts)}: {text[:50]}...")
                progress_bar.progress((i + 1) / len(texts))
                
                result = self.analyze_sentiment(text)
                if result:
                    results.append(result)
                    
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                st.error(f"Error analyzing text {i+1}: {str(e)}")
                continue
        
        status_text.text("Analysis complete!")
        return results

def create_sentiment_display(result):
    """Create enhanced sentiment display with error handling"""
    if not result:
        return "<div>No result to display</div>"
    
    try:
        confidence = result.get('confidence', 0)
        sentiment = result.get('sentiment', 'neutral')
        scores = result.get('scores', {'positive': 0, 'negative': 0, 'neutral': 1})
        
        confidence_bar = f"""
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span><strong>Confidence:</strong></span>
                <span>{confidence:.1%}</span>
            </div>
            <div style="background-color: #e9ecef; border-radius: 0.25rem; height: 20px;">
                <div style="width: {confidence*100}%; background-color: {'#28a745' if sentiment == 'positive' else '#dc3545' if sentiment == 'negative' else '#6c757d'}; height: 100%; border-radius: 0.25rem;"></div>
            </div>
        </div>
        """
        
        scores_html = "<div style='margin: 10px 0;'><strong>Detailed Scores:</strong><br>"
        for sentiment_type, score in scores.items():
            color = '#28a745' if sentiment_type == 'positive' else '#dc3545' if sentiment_type == 'negative' else '#6c757d'
            scores_html += f"<div style='margin: 5px 0; padding: 5px; background-color: {color}20; border-left: 3px solid {color};'>{sentiment_type.title()}: {score:.1%}</div>"
        scores_html += "</div>"
        
        return confidence_bar + scores_html
    except Exception as e:
        return f"<div>Error displaying result: {str(e)}</div>"

def create_charts(results):
    """Create visualization charts with error handling"""
    if not results:
        return None, None
    
    try:
        # Sentiment distribution
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for result in results:
            sentiment = result.get('sentiment', 'neutral')
            sentiment_counts[sentiment] += 1
        
        pie_chart = px.pie(
            values=list(sentiment_counts.values()),
            names=list(sentiment_counts.keys()),
            title="Sentiment Distribution",
            color_discrete_map={
                'positive': '#28a745',
                'negative': '#dc3545',
                'neutral': '#6c757d'
            }
        )
        
        # Confidence distribution
        confidence_data = []
        for r in results:
            confidence_data.append({
                'confidence': r.get('confidence', 0),
                'sentiment': r.get('sentiment', 'neutral')
            })
        
        df = pd.DataFrame(confidence_data)
        
        confidence_chart = px.box(
            df, x='sentiment', y='confidence', color='sentiment',
            title="Confidence Distribution",
            color_discrete_map={
                'positive': '#28a745',
                'negative': '#dc3545',
                'neutral': '#6c757d'
            }
        )
        
        return pie_chart, confidence_chart
    except Exception as e:
        st.error(f"Error creating charts: {str(e)}")
        return None, None

def export_results(results, format_type):
    """Export results in specified format with error handling"""
    if not results:
        return ""
    
    try:
        if format_type == "CSV":
            data = []
            for r in results:
                data.append({
                    'text': r.get('text', ''),
                    'sentiment': r.get('sentiment', 'neutral'),
                    'confidence': r.get('confidence', 0),
                    'positive_score': r.get('scores', {}).get('positive', 0),
                    'negative_score': r.get('scores', {}).get('negative', 0),
                    'neutral_score': r.get('scores', {}).get('neutral', 0),
                    'model': r.get('model', 'unknown')
                })
            return pd.DataFrame(data).to_csv(index=False)
        
        elif format_type == "JSON":
            return json.dumps(results, indent=2)
        
        else:  # TXT
            output = []
            for i, r in enumerate(results, 1):
                output.append(f"Result {i}:")
                output.append(f"Text: {r.get('text', '')}")
                output.append(f"Sentiment: {r.get('sentiment', 'neutral').title()}")
                output.append(f"Confidence: {r.get('confidence', 0):.3f}")
                scores = r.get('scores', {})
                output.append(f"Positive: {scores.get('positive', 0):.3f}")
                output.append(f"Negative: {scores.get('negative', 0):.3f}")
                output.append(f"Neutral: {scores.get('neutral', 0):.3f}")
                output.append(f"Model: {r.get('model', 'unknown')}")
                output.append("-" * 50)
            return "\n".join(output)
    except Exception as e:
        st.error(f"Error exporting results: {str(e)}")
        return ""

def main():
    st.markdown('<h1 class="main-header">📊 Enhanced Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    api_key = st.sidebar.text_input("Hugging Face API Key", type="password")
    
    if not api_key:
        st.warning("Please enter your Hugging Face API key in the sidebar.")
        st.info("Get your API key from: https://huggingface.co/settings/tokens")
        return
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # Initialize analyzer
    try:
        analyzer = SentimentAnalyzer(api_key)
    except Exception as e:
        st.error(f"Error initializing analyzer: {str(e)}")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Analysis", "📁 Batch Analysis", "📊 Analytics", "📥 Export"])
    
    with tab1:
        st.header("Single Text Analysis")
        
        # Input methods
        input_method = st.radio("Input method:", ["Direct Input", "File Upload"])
        
        if input_method == "Direct Input":
            user_text = st.text_area("Enter text:", height=150)
            
            # Enhanced example buttons with clear sentiment indicators
            st.subheader("Try These Examples:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("✅ Positive Example"):
                    st.session_state.user_text = "I absolutely love this product! It's fantastic and exceeded my expectations!"
                    st.success("Loaded positive example")
            with col2:
                if st.button("❌ Negative Example"):
                    st.session_state.user_text = "This is completely terrible. I hate it and want my money back!"
                    st.error("Loaded negative example")
            with col3:
                if st.button("⚪ Neutral Example"):
                    st.session_state.user_text = "The product is okay, nothing special. It works as expected."
                    st.info("Loaded neutral example")
            with col4:
                if st.button("🔄 Mixed Example"):
                    st.session_state.user_text = "The product has some good features but also some issues. It's average overall."
                    st.warning("Loaded mixed sentiment example")
            
            # Use session state value if available
            if 'user_text' in st.session_state:
                user_text = st.session_state.user_text
                
            if st.button("Analyze", type="primary") and user_text and user_text.strip():
                with st.spinner("Analyzing..."):
                    try:
                        result = analyzer.analyze_sentiment(user_text)
                        
                        if result:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Display sentiment with enhanced styling
                                sentiment_class = f"sentiment-{result['sentiment']}"
                                st.markdown(f'<div class="metric-card"><h3>Sentiment</h3><p class="{sentiment_class}">{result["sentiment"].title()}</p></div>', unsafe_allow_html=True)
                                
                                # Display detailed information
                                st.markdown(create_sentiment_display(result), unsafe_allow_html=True)
                                st.info(f"Model: {result['model']}")
                            
                            with col2:
                                # Enhanced chart with all scores
                                try:
                                    scores_df = pd.DataFrame([result['scores']]).T
                                    fig = px.bar(
                                        x=scores_df.index, y=scores_df[0],
                                        color=scores_df.index,
                                        title="Sentiment Scores",
                                        color_discrete_map={
                                            'positive': '#28a745',
                                            'negative': '#dc3545',
                                            'neutral': '#6c757d'
                                        }
                                    )
                                    fig.update_layout(showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating chart: {str(e)}")
                            
                            st.session_state.results.append(result)
                        else:
                            st.error("Failed to analyze text. Please try again.")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
        
        else:  # File Upload
            uploaded_file = st.file_uploader("Upload text file", type=['txt'])
            
            if uploaded_file:
                try:
                    text_content = str(uploaded_file.read(), "utf-8")
                    st.text_area("File content:", value=text_content[:500], disabled=True)
                    
                    if st.button("Analyze File", type="primary"):
                        with st.spinner("Analyzing..."):
                            result = analyzer.analyze_sentiment(text_content)
                            
                            if result:
                                # Same display as direct input
                                sentiment_class = f"sentiment-{result['sentiment']}"
                                st.markdown(f'<div class="metric-card"><h3>Sentiment</h3><p class="{sentiment_class}">{result["sentiment"].title()}</p></div>', unsafe_allow_html=True)
                                st.markdown(create_sentiment_display(result), unsafe_allow_html=True)
                                st.session_state.results.append(result)
                            else:
                                st.error("Failed to analyze file content.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.header("Batch Analysis")
        
        batch_method = st.radio("Batch method:", ["Multiple Texts", "CSV Upload"])
        
        if batch_method == "Multiple Texts":
            batch_text = st.text_area("Enter texts (one per line):", height=200)
            
            if st.button("Analyze Batch", type="primary") and batch_text.strip():
                try:
                    texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
                    
                    if texts:
                        batch_results = analyzer.batch_analyze(texts)
                        
                        if batch_results:
                            st.success(f"Analyzed {len(batch_results)} texts!")
                            
                            # Enhanced summary with sentiment breakdown
                            st.subheader("📊 Sentiment Analysis Summary")
                            positive = sum(1 for r in batch_results if r.get('sentiment') == 'positive')
                            negative = sum(1 for r in batch_results if r.get('sentiment') == 'negative')
                            neutral = sum(1 for r in batch_results if r.get('sentiment') == 'neutral')
                            total = len(batch_results)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("📝 Total Analyzed", total)
                            with col2:
                                st.metric("✅ Positive", positive, f"{positive/total*100:.1f}%")
                            with col3:
                                st.metric("❌ Negative", negative, f"{negative/total*100:.1f}%")
                            with col4:
                                st.metric("⚪ Neutral", neutral, f"{neutral/total*100:.1f}%")
                            
                            st.session_state.results.extend(batch_results)
                        else:
                            st.error("No results from batch analysis.")
                    else:
                        st.error("No valid texts found.")
                except Exception as e:
                    st.error(f"Error in batch analysis: {str(e)}")
        
        else:  # CSV Upload
            uploaded_csv = st.file_uploader("Upload CSV with 'text' column", type=['csv'])
            
            if uploaded_csv:
                try:
                    df = pd.read_csv(uploaded_csv)
                    st.dataframe(df.head())
                    
                    text_column = st.selectbox("Select text column:", df.columns)
                    
                    if st.button("Analyze CSV", type="primary"):
                        texts = df[text_column].dropna().astype(str).tolist()
                        batch_results = analyzer.batch_analyze(texts)
                        
                        if batch_results:
                            st.success(f"Analyzed {len(batch_results)} texts from CSV!")
                            st.session_state.results.extend(batch_results)
                        else:
                            st.error("No results from CSV analysis.")
                except Exception as e:
                    st.error(f"Error processing CSV: {str(e)}")
    
    with tab3:
        st.header("Analytics")
        
        if st.session_state.results:
            try:
                # Enhanced summary metrics
                total = len(st.session_state.results)
                positive = sum(1 for r in st.session_state.results if r.get('sentiment') == 'positive')
                negative = sum(1 for r in st.session_state.results if r.get('sentiment') == 'negative')
                neutral = sum(1 for r in st.session_state.results if r.get('sentiment') == 'neutral')
                
                confidences = [r.get('confidence', 0) for r in st.session_state.results]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total", total)
                with col2:
                    st.metric("Positive", positive, f"{positive/total*100:.1f}%")
                with col3:
                    st.metric("Negative", negative, f"{negative/total*100:.1f}%")
                with col4:
                    st.metric("Neutral", neutral, f"{neutral/total*100:.1f}%")
                with col5:
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                
                # Charts
                pie_chart, confidence_chart = create_charts(st.session_state.results)
                
                if pie_chart and confidence_chart:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(pie_chart, use_container_width=True)
                    with col2:
                        st.plotly_chart(confidence_chart, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in analytics: {str(e)}")
        else:
            st.info("No results yet. Analyze some texts first!")
    
    with tab4:
        st.header("Export Results")
        
        if st.session_state.results:
            try:
                export_format = st.selectbox("Format:", ["CSV", "JSON", "TXT"])
                
                if st.button("Generate Export", type="primary"):
                    exported_data = export_results(st.session_state.results, export_format)
                    
                    if exported_data:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        file_extension = export_format.lower()
                        mime_type = {
                            'CSV': 'text/csv',
                            'JSON': 'application/json',
                            'TXT': 'text/plain'
                        }[export_format]
                        
                        st.download_button(
                            label=f"📥 Download {export_format}",
                            data=exported_data,
                            file_name=f"sentiment_results_{timestamp}.{file_extension}",
                            mime=mime_type
                        )
                    else:
                        st.error("Failed to export results.")
                
                # Clear results
                if st.button("Clear Results", type="secondary"):
                    st.session_state.results = []
                    st.success("Results cleared!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error in export: {str(e)}")
        else:
            st.info("No results to export!")

if __name__ == "__main__":
    main()
    
