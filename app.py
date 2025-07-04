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
        self.models = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "siebert/sentiment-roberta-large-english",
            "distilbert-base-uncased-finetuned-sst-2-english"
        ]
        self.current_model = None
        self.api_url = None
    
    def normalize_sentiment(self, label):
        """Normalize sentiment labels to standard format"""
        label = label.upper().strip()
        if any(pos in label for pos in ['POSITIVE', 'POS', 'LABEL_2', '2']):
            return 'positive'
        elif any(neg in label for neg in ['NEGATIVE', 'NEG', 'LABEL_0', '0']):
            return 'negative'
        elif any(neu in label for neu in ['NEUTRAL', 'NEU', 'LABEL_1', '1']):
            return 'neutral'
        else:
            # Default fallback based on common patterns
            if 'LABEL_1' in label or '1' in label:
                return 'neutral'
            return 'neutral'
    
    def find_working_model(self):
        """Find the first working model"""
        if self.current_model:
            return True
            
        for model in self.models:
            try:
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                response = requests.post(api_url, headers=self.headers, 
                                       json={"inputs": "test"}, timeout=10)
                
                if response.status_code == 200:
                    self.current_model = model
                    self.api_url = api_url
                    return True
            except:
                continue
        return False
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if not self.find_working_model():
            return self.fallback_analysis(text)
        
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())[:500]
        
        try:
            response = requests.post(self.api_url, headers=self.headers, 
                                   json={"inputs": text}, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                return self.process_response(result, text)
            elif response.status_code == 503:
                st.warning("Model loading, please wait...")
                time.sleep(2)
                return self.analyze_sentiment(text)
            else:
                return self.fallback_analysis(text)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return self.fallback_analysis(text)
    
    def process_response(self, result, original_text):
        """Process API response - FIXED for proper neutral detection"""
        try:
            if isinstance(result, list) and len(result) > 0:
                sentiments = result[0] if isinstance(result[0], list) else result
                
                # Initialize scores
                raw_scores = {}
                
                # Extract raw scores from model response
                for item in sentiments:
                    if 'label' in item and 'score' in item:
                        normalized = self.normalize_sentiment(item['label'])
                        raw_scores[normalized] = float(item['score'])
                
                # Ensure we have all three sentiment categories
                final_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
                
                # Handle different model output formats
                if len(raw_scores) == 3:
                    # Model returned all three classes - use directly
                    final_scores.update(raw_scores)
                
                elif len(raw_scores) == 2:
                    # Model returned only 2 classes (usually pos/neg)
                    final_scores.update(raw_scores)
                    
                    # Calculate neutral based on confidence gap
                    pos_score = final_scores.get('positive', 0.0)
                    neg_score = final_scores.get('negative', 0.0)
                    
                    # If both positive and negative scores are low, it's likely neutral
                    max_score = max(pos_score, neg_score)
                    
                    if max_score < 0.75:  # Low confidence threshold
                        # Redistribute to make it more neutral
                        confidence_gap = 1.0 - max_score
                        final_scores['neutral'] = confidence_gap * 0.8
                        
                        # Adjust other scores
                        remaining = 1.0 - final_scores['neutral']
                        total_existing = pos_score + neg_score
                        if total_existing > 0:
                            final_scores['positive'] = (pos_score / total_existing) * remaining
                            final_scores['negative'] = (neg_score / total_existing) * remaining
                
                elif len(raw_scores) == 1:
                    # Model returned only one class
                    final_scores.update(raw_scores)
                    single_score = list(raw_scores.values())[0]
                    
                    if single_score < 0.8:  # Low confidence - likely neutral
                        final_scores['neutral'] = 0.6
                        remaining = 0.4
                        for key in final_scores:
                            if key != 'neutral' and key not in raw_scores:
                                final_scores[key] = remaining / 2
                
                # Normalize scores to ensure they sum to 1
                total = sum(final_scores.values())
                if total > 0:
                    final_scores = {k: v/total for k, v in final_scores.items()}
                
                # Determine primary sentiment with better neutral detection
                max_sentiment = max(final_scores.keys(), key=lambda k: final_scores[k])
                max_score = final_scores[max_sentiment]
                
                # Enhanced neutral detection logic
                sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
                highest_score = sorted_scores[0][1]
                second_highest = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
                
                # If scores are close or highest score is low, classify as neutral
                if (highest_score - second_highest < 0.2) or (highest_score < 0.6):
                    primary_sentiment = 'neutral'
                    # Boost neutral score if it's the most appropriate
                    if final_scores['neutral'] < 0.4:
                        final_scores['neutral'] = 0.5
                        remaining = 0.5
                        final_scores['positive'] = remaining * (final_scores['positive'] / (final_scores['positive'] + final_scores['negative'] + 0.001))
                        final_scores['negative'] = remaining * (final_scores['negative'] / (final_scores['positive'] + final_scores['negative'] + 0.001))
                else:
                    primary_sentiment = max_sentiment
                
                return {
                    'text': original_text,
                    'sentiment': primary_sentiment,
                    'confidence': final_scores[primary_sentiment],
                    'scores': final_scores,
                    'model': self.current_model
                }
                
        except Exception as e:
            st.error(f"Error processing response: {str(e)}")
            return self.fallback_analysis(original_text)
        
        return self.fallback_analysis(original_text)
    
    def fallback_analysis(self, text):
        """Enhanced rule-based fallback with better neutral detection"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'awesome', 'fantastic', 'wonderful', 'outstanding', 'brilliant', 'superb']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'pathetic', 'useless', 'disappointing', 'annoying', 'frustrating']
        neutral_words = ['okay', 'fine', 'average', 'normal', 'standard', 'typical', 'regular', 'moderate', 'fair', 'decent']
        
        text_lower = text.lower()
        
        # Count occurrences
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        neu_count = sum(1 for word in neutral_words if word in text_lower)
        
        # Enhanced logic for neutral detection
        if neu_count > 0 and (pos_count + neg_count) <= neu_count:
            sentiment = 'neutral'
            confidence = min(0.6 + neu_count * 0.1, 0.85)
        elif pos_count > neg_count and pos_count > neu_count:
            sentiment = 'positive'
            confidence = min(0.6 + (pos_count - max(neg_count, neu_count)) * 0.1, 0.9)
        elif neg_count > pos_count and neg_count > neu_count:
            sentiment = 'negative'
            confidence = min(0.6 + (neg_count - max(pos_count, neu_count)) * 0.1, 0.9)
        else:
            # Default to neutral if no clear sentiment
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
            scores = {sentiment: confidence}
            remaining = 1 - confidence
            for s in ['positive', 'negative', 'neutral']:
                if s not in scores:
                    if s == 'neutral':
                        scores[s] = remaining * 0.4
                    else:
                        scores[s] = remaining * 0.3
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': scores,
            'model': 'fallback'
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
            time.sleep(0.1)
        
        return results

def create_sentiment_display(result):
    """Create enhanced sentiment display"""
    confidence_bar = f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span><strong>Confidence:</strong></span>
            <span>{result['confidence']:.1%}</span>
        </div>
        <div style="background-color: #e9ecef; border-radius: 0.25rem; height: 20px;">
            <div style="width: {result['confidence']*100}%; background-color: {'#28a745' if result['sentiment'] == 'positive' else '#dc3545' if result['sentiment'] == 'negative' else '#6c757d'}; height: 100%; border-radius: 0.25rem;"></div>
        </div>
    </div>
    """
    
    scores_html = "<div style='margin: 10px 0;'><strong>Scores:</strong><br>"
    for sentiment, score in result['scores'].items():
        color = '#28a745' if sentiment == 'positive' else '#dc3545' if sentiment == 'negative' else '#6c757d'
        scores_html += f"<div style='margin: 5px 0; padding: 5px; background-color: {color}20; border-left: 3px solid {color};'>{sentiment.title()}: {score:.1%}</div>"
    scores_html += "</div>"
    
    return confidence_bar + scores_html

def create_charts(results):
    """Create visualization charts"""
    if not results:
        return None, None
    
    # Sentiment distribution
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for result in results:
        sentiment_counts[result['sentiment']] += 1
    
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
    confidence_data = [{'confidence': r['confidence'], 'sentiment': r['sentiment']} for r in results]
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

def export_results(results, format_type):
    """Export results in specified format"""
    if format_type == "CSV":
        data = []
        for r in results:
            data.append({
                'text': r['text'],
                'sentiment': r['sentiment'],
                'confidence': r['confidence'],
                'positive_score': r['scores']['positive'],
                'negative_score': r['scores']['negative'],
                'neutral_score': r['scores']['neutral'],
                'model': r['model']
            })
        return pd.DataFrame(data).to_csv(index=False)
    
    elif format_type == "JSON":
        return json.dumps(results, indent=2)
    
    else:  # TXT
        output = []
        for i, r in enumerate(results, 1):
            output.append(f"Result {i}:")
            output.append(f"Text: {r['text']}")
            output.append(f"Sentiment: {r['sentiment'].title()}")
            output.append(f"Confidence: {r['confidence']:.3f}")
            output.append(f"Model: {r['model']}")
            output.append("-" * 50)
        return "\n".join(output)

def main():
    st.markdown('<h1 class="main-header">Enhanced Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    api_key = st.sidebar.text_input("Hugging Face API Key", type="password")
    
    if not api_key:
        st.warning("Please enter your Hugging Face API key in the sidebar.")
        st.info("Get your API key from: https://huggingface.co/settings/tokens")
        return
    
    analyzer = SentimentAnalyzer(api_key)
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Single Analysis", "Batch Analysis", "Analytics", "Export"])
    
    with tab1:
        st.header("Single Text Analysis")
        
        # Input methods
        input_method = st.radio("Input method:", ["Direct Input", "File Upload"])
        
        if input_method == "Direct Input":
            user_text = st.text_area("Enter text:", height=150)
            
            # Example buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Positive Example"):
                    user_text = "I love this product! It's amazing and fantastic!"
                    st.rerun()
            with col2:
                if st.button("Negative Example"):
                    user_text = "This is terrible and awful. I hate it completely!"
                    st.rerun()
            with col3:
                if st.button("Neutral Example"):
                    user_text = "The product is okay and average. It's fine, nothing special."
                    st.rerun()
            
            if st.button("Analyze", type="primary") and user_text.strip():
                with st.spinner("Analyzing..."):
                    result = analyzer.analyze_sentiment(user_text)
                    
                    if result:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Display sentiment
                            sentiment_class = f"sentiment-{result['sentiment']}"
                            st.markdown(f'<div class="metric-card"><h3>Sentiment</h3><p class="{sentiment_class}">{result["sentiment"].title()}</p></div>', unsafe_allow_html=True)
                            
                            # Display details
                            st.markdown(create_sentiment_display(result), unsafe_allow_html=True)
                            st.info(f"Model: {result['model']}")
                        
                        with col2:
                            # Chart
                            scores_df = pd.DataFrame([result['scores']]).T
                            fig = px.bar(
                                x=scores_df.index, y=scores_df[0],
                                color=scores_df.index,
                                color_discrete_map={
                                    'positive': '#28a745',
                                    'negative': '#dc3545',
                                    'neutral': '#6c757d'
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.session_state.results.append(result)
        
        else:  # File Upload
            uploaded_file = st.file_uploader("Upload text file", type=['txt'])
            
            if uploaded_file:
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
    
    with tab2:
        st.header("Batch Analysis")
        
        batch_method = st.radio("Batch method:", ["Multiple Texts", "CSV Upload"])
        
        if batch_method == "Multiple Texts":
            batch_text = st.text_area("Enter texts (one per line):", height=200)
            
            if st.button("Example Batch"):
                batch_text = """I love this product! It's amazing!
This is terrible quality and awful.
The service was okay and average.
Amazing customer support! Fantastic!
The product is fine, nothing special.
Horrible experience, worst ever!"""
                st.rerun()
            
            if st.button("Analyze Batch", type="primary") and batch_text.strip():
                texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
                
                batch_results = analyzer.batch_analyze(texts)
                
                if batch_results:
                    st.success(f"Analyzed {len(batch_results)} texts!")
                    
                    # Summary
                    positive = sum(1 for r in batch_results if r['sentiment'] == 'positive')
                    negative = sum(1 for r in batch_results if r['sentiment'] == 'negative')
                    neutral = sum(1 for r in batch_results if r['sentiment'] == 'neutral')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Positive", positive)
                    with col2:
                        st.metric("Negative", negative)
                    with col3:
                        st.metric("Neutral", neutral)
                    
                    # Individual results
                    for i, result in enumerate(batch_results):
                        with st.expander(f"Text {i+1}: {result['sentiment'].title()} ({result['confidence']:.1%})"):
                            st.write(f"**Text:** {result['text']}")
                            st.write(f"**Sentiment:** {result['sentiment'].title()}")
                            st.write(f"**Confidence:** {result['confidence']:.1%}")
                            st.write(f"**Scores:** Positive: {result['scores']['positive']:.1%}, Negative: {result['scores']['negative']:.1%}, Neutral: {result['scores']['neutral']:.1%}")
                    
                    st.session_state.results.extend(batch_results)
        
        else:  # CSV Upload
            uploaded_csv = st.file_uploader("Upload CSV with 'text' column", type=['csv'])
            
            if uploaded_csv:
                df = pd.read_csv(uploaded_csv)
                st.dataframe(df.head())
                
                text_column = st.selectbox("Select text column:", df.columns)
                
                if st.button("Analyze CSV", type="primary"):
                    texts = df[text_column].dropna().tolist()
                    batch_results = analyzer.batch_analyze(texts)
                    
                    if batch_results:
                        st.success(f"Analyzed {len(batch_results)} texts!")
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame([{
                            'text': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
                            'sentiment': r['sentiment'],
                            'confidence': r['confidence'],
                            'positive_score': r['scores']['positive'],
                            'negative_score': r['scores']['negative'],
                            'neutral_score': r['scores']['neutral']
                        } for r in batch_results])
                        
                        st.dataframe(results_df)
                        st.session_state.results.extend(batch_results)
    
    with tab3:
        st.header("Analytics")
        
        if st.session_state.results:
            # Summary metrics
            total = len(st.session_state.results)
            positive = sum(1 for r in st.session_state.results if r['sentiment'] == 'positive')
            negative = sum(1 for r in st.session_state.results if r['sentiment'] == 'negative')
            neutral = sum(1 for r in st.session_state.results if r['sentiment'] == 'neutral')
            avg_confidence = sum(r['confidence'] for r in st.session_state.results) / total
            
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
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            # Charts
            pie_chart, confidence_chart = create_charts(st.session_state.results)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(pie_chart, use_container_width=True)
            with col2:
                st.plotly_chart(confidence_chart, use_container_width=True)
            
            # Detailed results
            st.subheader("Detailed Results")
            results_df = pd.DataFrame([{
                'Text': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
                'Sentiment': r['sentiment'].title(),
                'Confidence': f"{r['confidence']:.1%}",
                'Positive': f"{r['scores']['positive']:.1%}",
                'Negative': f"{r['scores']['negative']:.1%}",
                'Neutral': f"{r['scores']['neutral']:.1%}",
                'Model': r['model']
            } for r in st.session_state.results])
            
            st.dataframe(results_df)
        
        else:
            st.info("No results yet. Analyze some texts first!")
    
    with tab4:
        st.header("Export Results")
        
        if st.session_state.results:
            export_format = st.selectbox("Format:", ["CSV", "JSON", "TXT"])
            
            if st.button("Generate Export", type="primary"):
                exported_data = export_results(st.session_state.results, export_format)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                file_extension = export_format.lower()
                mime_type = {
                    'CSV': 'text/csv',
                    'JSON': 'application/json',
                    'TXT': 'text/plain'
                }[export_format]
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=exported_data,
                    file_name=f"sentiment_results_{timestamp}.{file_extension}",
                    mime=mime_type
                )
            
            # Clear results
            if st.button("Clear Results", type="secondary"):
                st.session_state.results = []
                st.success("Results cleared!")
                st.rerun()
        
        else:
            st.info("No results to export!")

if __name__ == "__main__":
    main()
