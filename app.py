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

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
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
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class SentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        # Try multiple models as fallbacks
        self.models = [
            "distilbert-base-uncased-finetuned-sst-2-english",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "siebert/sentiment-roberta-large-english"
        ]
        self.current_model = None
        self.api_url = None
    
    def test_model(self, model_name, test_text="I love this product!"):
        """Test if a model is working"""
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        payload = {"inputs": test_text}
        
        try:
            response = requests.post(api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"Status {response.status_code}: {response.text}"
        except Exception as e:
            return False, str(e)
    
    def find_working_model(self):
        """Find the first working model from the list"""
        if self.current_model:
            return True
            
        st.info("🔍 Finding the best available model...")
        
        for model in self.models:
            st.write(f"Testing {model}...")
            success, result = self.test_model(model)
            
            if success:
                self.current_model = model
                self.api_url = f"https://api-inference.huggingface.co/models/{model}"
                st.success(f"✅ Using model: {model}")
                return True
            else:
                st.warning(f"❌ {model}: {result}")
        
        st.error("❌ No working models found. Please check your API key or try again later.")
        return False
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of a single text"""
        # Ensure we have a working model
        if not self.find_working_model():
            return None
            
        try:
            payload = {"inputs": text}
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        # Format: [[{"label": "POSITIVE", "score": 0.9}]]
                        sentiments = result[0]
                    else:
                        # Format: [{"label": "POSITIVE", "score": 0.9}]
                        sentiments = result
                    
                    processed_result = {}
                    
                    for item in sentiments:
                        label = item['label'].upper()
                        score = item['score']
                        
                        # Map different label formats to standard names
                        if 'NEGATIVE' in label or 'NEG' in label or 'LABEL_0' in label:
                            processed_result['negative'] = score
                        elif 'POSITIVE' in label or 'POS' in label or 'LABEL_1' in label:
                            processed_result['positive'] = score
                        else:
                            processed_result['neutral'] = score
                    
                    # Ensure we have at least positive and negative
                    if 'positive' not in processed_result:
                        processed_result['positive'] = 0.0
                    if 'negative' not in processed_result:
                        processed_result['negative'] = 0.0
                    if 'neutral' not in processed_result:
                        processed_result['neutral'] = 1.0 - processed_result['positive'] - processed_result['negative']
                    
                    # Determine primary sentiment
                    primary_sentiment = max(processed_result.keys(), key=lambda k: processed_result[k])
                    confidence = processed_result[primary_sentiment]
                    
                    return {
                        'sentiment': primary_sentiment,
                        'confidence': confidence,
                        'scores': processed_result,
                        'text': text,
                        'model': self.current_model
                    }
                else:
                    st.error("Unexpected API response format")
                    return None
                    
            elif response.status_code == 503:
                st.warning("⏳ Model is loading, please wait 10-20 seconds and try again")
                return None
            elif response.status_code == 429:
                st.warning("⏸️ Rate limit reached, please wait a moment and try again")
                return None
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Error analyzing sentiment: {str(e)}")
            return None
    
    def batch_analyze(self, texts, progress_bar=None):
        """Analyze sentiment for multiple texts"""
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if progress_bar:
                progress_bar.progress((i + 1) / total, f"Processing {i + 1}/{total}")
            
            result = self.analyze_sentiment(text)
            if result:
                results.append(result)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        return results

def extract_keywords(text, sentiment_result):
    """Simple keyword extraction based on sentiment"""
    # This is a simplified approach - in production, you'd use more sophisticated NLP
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'perfect', 'best']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'poor']
    
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = []
    
    for word in words:
        if word in positive_words and sentiment_result['sentiment'] == 'positive':
            keywords.append(word)
        elif word in negative_words and sentiment_result['sentiment'] == 'negative':
            keywords.append(word)
    
    return keywords[:5]  # Return top 5 keywords

def create_sentiment_chart(results):
    """Create sentiment distribution chart"""
    if not results:
        return None
    
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for result in results:
        sentiment_counts[result['sentiment']] += 1
    
    fig = px.pie(
        values=list(sentiment_counts.values()),
        names=list(sentiment_counts.keys()),
        title="Sentiment Distribution",
        color_discrete_map={
            'positive': '#28a745',
            'negative': '#dc3545',
            'neutral': '#6c757d'
        }
    )
    
    return fig

def create_confidence_chart(results):
    """Create confidence score distribution"""
    if not results:
        return None
    
    confidences = [result['confidence'] for result in results]
    sentiments = [result['sentiment'] for result in results]
    
    fig = px.histogram(
        x=confidences,
        color=sentiments,
        title="Confidence Score Distribution",
        labels={'x': 'Confidence Score', 'y': 'Count'},
        color_discrete_map={
            'positive': '#28a745',
            'negative': '#dc3545',
            'neutral': '#6c757d'
        }
    )
    
    return fig

def export_results(results, format_type):
    """Export results in different formats"""
    if format_type == "CSV":
        df = pd.DataFrame(results)
        return df.to_csv(index=False)
    elif format_type == "JSON":
        return json.dumps(results, indent=2)
    elif format_type == "TXT":
        output = []
        for result in results:
            output.append(f"Text: {result['text']}")
            output.append(f"Sentiment: {result['sentiment']}")
            output.append(f"Confidence: {result['confidence']:.3f}")
            output.append("-" * 50)
        return "\n".join(output)

# Main Application
def main():
    st.markdown('<h1 class="main-header">📊 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for API configuration
    st.sidebar.header("🔧 Configuration")
    api_key = st.sidebar.text_input("Hugging Face API Key", type="password", help="Enter your Hugging Face API key")
    
    if not api_key:
        st.warning("Please enter your Hugging Face API key in the sidebar to begin.")
        st.info("You can get your API key from: https://huggingface.co/settings/tokens")
        return
    
    # Initialize the analyzer
    analyzer = SentimentAnalyzer(api_key)
    
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
            
            if st.button("Analyze Sentiment", type="primary"):
                if user_text.strip():
                    with st.spinner("Analyzing sentiment..."):
                        result = analyzer.analyze_sentiment(user_text)
                        
                        if result:
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                sentiment_class = f"sentiment-{result['sentiment']}"
                                st.markdown(f'<div class="metric-card"><h3>Sentiment</h3><p class="{sentiment_class}">{result["sentiment"].title()}</p></div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f'<div class="metric-card"><h3>Confidence</h3><p>{result["confidence"]:.3f}</p></div>', unsafe_allow_html=True)
                            
                            with col3:
                                keywords = extract_keywords(user_text, result)
                                st.markdown(f'<div class="metric-card"><h3>Key Words</h3><p>{", ".join(keywords) if keywords else "None detected"}</p></div>', unsafe_allow_html=True)
                            
                            # Detailed scores
                            st.subheader("Detailed Scores")
                            scores_df = pd.DataFrame([result['scores']]).T
                            scores_df.columns = ['Score']
                            scores_df.index.name = 'Sentiment'
                            st.bar_chart(scores_df)
                            
                            # Add to results for later analysis
                            st.session_state.results.append(result)
                            
                        else:
                            st.error("Failed to analyze sentiment. Please try again.")
                else:
                    st.warning("Please enter some text to analyze.")
        
        else:  # File Upload
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            
            if uploaded_file is not None:
                text_content = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", value=text_content, height=150, disabled=True)
                
                if st.button("Analyze File Content", type="primary"):
                    with st.spinner("Analyzing sentiment..."):
                        result = analyzer.analyze_sentiment(text_content)
                        
                        if result:
                            # Same display logic as direct input
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                sentiment_class = f"sentiment-{result['sentiment']}"
                                st.markdown(f'<div class="metric-card"><h3>Sentiment</h3><p class="{sentiment_class}">{result["sentiment"].title()}</p></div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f'<div class="metric-card"><h3>Confidence</h3><p>{result["confidence"]:.3f}</p></div>', unsafe_allow_html=True)
                            
                            with col3:
                                keywords = extract_keywords(text_content, result)
                                st.markdown(f'<div class="metric-card"><h3>Key Words</h3><p>{", ".join(keywords) if keywords else "None detected"}</p></div>', unsafe_allow_html=True)
                            
                            st.session_state.results.append(result)
    
    with tab2:
        st.header("Batch Processing")
        
        batch_method = st.radio("Choose batch input method:", ["Multiple Texts", "CSV Upload"])
        
        if batch_method == "Multiple Texts":
            batch_text = st.text_area("Enter multiple texts (one per line):", height=200, placeholder="Enter each text on a new line...")
            
            if st.button("Analyze Batch", type="primary"):
                if batch_text.strip():
                    texts = [text.strip() for text in batch_text.split('\n') if text.strip()]
                    
                    if texts:
                        progress_bar = st.progress(0, "Starting batch analysis...")
                        
                        batch_results = analyzer.batch_analyze(texts, progress_bar)
                        
                        if batch_results:
                            st.success(f"Successfully analyzed {len(batch_results)} texts!")
                            
                            # Display batch results
                            for i, result in enumerate(batch_results):
                                with st.expander(f"Text {i+1}: {result['sentiment'].title()} (Confidence: {result['confidence']:.3f})"):
                                    st.write(f"**Text:** {result['text'][:200]}...")
                                    st.write(f"**Sentiment:** {result['sentiment'].title()}")
                                    st.write(f"**Confidence:** {result['confidence']:.3f}")
                            
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
                    
                    if 'text' in df.columns:
                        if st.button("Analyze CSV Data", type="primary"):
                            texts = df['text'].tolist()
                            progress_bar = st.progress(0, "Processing CSV data...")
                            
                            batch_results = analyzer.batch_analyze(texts, progress_bar)
                            
                            if batch_results:
                                st.success(f"Successfully analyzed {len(batch_results)} texts from CSV!")
                                st.session_state.results.extend(batch_results)
                            else:
                                st.error("Failed to analyze CSV data.")
                    else:
                        st.error("CSV file must contain a 'text' column.")
                        
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        if st.session_state.results:
            # Summary metrics
            total_texts = len(st.session_state.results)
            positive_count = sum(1 for r in st.session_state.results if r['sentiment'] == 'positive')
            negative_count = sum(1 for r in st.session_state.results if r['sentiment'] == 'negative')
            neutral_count = sum(1 for r in st.session_state.results if r['sentiment'] == 'neutral')
            avg_confidence = sum(r['confidence'] for r in st.session_state.results) / total_texts
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Texts", total_texts)
            with col2:
                st.metric("Positive", positive_count)
            with col3:
                st.metric("Negative", negative_count)
            with col4:
                st.metric("Neutral", neutral_count)
            with col5:
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_chart = create_sentiment_chart(st.session_state.results)
                if sentiment_chart:
                    st.plotly_chart(sentiment_chart, use_container_width=True)
            
            with col2:
                confidence_chart = create_confidence_chart(st.session_state.results)
                if confidence_chart:
                    st.plotly_chart(confidence_chart, use_container_width=True)
            
            # Results table
            st.subheader("Detailed Results")
            results_df = pd.DataFrame(st.session_state.results)
            st.dataframe(results_df, use_container_width=True)
            
        else:
            st.info("No analysis results yet. Analyze some texts in the other tabs to see analytics here.")
    
    with tab4:
        st.header("Export Results")
        
        if st.session_state.results:
            col1, col2 = st.columns(2)
            
            with col1:
                export_format = st.selectbox("Choose export format:", ["CSV", "JSON", "TXT"])
            
            with col2:
                if st.button("Generate Export", type="primary"):
                    exported_data = export_results(st.session_state.results, export_format)
                    
                    if export_format == "CSV":
                        st.download_button(
                            label="Download CSV",
                            data=exported_data,
                            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    elif export_format == "JSON":
                        st.download_button(
                            label="Download JSON",
                            data=exported_data,
                            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    else:  # TXT
                        st.download_button(
                            label="Download TXT",
                            data=exported_data,
                            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
            
            # Preview export
            st.subheader("Export Preview")
            if export_format == "CSV":
                preview_df = pd.DataFrame(st.session_state.results)
                st.dataframe(preview_df.head())
            elif export_format == "JSON":
                st.json(st.session_state.results[:3])  # Show first 3 results
            else:  # TXT
                preview = export_results(st.session_state.results[:2], "TXT")
                st.text(preview)
            
            # Clear results option
            if st.button("Clear All Results", type="secondary"):
                st.session_state.results = []
                st.success("All results cleared!")
                st.rerun()
        
        else:
            st.info("No results to export. Analyze some texts first!")
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with Streamlit and Hugging Face API** | [Documentation](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)")

if __name__ == "__main__":
    main()