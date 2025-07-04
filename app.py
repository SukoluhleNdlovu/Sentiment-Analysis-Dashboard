import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json
import time
from datetime import datetime
import re

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
        self.neutral_threshold = 0.3
        self.neutral_min_score = 0.4

    def normalize_sentiment(self, label):
        """Normalize sentiment labels to standard format"""
        label = str(label).upper().strip()
        if any(pos in label for pos in ["POSITIVE", "POS", "LABEL_2", "2"]):
            return "positive"
        elif any(neg in label for neg in ["NEGATIVE", "NEG", "LABEL_0", "0"]):
            return "negative"
        elif any(neu in label for neu in ["NEUTRAL", "NEU", "LABEL_1", "1"]):
            return "neutral"
        return "neutral"

    def contains_neutral_phrases(self, text):
        """Check for common neutral phrases"""
        neutral_phrases = [
            "it's okay", "not bad", "not great", "so so", 
            "nothing special", "average", "it's fine", "no strong opinion",
            "neither good nor bad", "middle of the road"
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

                total = sum(scores.values())
                if total > 0:
                    scores = {k: v / total for k, v in scores.items()}

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
        neu_count = sum(1 for word in neutral_indicators if word in text_lower) * 2

        sentiment_weights = {
            "positive": pos_count,
            "negative": neg_count,
            "neutral": neu_count
        }

        primary_sentiment = max(sentiment_weights, key=sentiment_weights.get)
        
        total = sum(sentiment_weights.values())
        if total == 0:
            primary_sentiment = "neutral"
            confidence = 0.5
        else:
            confidence = sentiment_weights[primary_sentiment] / total

        if primary_sentiment == "neutral":
            confidence = min(max(confidence, 0.4), 0.8)

        scores = {
            "positive": pos_count / max(total, 1),
            "negative": neg_count / max(total, 1),
            "neutral": neu_count / max(total, 1)
        }
        
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
            time.sleep(0.1)

        return results

def create_sentiment_display(result):
    """Create enhanced sentiment display"""
    sentiment_class = f"sentiment-{result['sentiment']}"
    confidence_bar = f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span><strong>Sentiment:</strong> <span class="{sentiment_class}">{result['sentiment'].title()}</span></span>
            <span><strong>Confidence:</strong> {result['confidence']:.1%}</span>
        </div>
        <div style="background-color: #e9ecef; border-radius: 0.25rem; height: 20px;">
            <div style="width: {result['confidence']*100}%; background-color: {'#28a745' if result['sentiment'] == 'positive' else '#dc3545' if result['sentiment'] == 'negative' else '#6c757d'}; height: 100%; border-radius: 0.25rem;"></div>
        </div>
    </div>

    scores_html = "<div style='margin: 10px 0;'><strong>Scores:</strong><br>"
    for sentiment, score in result["scores"].items():
        color = (
            "#28a745"
            if sentiment == "positive"
            else "#dc3545"
            if sentiment == "negative"
            else "#6c757d"
        )
        scores_html += f"<div style='margin: 5px 0; padding: 5px; background-color: {color}20; border-left: 3px solid {color};'>{sentiment.title()}: {score:.1%}</div>"
    scores_html += "</div>"

    return confidence_bar + scores_html

def create_charts(results):
    """Create visualization charts"""
    if not results:
        return None, None

    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for result in results:
        sentiment_counts[result["sentiment"]] += 1

    pie_chart = px.pie(
        values=list(sentiment_counts.values()),
        names=list(sentiment_counts.keys()),
        title="Sentiment Distribution",
        color_discrete_map={
            "positive": "#28a745",
            "negative": "#dc3545",
            "neutral": "#6c757d",
        },
    )

    confidence_data = [
        {"confidence": r["confidence"], "sentiment": r["sentiment"]} for r in results
    ]
    df = pd.DataFrame(confidence_data)

    confidence_chart = px.box(
        df,
        x="sentiment",
        y="confidence",
        color="sentiment",
        title="Confidence Distribution",
        color_discrete_map={
            "positive": "#28a745",
            "negative": "#dc3545",
            "neutral": "#6c757d",
        },
    )

    return pie_chart, confidence_chart

def export_results(results, format_type):
    """Export results in specified format"""
    if format_type == "CSV":
        data = []
        for r in results:
            data.append(
                {
                    "text": r["text"],
                    "sentiment": r["sentiment"],
                    "confidence": r["confidence"],
                    "positive_score": r["scores"]["positive"],
                    "negative_score": r["scores"]["negative"],
                    "neutral_score": r["scores"]["neutral"],
                    "model": r["model"],
                }
            )
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
    st.markdown(
        '<h1 class="main-header">Enhanced Sentiment Analysis Dashboard</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.header("🔧 Configuration")
    api_key = st.sidebar.text_input("Hugging Face API Key", type="password")

    if not api_key:
        st.warning("Please enter your Hugging Face API key in the sidebar.")
        st.info("Get your API key from: https://huggingface.co/settings/tokens")
        return

    analyzer = SentimentAnalyzer(api_key)

    if "results" not in st.session_state:
        st.session_state.results = []

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Single Analysis", "Batch Analysis", "Analytics", "Export"]
    )

    with tab1:
        st.header("Single Text Analysis")
        input_method = st.radio("Input method:", ["Direct Input", "File Upload"])

        if input_method == "Direct Input":
            user_text = st.text_area("Enter text:", height=150)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Positive Example"):
                    user_text = "I love this product! It's amazing!"
            with col2:
                if st.button("Negative Example"):
                    user_text = "This is terrible. I hate it!"
            with col3:
                if st.button("Neutral Example"):
                    user_text = "The product is okay, nothing special."

            if st.button("Analyze", type="primary") and user_text.strip():
                with st.spinner("Analyzing..."):
                    result = analyzer.analyze_sentiment(user_text)
                    if result:
                        st.markdown(create_sentiment_display(result), unsafe_allow_html=True)
                        st.info(f"Model: {result['model']}")
                        st.session_state.results.append(result)

        else:
            uploaded_file = st.file_uploader("Upload text file", type=["txt"])
            if uploaded_file:
                text_content = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", value=text_content[:500], disabled=True)
                if st.button("Analyze File", type="primary"):
                    with st.spinner("Analyzing..."):
                        result = analyzer.analyze_sentiment(text_content)
                        if result:
                            st.markdown(create_sentiment_display(result), unsafe_allow_html=True)
                            st.session_state.results.append(result)

    with tab2:
        st.header("Batch Analysis")
        batch_method = st.radio("Batch method:", ["Multiple Texts", "CSV Upload"])

        if batch_method == "Multiple Texts":
            batch_text = st.text_area("Enter texts (one per line):", height=200)
            
            if st.button("Example Batch"):
                batch_text = """I love this product!
This is terrible quality.
The service was okay.
Amazing customer support!
It's neither good nor bad.
Very disappointing experience."""
            
            if st.button("Analyze Batch", type="primary") and batch_text.strip():
                texts = [t.strip() for t in batch_text.split("\n") if t.strip()]
                batch_results = analyzer.batch_analyze(texts)
                
                if batch_results:
                    st.success(f"Analyzed {len(batch_results)} texts!")
                    
                    positive = sum(1 for r in batch_results if r["sentiment"] == "positive")
                    negative = sum(1 for r in batch_results if r["sentiment"] == "negative")
                    neutral = sum(1 for r in batch_results if r["sentiment"] == "neutral")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Positive", positive)
                    with col2:
                        st.metric("Negative", negative)
                    with col3:
                        st.metric("Neutral", neutral)
                    
                    for i, result in enumerate(batch_results):
                        with st.expander(f"Text {i+1}: {result['sentiment'].title()} ({result['confidence']:.1%})"):
                            st.write(f"**Text:** {result['text']}")
                            st.markdown(create_sentiment_display(result), unsafe_allow_html=True)
                    
                    st.session_state.results.extend(batch_results)

        else:
            uploaded_csv = st.file_uploader("Upload CSV with 'text' column", type=["csv"])
            if uploaded_csv:
                df = pd.read_csv(uploaded_csv)
                st.dataframe(df.head())
                text_column = st.selectbox("Select text column:", df.columns)
                
                if st.button("Analyze CSV", type="primary"):
                    texts = df[text_column].dropna().tolist()
                    batch_results = analyzer.batch_analyze(texts)
                    
                    if batch_results:
                        st.success(f"Analyzed {len(batch_results)} texts!")
                        results_df = pd.DataFrame([
                            {
                                "text": r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"],
                                "sentiment": r["sentiment"],
                                "confidence": r["confidence"],
                            }
                            for r in batch_results
                        ])
                        st.dataframe(results_df)
                        st.session_state.results.extend(batch_results)

    with tab3:
        st.header("Analytics")
        if st.session_state.results:
            total = len(st.session_state.results)
            positive = sum(1 for r in st.session_state.results if r["sentiment"] == "positive")
            negative = sum(1 for r in st.session_state.results if r["sentiment"] == "negative")
            neutral = sum(1 for r in st.session_state.results if r["sentiment"] == "neutral")
            avg_confidence = sum(r["confidence"] for r in st.session_state.results) / total
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", total)
            with col2:
                st.metric("Positive", positive, f"{positive/total*100:.1f}%")
            with col3:
                st.metric("Negative", negative, f"{negative/total*100:.1f}%")
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            pie_chart, confidence_chart = create_charts(st.session_state.results)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(pie_chart, use_container_width=True)
            with col2:
                st.plotly_chart(confidence_chart, use_container_width=True)
            
            st.subheader("Detailed Results")
            results_df = pd.DataFrame([
                {
                    "Text": r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"],
                    "Sentiment": r["sentiment"].title(),
                    "Confidence": f"{r['confidence']:.1%}",
                    "Model": r["model"],
                }
                for r in st.session_state.results
            ])
            st.dataframe(results_df)
        else:
            st.info("No results yet. Analyze some texts first!")

    with tab4:
        st.header("Export Results")
        if st.session_state.results:
            export_format = st.selectbox("Format:", ["CSV", "JSON", "TXT"])
            
            if st.button("Generate Export", type="primary"):
                exported_data = export_results(st.session_state.results, export_format)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_extension = export_format.lower()
                mime_type = {
                    "CSV": "text/csv",
                    "JSON": "application/json",
                    "TXT": "text/plain",
                }[export_format]
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=exported_data,
                    file_name=f"sentiment_results_{timestamp}.{file_extension}",
                    mime=mime_type,
                )
            
            if st.button("Clear Results", type="secondary"):
                st.session_state.results = []
                st.success("Results cleared!")
                st.rerun()
        else:
            st.info("No results to export!")

if __name__ == "__main__":
    main()
