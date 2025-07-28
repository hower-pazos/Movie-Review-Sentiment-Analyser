import streamlit as st

st.set_page_config(
    page_title="Movie Review Sentiment Analyser", 
    page_icon="🎬",
    layout="wide"
)

import pandas as pd
import numpy as np
import nltk
nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)
nltk.download('wordnet', force=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
from time import time
import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = "IMDB_Dataset.csv"

MODEL_COMPARISON_RESULTS = {
    'Neural Network': {'accuracy': 0.8585, 'training_time': 113.51, 'precision': 0.8596, 'recall': 0.8585, 'f1_score': 0.8584},
    'Naive Bayes': {'accuracy': 0.8555, 'training_time': 1.27, 'precision': 0.8562, 'recall': 0.8555, 'f1_score': 0.8554},
    'Logistic Regression': {'accuracy': 0.8393, 'training_time': 82.68, 'precision': 0.8393, 'recall': 0.8393, 'f1_score': 0.8392},
    'Random Forest': {'accuracy': 0.8297, 'training_time': 30.06, 'precision': 0.8327, 'recall': 0.8297, 'f1_score': 0.8294},
    'Linear SVM': {'accuracy': 0.8045, 'training_time': 39.60, 'precision': 0.8045, 'recall': 0.8045, 'f1_score': 0.8045},
}

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_data()

@st.cache_resource
def check_dataset_exists():
    return os.path.exists(DATASET_PATH)

class ImprovedMovieReviewClassifier:
    def __init__(self, sample_size=20000):
        self.sample_size = sample_size
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 3),
            stop_words='english',
            sublinear_tf=True
        )

        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42,
            alpha=0.001,
            solver='adam',
            learning_rate_init=0.001,
            early_stopping=False,  
            batch_size='auto',
            shuffle=True,
            tol=1e-4,
            warm_start=False
        )

        self.scaler = StandardScaler()
        self.accuracy = 0
        self.is_trained = False

    def preprocess_text(self, text):
        if pd.isna(text) or text == '':
            return ''

        text = str(text).lower()

        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "it's": "it is",
            "that's": "that is", "what's": "what is", "there's": "there is"
        }

        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        text = re.sub(r'<[^>]+>', '', text)

        text = re.sub(r'[^a-zA-Z\s]', '', text)

        tokens = word_tokenize(text)

        negation_words = {"not", "no", "never", "none", "nobody", "nothing", 
                         "neither", "nowhere", "hardly", "scarcely", "barely"}

        processed_tokens = []
        negate = False

        for i, token in enumerate(tokens):
            if token in negation_words:
                negate = True
                processed_tokens.append(token)
            elif negate and i < len(tokens):
                processed_tokens.append(f"NOT_{token}")
                negate = False
            elif token not in self.stop_words or len(token) <= 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                if len(lemmatized) > 2:
                    processed_tokens.append(lemmatized)

        return ' '.join(processed_tokens)

    def clean_and_validate_data(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        nan_mask = np.isnan(X).any(axis=1)
        inf_mask = np.isinf(X).any(axis=1)
        invalid_mask = nan_mask | inf_mask

        if invalid_mask.any():
            st.warning(f"Removing {invalid_mask.sum()} samples with invalid values")
            X = X[~invalid_mask]
            y = y[~invalid_mask]

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X = np.clip(X, -1e10, 1e10)

        return X, y

    def prepare_data(self, sample_size=None):
        if sample_size:
            self.sample_size = sample_size

        st.info("Loading IMDB dataset...")

        try:
            df = pd.read_csv(DATASET_PATH)
        except FileNotFoundError:
            st.error(f"Dataset file '{DATASET_PATH}' not found.")
            return None, None, None, None
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None, None, None, None

        if 'review' not in df.columns or 'sentiment' not in df.columns:
            st.error("Dataset must contain 'review' and 'sentiment' columns")
            return None, None, None, None

        st.success(f"Dataset loaded successfully! Total reviews: {len(df):,}")

        original_size = len(df)
        df = df.drop_duplicates(subset=['review'])
        df = df.dropna(subset=['review', 'sentiment'])
        df = df[df['review'].str.len() > 0]  

        if len(df) < original_size:
            st.info(f"Removed {original_size - len(df):,} invalid reviews")

        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
            st.info(f"Using sample of {self.sample_size:,} reviews for analysis")
        else:
            st.info(f"Using all {len(df):,} reviews for analysis")

        sentiment_counts = df['sentiment'].value_counts()
        st.write("**Sentiment Distribution:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Positive Reviews", sentiment_counts.get('positive', 0))
        with col2:
            st.metric("Negative Reviews", sentiment_counts.get('negative', 0))

        processed_reviews = []
        batch_size = 1000

        st.info("Processing reviews...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(0, len(df), batch_size):
            batch = df['review'].iloc[i:i+batch_size]
            processed_batch = []

            for review in batch:
                processed = self.preprocess_text(review)
                if processed.strip():  
                    processed_batch.append(processed)
                else:
                    processed_batch.append("empty review")  

            processed_reviews.extend(processed_batch)

            progress = min(i + batch_size, len(df)) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"Processed {min(i+batch_size, len(df)):,} reviews")

        progress_bar.empty()
        status_text.empty()
        st.success("Text preprocessing completed!")

        df['processed_review'] = processed_reviews

        df = df[df['processed_review'].str.len() > 5]  

        if len(df) == 0:
            st.error("No valid reviews after preprocessing!")
            return None, None, None, None

        st.info(f"Final dataset size after preprocessing: {len(df):,}")

        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_review'], 
            df['sentiment'], 
            test_size=0.2, 
            random_state=42,
            stratify=df['sentiment']
        )

        st.info("Vectorising text...")
        try:
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)

            st.info("Converting to dense arrays...")
            X_train_dense = X_train_vec.toarray()
            X_test_dense = X_test_vec.toarray()

            st.info("Applying feature scaling...")
            X_train_scaled = self.scaler.fit_transform(X_train_dense)
            X_test_scaled = self.scaler.transform(X_test_dense)

            st.info("Validating data for neural network...")
            X_train_clean, y_train_clean = self.clean_and_validate_data(X_train_scaled, y_train)
            X_test_clean, y_test_clean = self.clean_and_validate_data(X_test_scaled, y_test)

            st.success("Data preparation completed!")
            return X_train_clean, X_test_clean, y_train_clean, y_test_clean

        except Exception as e:
            st.error(f"Error in data preparation: {str(e)}")
            return None, None, None, None

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        st.info("Training Neural Network model...")

        if X_train is None or len(X_train) == 0:
            st.error("No training data available!")
            return None

        st.info(f"Training on {len(X_train):,} samples with {X_train.shape[1]:,} features")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            start_time = time()

            status_text.text("Initialising neural network...")
            progress_bar.progress(0.1)

            status_text.text("Training neural network (this may take a few minutes)...")
            progress_bar.progress(0.3)

            self.model.fit(X_train, y_train)

            progress_bar.progress(0.8)
            status_text.text("Making predictions...")

            training_time = time() - start_time

            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)

            progress_bar.progress(1.0)
            status_text.text("Training completed!")

            self.accuracy = accuracy_score(y_test, y_pred)
            self.is_trained = True

            results = {
                'accuracy': self.accuracy,
                'training_time': training_time,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'training_loss': getattr(self.model, 'loss_curve_', None)
            }

            progress_bar.empty()
            status_text.empty()

            st.success(f"Neural Network training completed! Accuracy: {self.accuracy:.1%}")

            if hasattr(self.model, 'n_iter_'):
                st.info(f"Training converged after {self.model.n_iter_} iterations")

            return results

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error during neural network training: {str(e)}")

            st.warning("Trying with simplified neural network configuration...")

            try:

                simple_model = MLPClassifier(
                    hidden_layer_sizes=(50,),  
                    max_iter=500,
                    random_state=42,
                    alpha=0.01,
                    solver='lbfgs',  
                    early_stopping=False
                )

                start_time = time()
                simple_model.fit(X_train, y_train)
                training_time = time() - start_time

                y_pred = simple_model.predict(X_test)
                y_pred_proba = simple_model.predict_proba(X_test)

                self.model = simple_model  
                self.accuracy = accuracy_score(y_test, y_pred)
                self.is_trained = True

                results = {
                    'accuracy': self.accuracy,
                    'training_time': training_time,
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'training_loss': getattr(self.model, 'loss_curve_', None)
                }

                st.success(f"Simplified Neural Network training completed! Accuracy: {self.accuracy:.1%}")
                return results

            except Exception as e2:
                st.error(f"Both neural network configurations failed: {str(e2)}")
                return None

    def predict(self, text):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        processed_text = self.preprocess_text(text)
        if not processed_text or processed_text.strip() == "":
            return "neutral", 0.5

        try:
            text_vec = self.vectorizer.transform([processed_text])
            text_vec_dense = text_vec.toarray()
            text_vec_scaled = self.scaler.transform(text_vec_dense)

            text_vec_clean = np.nan_to_num(text_vec_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            text_vec_clean = np.clip(text_vec_clean, -1e10, 1e10)
            text_vec_clean = text_vec_clean.astype(np.float32)

            prediction = self.model.predict(text_vec_clean)[0]
            probabilities = self.model.predict_proba(text_vec_clean)[0]
            confidence = max(probabilities)

            return prediction, confidence

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return "unknown", 0.5

def plot_confusion_matrix(cm, labels):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        showscale=True
    ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        autosize=True
    )

    return fig

def plot_training_loss(training_results):
    if 'training_loss' in training_results and training_results['training_loss'] is not None:
        loss_curve = training_results['training_loss']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(loss_curve) + 1)),
            y=loss_curve,
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))

        fig.update_layout(
            title="Neural Network Training Loss Curve",
            xaxis_title="Iteration",
            yaxis_title="Loss",
            width=600,
            height=400,
            showlegend=False
        )

        return fig
    return None
def plot_architecture():
    fig = go.Figure()

    layers = [
        {"name": "Input Layer", "y": 1.0, "color": "lightblue", "border": "blue", "text": "(15,000 features)"},
        {"name": "Hidden Layer 1", "y": 0.75, "color": "lightgreen", "border": "green", "text": "(100 neurons)"},
        {"name": "Hidden Layer 2", "y": 0.5, "color": "lightyellow", "border": "orange", "text": "(50 neurons)"},
        {"name": "Output Layer", "y": 0.25, "color": "lightpink", "border": "red", "text": "(2 classes)"}
    ]

    box_half_height = 0.07
    arrow_padding = 0.02

    for i, layer in enumerate(layers):
        fig.add_shape(
            type="rect",
            x0=0.3, x1=0.7,
            y0=layer["y"] - box_half_height,
            y1=layer["y"] + box_half_height,
            line=dict(color=layer["border"], width=2),
            fillcolor=layer["color"]
        )


        fig.add_annotation(
            x=0.5, y=layer["y"],
            text=f"<b>{layer['name']}</b><br>{layer['text']}",
            showarrow=False,
            font=dict(size=14, color='black'),
            align="center"
        )

        if i < len(layers) - 1:
            arrow_tail_y = layer["y"] - box_half_height - arrow_padding
            arrow_head_y = layers[i+1]["y"] + box_half_height + arrow_padding

            fig.add_annotation(
                x=0.5, y=arrow_head_y,   
                ax=0.5, ay=arrow_tail_y, 
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="gray"
            )

    fig.update_layout(
        title="Neural Network Architecture",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1.2]),
        width=600,
        height=600
    )

    return fig

def create_model_comparison_chart():
    models = list(MODEL_COMPARISON_RESULTS.keys())
    accuracies = [MODEL_COMPARISON_RESULTS[model]['accuracy'] for model in models]
    training_times = [MODEL_COMPARISON_RESULTS[model]['training_time'] for model in models]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy Comparison', 'Training Time Comparison', 
                      'Accuracy vs Training Time', 'Performance Summary'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )

    fig.add_trace(
        go.Bar(x=models, y=accuracies, name='Accuracy', 
               marker_color='lightblue', showlegend=False),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=models, y=training_times, name='Training Time', 
               marker_color='lightcoral', showlegend=False),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=training_times, y=accuracies, mode='markers+text',
                  text=models, textposition="top center",
                  marker=dict(size=12, color='green'),
                  name='Models', showlegend=False),
        row=2, col=1
    )

    f1_scores = [MODEL_COMPARISON_RESULTS[model]['f1_score'] for model in models]
    fig.add_trace(
        go.Bar(x=models, y=f1_scores, name='F1-Score', 
               marker_color='lightgreen', showlegend=False),
        row=2, col=2
    )

    fig.update_layout(
        title_text="Comprehensive Model Performance Analysis",
        height=800,
        showlegend=False
    )

    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Training Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Training Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    fig.update_yaxes(title_text="F1-Score", row=2, col=2)

    return fig

def main():
    st.title("🎬 Movie Review Sentiment Analyser")
    st.markdown("*Powered by Neural Networks - The Best Performing Model*")
    st.markdown("---")

    if not check_dataset_exists():
        st.error(f"❌ Dataset file '{DATASET_PATH}' not found!")
        st.info("Please ensure the IMDB_Dataset.csv file is in the same directory as this script.")
        st.stop()
    else:
        st.success(f"✅ Dataset '{DATASET_PATH}' found!")

    st.sidebar.title("⚙️ Configuration")

    sample_size = st.sidebar.slider(
        "Sample Size", 
        min_value=1000, 
        max_value=50000, 
        value=10000,  
        step=1000,
        help="Number of reviews to use for training (smaller sizes train faster)"
    )

    with st.sidebar.expander("🧠 Model Information"):
        st.markdown("""
        **Neural Network Architecture**
        - Input Layer: 15,000 features
        - Hidden Layer 1: 100 neurons (or 50 if simplified)
        - Hidden Layer 2: 50 neurons (if using full model)
        - Output Layer: 2 classes
        - Activation: ReLU
        - Optimiser: Adam (or L-BFGS for simplified)
        - Expected Accuracy: ~85%
        """)

    if 'classifier' not in st.session_state:
        st.session_state.classifier = ImprovedMovieReviewClassifier(sample_size)

    if st.session_state.classifier.sample_size != sample_size:
        st.session_state.classifier = ImprovedMovieReviewClassifier(sample_size)
        if 'model_trained' in st.session_state:
            del st.session_state.model_trained

    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Model Training", "🔍 Prediction", "📈 Model Comparison", "ℹ️ About"])

    with tab1:
        st.header("Neural Network Training & Evaluation")
        st.info("🧠 Training the best performing model from our comparison study")

        if st.button("🚀 Train Neural Network", type="primary", use_container_width=True):
            try:

                X_train, X_test, y_train, y_test = st.session_state.classifier.prepare_data(sample_size)

                if X_train is not None and len(X_train) > 0:
                    results = st.session_state.classifier.train_and_evaluate(
                        X_train, X_test, y_train, y_test
                    )

                    if results is not None:

                        st.session_state.training_results = results
                        st.session_state.model_trained = True

                        st.markdown("---")

                        st.subheader("📈 Neural Network Performance")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Accuracy", f"{results['accuracy']:.1%}")
                        with col2:
                            st.metric("Training Time", f"{results['training_time']:.2f}s")
                        with col3:
                            precision = results['classification_report']['weighted avg']['precision']
                            st.metric("Precision", f"{precision:.1%}")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Confusion Matrix")
                            cm_fig = plot_confusion_matrix(
                                results['confusion_matrix'], 
                                ['Negative', 'Positive']
                            )
                            st.plotly_chart(cm_fig, use_container_width=True)

                        with col2:
                            st.subheader("Model Architecture")
                            arch_fig = plot_architecture()
                            st.plotly_chart(arch_fig, use_container_width=True)

                        loss_fig = plot_training_loss(results)
                        if loss_fig:
                            st.subheader("Training Progress")
                            st.plotly_chart(loss_fig, use_container_width=True)

                    else:
                        st.error("Training failed. Please try with a smaller sample size.")
                else:
                    st.error("No training data available. Please check your dataset.")

            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                st.info("💡 Try reducing the sample size or check your dataset format.")

    with tab2:
        st.header("Sentiment Prediction")
        st.info("🧠 Powered by Neural Network - High Accuracy Predictions")

        if hasattr(st.session_state, 'model_trained') and st.session_state.model_trained:

            st.subheader("🎬 Single Review Analysis")

            example_reviews = {
                "Select an example...": "",
                "Positive Example": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. Highly recommended!",
                "Negative Example": "This was one of the worst movies I've ever seen. The plot was confusing and the acting was terrible. Complete waste of time.",
                "Mixed Example": "The movie had great visuals but the story was not very compelling. Some good moments but overall disappointing."
            }

            selected_example = st.selectbox("Try an example:", list(example_reviews.keys()))

            review_text = st.text_area(
                "Enter a movie review:",
                value=example_reviews[selected_example],
                height=150,
                placeholder="Type your movie review here..."
            )

            if st.button("🔍 Analyse Sentiment", use_container_width=True) and review_text.strip():
                try:
                    with st.spinner("Neural network is analysing sentiment..."):
                        sentiment, confidence = st.session_state.classifier.predict(review_text)

                    if sentiment != "unknown":

                        col1, col2 = st.columns(2)
                        with col1:
                            color = "green" if sentiment == "positive" else "red"
                            st.markdown(f"**Sentiment:** <span style='color:{color}; font-size: 24px;'>{sentiment.upper()}</span>", 
                                      unsafe_allow_html=True)
                        with col2:
                            st.metric("Confidence", f"{confidence:.1%}")

                        color = "darkgreen" if sentiment == "positive" else "darkred"
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = confidence * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Neural Network Confidence (%)"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Unable to analyse this review. Please try a different one.")

                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
        else:
            st.warning("⚠️ Please train the neural network first in the 'Model Training' tab.")

    with tab3:
        st.header("Model Comparison Results")
        st.info("📊 Comprehensive comparison study - Neural Network selected as the best performer")

        comparison_df = pd.DataFrame.from_dict(MODEL_COMPARISON_RESULTS, orient='index')
        comparison_df = comparison_df.sort_values('accuracy', ascending=False)
        comparison_df.index.name = 'Model'

        st.success("🏆 **Neural Network** selected for this application based on highest accuracy (85.9%)")

        st.subheader("🏆 Performance Rankings")

        comparison_df_display = comparison_df.copy()
        comparison_df_display['Rank'] = range(1, len(comparison_df_display) + 1)
        comparison_df_display = comparison_df_display[['Rank'] + [col for col in comparison_df_display.columns if col != 'Rank']]

        formatted_df = comparison_df_display.copy()
        formatted_df['accuracy'] = formatted_df['accuracy'].apply(lambda x: f"{x:.1%}")
        formatted_df['precision'] = formatted_df['precision'].apply(lambda x: f"{x:.1%}")
        formatted_df['recall'] = formatted_df['recall'].apply(lambda x: f"{x:.1%}")
        formatted_df['f1_score'] = formatted_df['f1_score'].apply(lambda x: f"{x:.1%}")
        formatted_df['training_time'] = formatted_df['training_time'].apply(lambda x: f"{x:.1f}s")

        formatted_df.columns = ['Rank', 'Accuracy', 'Training Time', 'Precision', 'Recall', 'F1-Score']

        st.dataframe(formatted_df, use_container_width=True)

        st.subheader("📊 Performance Visualisation")
        comparison_fig = create_model_comparison_chart()
        st.plotly_chart(comparison_fig, use_container_width=True)

    with tab4:
      st.header("About This Project")
      
      st.markdown("""
      ## 🎯 Project Overview
      
      This Movie Review Sentiment Analyser is the result of a comprehensive machine learning study that compared multiple algorithms to identify the best-performing model for sentiment classification of movie reviews.
      
      ## 🔬 Research Methodology
      
      ### Phase 1: Model Comparison Study
      We conducted an extensive comparison of 5 different machine learning algorithms using the IMDB Dataset of 50K movie reviews:
      
      **Models Evaluated:**
      - **Neural Network (MLPClassifier)** - Multi-layer perceptron with optimised architecture
      - **Naive Bayes** - Probabilistic classifier optimised for text data
      - **Logistic Regression** - Linear classifier with balanced class weights
      - **Random Forest** - Ensemble method with 100 decision trees
      - **Linear SVM** - Support Vector Machine with linear kernel
      
      ### Phase 2: Performance Analysis
      Each model was trained and evaluated on 20,000 carefully preprocessed reviews using:
      - **80/20 train-test split** with stratified sampling
      - **Advanced text preprocessing** including negation handling and lemmatisation
      - **TF-IDF vectorisation** with up to 15,000 features and n-grams (1,3)
      - **Comprehensive metrics** including accuracy, precision, recall, and F1-score
      
      ### Phase 3: Model Selection
      Based on rigorous testing, the **Neural Network** emerged as the clear winner:
      """)
      
      # Create a comparison table
      results_df = pd.DataFrame({
          'Model': ['Neural Network', 'Naive Bayes', 'Logistic Regression', 'Random Forest', 'Linear SVM'],
          'Accuracy': ['85.85%', '85.55%', '83.93%', '82.97%', '80.45%'],
          'Training Time': ['113.51s', '1.27s', '82.68s', '30.06s', '39.60s'],
          'Status': ['✅ Selected', '❌', '❌', '❌', '❌']
      })
      
      st.dataframe(results_df, use_container_width=True, hide_index=True)
      
      st.markdown("""
      ## 🧠 Neural Network Architecture
      
      The selected model uses a sophisticated multi-layer architecture:
      
      - **Input Layer**: 15,000 TF-IDF features from preprocessed text
      - **Hidden Layer 1**: 100 neurons with ReLU activation
      - **Hidden Layer 2**: 50 neurons with ReLU activation  
      - **Output Layer**: 2 neurons (positive/negative) with softmax activation
      - **Optimiser**: Adam with learning rate 0.001
      - **Regularisation**: L2 with alpha=0.001
      
      ## 🛠️ Advanced Features
      
      ### Text Preprocessing Pipeline
      - **Contraction expansion** (e.g., "won't" → "will not")
      - **HTML tag removal** and special character cleaning
      - **Negation-aware tokenisation** (preserves "not good" as "NOT_good")
      - **Smart stop word removal** with negation preservation
      - **WordNet lemmatisation** for word normalisation
      - **Batch processing** for memory efficiency
      
      ### Model Robustness
      - **Data validation** with NaN/infinity handling
      - **Feature scaling** using StandardScaler
      - **Fallback system** with simplified architecture if training fails
      - **Real-time training progress** monitoring
      - **Comprehensive error handling** for production deployment
      
      ## 📊 Performance Highlights
      
      Our Neural Network achieves:
      - **85.85% accuracy** - highest among all tested models
      - **85.96% precision** - excellent positive prediction reliability
      - **85.85% recall** - comprehensive sentiment detection
      - **85.84% F1-score** - optimal balance of precision and recall
      
      ## 🔧 Technical Stack
      
      - **Frontend**: Streamlit for interactive web interface
      - **Machine Learning**: Scikit-learn MLPClassifier (Neural Network)
      - **NLP Processing**: NLTK for advanced text preprocessing
      - **Data Handling**: Pandas & NumPy with robust validation
      - **Visualization**: Plotly for interactive charts and metrics
      - **Dataset**: IMDB 50K Movie Reviews (Kaggle)
      
      ## 🎯 Why Neural Network?
      
      While Naive Bayes was faster (1.27s vs 113.51s), the Neural Network's superior accuracy (85.85% vs 85.55%) and robust handling of complex text patterns made it the optimal choice for this application. The slightly longer training time is justified by the significantly better performance and more reliable predictions.
      
      ---
      
      *This project demonstrates the importance of systematic model comparison in machine learning. By testing multiple algorithms and selecting the best performer, we ensure users receive the most accurate sentiment predictions possible.*
      """)

if __name__ == "__main__":
    main()