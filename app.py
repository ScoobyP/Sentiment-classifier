import streamlit as st
import pickle
from nltk.stem.porter import PorterStemmer
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import contractions
import joblib

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

# Initialize stemmer
ps = PorterStemmer()
sia = SentimentIntensityAnalyzer()


# Sidebar navigation
st.sidebar.title('Projects')
app_mode = st.sidebar.radio("Select Project",
                            ( 'Sentiment Analysis'))






@st.cache_resource
def load_sentiment_model():
    pipeline = joblib.load('sentiment_pipeline10.pkl')
    return pipeline

# Spam Classifier Page

# Sentiment Analysis Page
if app_mode == 'Sentiment Analysis':
    st.title('😊 Sentiment Classifier')
    st.markdown("""
    This app analyzes text sentiment as **positive** or **negative**.
    """)

    # Load models
    sentiment_pipeline = load_sentiment_model()

    # Input area
    text_input2 = st.text_area('Enter your text here:',
                               height=150,
                               placeholder="Type your review or comment here...")


    if st.button("Predict Sentiment"):
        if not text_input2.strip():
            st.warning("Please enter some text to analyze")
        else:
            with st.spinner('Analyzing sentiment...'):
                # Preprocess and predict

                input_x = pd.DataFrame([{
                    'Summary': contractions.fix(text_input2),
                    'char_len': len(text_input2),
                    'exclam': text_input2.count('!'),
                    'quest': text_input2.count('?'),
                    'fs': text_input2.count('.'),
                    'vader_score': sia.polarity_scores(text_input2)['compound']* (4.6279433295293515 if len(text_input2.split()) <= 15 and sia.polarity_scores(text_input2)['compound'] != 0 else 1.0)
                }])

                result2 = sentiment_pipeline.predict(input_x)[0]

                # Display result
                st.subheader("Prediction Result")
                if result2 == 1:
                    st.success("😊 Positive Sentiment")
                else:
                    st.error("😞 Negative Sentiment")

                # Show confidence (if your model supports predict_proba)
                try:
                    proba2 = sentiment_pipeline.decision_function(input_x)
                    st.write(f"Confidence: {max(proba2) * 100:.1f}%")
                except:
                    pass
    with st.expander('Model Limitations'):
        st.write('The above classification model has been trained on a mix of data from - amazon reviews, flight related tweets, IMDB movie reviews and restaurant reviews.\n After extensive datacleaning, preprocessing, vectorizing, hyperparameter tuning and model selection, LogisticRegression with certain hyperparameters was  chosen to be the best ML model on this dataset.')
        st.write('Still the model may fail on inputs that: \n 1. Clearly belong to one class but seem to be of the opposite. For e.g. - "I couldn\'t be happier." \n 2. Belong to very niche domain/category, or \n 3. Words on which the model is not trained at all. For e.g. - "This product is a *game changer*" and \n 4. Double negatives like - "Not bad!" or "He’s not entirely wrong."')
# Add some styling
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


