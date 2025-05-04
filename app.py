import streamlit as st
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK data (only needed once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

# Initialize stemmer
ps = PorterStemmer()
sia = SentimentIntensityAnalyzer()


# Text preprocessing function
def text_preprocessing(text):
    default_stopwords = set(stopwords.words('english'))
    words_to_keep = {'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'can', "aren't", 'couldn',
                     "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                     "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                     'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                     'wouldn', "wouldn't", 'no', 'nor', 'not', 'only', 'but', 'from', "although", "even though",
                     "however", "despite"}
    custom_stopwords = default_stopwords - words_to_keep
    # Removing special characters

    cleaned = re.sub(r'[^A-Za-z!?\s]', '', text)

    filtered1 = []
    for word in word_tokenize(cleaned):
        if word.lower() not in custom_stopwords:
            if word.isupper():
                filtered1.append(word)
            else:
                filtered1.append(word.lower())

    filtered2 = []
    for word2 in filtered1:
        if word2.isupper():
            filtered2.append(ps.stem(word2.lower()).upper())
        else:
            filtered2.append(ps.stem(word2))

    return ' '.join(filtered2)


# Sidebar navigation
st.sidebar.title('Projects')
app_mode = st.sidebar.radio("Select Project",
                            ( 'Sentiment Analysis'))


# Load models only once using caching



@st.cache_resource
def load_sentiment_model():
    pipeline = pickle.load(open('sentiment_pipeline3.pkl', 'rb'))
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
                    'Summary': text_input2,
                    'char_len': len(text_input2),
                    'exclam': text_input2.count('!'),
                    'quest': text_input2.count('?'),
                    'fs': text_input2.count('.'),
                    'vader_score': sia.polarity_scores(text_input2)['compound']
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
                    proba = sentiment_pipeline.predict_proba(input_x)[0]
                    st.write(f"Confidence: {max(proba) * 100:.1f}%")
                except:
                    pass

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


