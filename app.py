import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


# Set custom title for the browser tab
st.set_page_config(page_title="Email Classifier App | Shubam Sharma", page_icon=":envelope:")

st.title("📧 Email Classifier")
input_msg = st.text_area("Enter your Message")

# Check if text area is empty or contains only whitespace
input_empty = not input_msg.strip()

# Custom CSS for the cool button style
st.markdown(
    """
        <style>
    .stButton button {
        background-color: #66CCFF; /* Light blue */
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        color: black;

    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Disable the "Predict" button if the text area is empty or contains only whitespace
if st.button("Predict", disabled=input_empty):
    with st.spinner("Predicting..."):
        transformed_sms = transform_text(input_msg)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

    # Remove the spinner
    st.empty()

    if result == 1:
        st.success("Prediction: Spam")
    else:
        st.success("Prediction: Not Spam")
