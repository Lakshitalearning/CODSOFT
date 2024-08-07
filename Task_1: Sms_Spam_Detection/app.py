
import streamlit as st
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources if not already available
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the stemmer
ps = PorterStemmer()

# Background image style
page_bg_img = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');

[data-testid="stAppViewContainer"] {
    background-image: url("https://plus.unsplash.com/premium_photo-1682310098267-b89c6601aab8?q=80&w=1824&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D.jpg");
    background-size: cover;
    position: absolute;
    width: 100%;
    height: 100%;
    background-position: center;
}
[data-testid="stHeader"], [data-testid="stSidebar"] {
    background-color: rgba(0,0,0,0.5) !important;
}
[data-testid="stAppViewContainer"] > div:first-child {
    padding-top: 5rem;
}
h1 {
    font-family: 'Pacifico', cursive;
    font-size: 4rem; /* Increased font size */
    color: black;
    text-align: center;
    margin-bottom: 2rem;
    margin-top: -2rem; /* Shift title up */
    font-weight: bold; /* Make title bold */
    position: relative;
    text-shadow: 0 0 10px rgba(255,255,255,0.8), 0 0 20px rgba(255,255,255,0.6), 0 0 30px rgba(255,255,255,0.4); /* Glowing effect */
}
h1::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: -10px; /* Adjust this value as needed */
    width: 100%;
    height: 5px; /* Thickness of the underline */
    background: linear-gradient(135deg, rgba(255,255,255,0.6), rgba(255,255,255,0.8));
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}
.result-box {
    font-family: 'Arial', sans-serif;
    font-size: 2rem;
    text-align: center;
    padding: 1.5rem;
    margin: 0 auto;
    width: 250px;
    height: 250px;
    line-height: 250px;
    border-radius: 50%;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    transition: transform 0.3s, box-shadow 0.3s;
}
.result-box:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 16px rgba(0,0,0,0.4);
}
.spam {
    background: linear-gradient(135deg, rgba(255,0,0,0.8), rgba(200,0,0,0.8));
    color: white;
    border: 2px solid rgba(255,0,0,0.5);
}
.not-spam {
    background: linear-gradient(135deg, rgba(0,0,0,0.8), rgba(0,0,0,0.6));
    color: white;
    border: 2px solid rgba(0,0,0,0.5);
}
.sidebar-content {
    color: white;
    font-size: 1.1rem;
    line-height: 1.5;
}
.sidebar-title {
    font-size: 1.5rem;
    font-weight: bold;
    color: white;
}
.sidebar-arrow {
    color: white;
    font-size: 2.5rem; /* Increased size of the arrow */
    font-weight: bold; /* Make the arrow bold */
    margin-bottom: 1rem;
}
.expander-header {
    font-size: 1.5rem;
    font-weight: bold;
}
.expander-content {
    font-size: 1.1rem;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.markdown("<div class='sidebar-arrow'>→</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-title'>Spam Detection Guide</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-content'>Spam Messages: These are unsolicited or unwanted messages often used for advertising or phishing. They can include:<br>- Promotional content<br>- Links to suspicious websites<br>- Requests for personal information<br><br>Not Spam Messages: These are legitimate and expected messages. They typically:<br>- Are from known contacts<br>- Contain relevant and expected content<br>- Do not ask for sensitive information<br><br>How to Recognize Spam:<br>- Check the Sender: Be cautious of unknown or suspicious email addresses.<br>- Look for Red Flags: Unusual requests, urgent language, or poor grammar.<br>- Verify Links: Hover over links to see if they lead to reputable sites.<br>- Use a Spam Filter: Advanced tools can help identify and filter out spam messages.</div>", unsafe_allow_html=True)

# Preprocess function
def transform_text(text):
    text = text.lower()  # Convert to lower case
    text = nltk.word_tokenize(text)  # Tokenize words
    y = [i for i in text if i.isalnum()]  # Remove punctuation
    y = [i for i in y if i not in stopwords.words('english')]  # Remove stopwords
    y = [ps.stem(i) for i in y]  # Apply stemming
    return " ".join(y)

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer (2).pkl', 'rb'))
model = pickle.load(open('trained_model.pkl', 'rb'))

# Streamlit UI
st.markdown("<h1>Email Spam Classifier</h1>", unsafe_allow_html=True)
input_sms = st.text_input("Enter the message", placeholder="Type your message here...")

# Pop-up feature using expander
with st.expander("How to Use This App", expanded=False):
    st.markdown("<div class='expander-header'>How to Use This App</div>", unsafe_allow_html=True)
    st.markdown("<div class='expander-content'>- Enter your message in the text box.<br>- Click the 'Predict' button to analyze the message.<br>- The app will classify the message as either 'Spam' or 'Not Spam'.</div>", unsafe_allow_html=True)

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)  # Preprocess
    vector_input = tfidf.transform([transformed_sms])  # Vectorize
    result = model.predict(vector_input)[0]  # Predict
    
    # Display result with enhanced shapes
    if result == 1:
        st.markdown("<div class='result-box spam'>Spam</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box not-spam'>Not Spam</div>", unsafe_allow_html=True)
