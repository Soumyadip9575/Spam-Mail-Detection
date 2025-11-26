# spam_app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import textwrap

st.set_page_config(page_title="Spam Mail Detection", layout="wide")

# Load dataset and train model 
DATA_PATH = "spam.csv"
try:
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
except FileNotFoundError:
    st.error(f"Dataset not found at '{DATA_PATH}'. Please place your CSV there and restart the app.")
    st.stop()


if 'Category' in df.columns and 'Message' in df.columns:
    label_col, text_col = 'Category', 'Message'
elif 'v1' in df.columns and 'v2' in df.columns:
    label_col, text_col = 'v1', 'v2'
elif 'label' in df.columns and 'text' in df.columns:
    label_col, text_col = 'label', 'text'
else:
    label_col, text_col = df.columns[0], df.columns[1]

df = df[[label_col, text_col]].dropna().copy()
df.columns = ['label', 'message']

# Map labels to binary (spam=1, ham/ham/0=0)
df['label'] = df['label'].astype(str).str.strip().str.lower()
df['label'] = df['label'].replace({'ham': 'ham', 'not spam': 'ham', 'ham\n': 'ham'})
df['label'] = df['label'].apply(lambda x: 1 if 'spam' in x else 0)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Train/test split
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Vectorize
cv = CountVectorizer(stop_words='english', ngram_range=(1,2), max_df=0.95, min_df=1)
X_train_count = cv.fit_transform(X_train.astype(str))

# Train model
model = MultinomialNB()
model.fit(X_train_count, y_train)

# Evaluate
X_test_count = cv.transform(X_test.astype(str))
accuracy = model.score(X_test_count, y_test)

# -------------------------
# Sidebar Navigation
# -------------------------
with st.sidebar:
    st.markdown('<img src="https://www.gstatic.com/images/branding/product/1x/gmail_2020q4_32dp.png" alt="Gmail Logo" width="50" style="display:block;margin:0 auto;">', unsafe_allow_html=True)
    st.markdown("## Navigate")
    page = st.selectbox("", ["Home", "Detection"])
    st.markdown(f"**Model Accuracy:** {accuracy:.2f}")

# -------------------------
# Home Page 
# -------------------------
if page == "Home":
    # Centered container to mimic the card in screenshot
    st.write("")  # spacing
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <style>
            .card {
                background: #ffffffcc;
                border-radius: 18px;
                padding: 48px 40px;
                box-shadow: 0 18px 40px rgba(0,0,0,0.15);
                text-align: center;
                margin-top: 60px;
            }
            .card h1 {
                font-size: 44px;
                margin: 0 0 10px 0;
                font-weight: 800;
                color: #0b0b0b;
                font-family: 'Montserrat', sans-serif;
            }
            .card p {
                font-size: 22px;
                color: #333333;
                margin-top: 12px;
                font-family: 'Roboto', sans-serif;
            }
            </style>
            <div class="card">
                <h1>Welcome to our Spam Mail<br>Detection Page!</h1>
                <p>Verify your mail</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------------
# Detection Page
# -------------------------
else:
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # Layout: center form
    left, center = st.columns([1, 4])
    with center:
        st.markdown('<img src="https://www.gstatic.com/images/branding/product/1x/gmail_2020q4_32dp.png" alt="Gmail Logo" width="100" style="display:block;margin:0 auto;">', unsafe_allow_html=True)
        st.markdown("<marquee scrollamount='15'><h1 style='color:#1b5e20;'>Spam Detection</h1></marquee>", unsafe_allow_html=True)

        # Input area
        input_text = st.text_area("Enter Message Here", height=120, placeholder="Type or paste the email / message text to validate...")

        # Validate button
        if st.button("Validate"):
            if not input_text or input_text.strip() == "":
                st.info("Please type a message to validate.")
            else:
                # predict
                vec = cv.transform([input_text])
                pred = model.predict(vec)[0]
                # show result
                if pred == 1:
                    st.markdown("<span style='color:red;font-size:20px;font-weight:700;'>Spam</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:blue;font-size:20px;font-weight:700;'>Not Spam</span>", unsafe_allow_html=True)





#streamlit run spam_app.py