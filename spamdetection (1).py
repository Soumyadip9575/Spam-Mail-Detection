import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Home", "Detection"])

if page == "Home":
    st.markdown("""
    <style>
    .content-container {
        position: relative;
        z-index: 1;
        background: rgba(255, 255, 255, 0.8);
        max-width: 700px;
        margin: 0 auto;
        padding: 40px;
        border-radius: 25px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.3), 0 8px 20px rgba(0,0,0,0.2);
    }
    .welcome {
        font-size: 48px;
        color: black;
        text-align: center;
        font-weight: 700;
        margin-top: 100px;
        position: relative;
        z-index: 1;
    }
    .description {
        font-size: 32px;
        text-align: center;
        color: #333333;
        margin-top: 30px;
        position: relative;
        z-index: 1;
        text-shadow: none;
    }
   
    </style>
    <div class="background-div"></div>
    <div class="content-container">
        <div class="welcome">
        Welcome to our Spam Mail Detection Page!
        </div>
        <div class="description">
        Verify your mail
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Detection":
    # Load the dataset
    df = pd.read_csv("spam_fixed.csv")

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Map categories to binary values
    df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

    # Split data into messages and categories
    X = df['Message']
    y = df['Category']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature extraction
    cv = CountVectorizer(stop_words='english')
    X_train_count = cv.fit_transform(X_train)

    # Model creation and training
    model = MultinomialNB()
    model.fit(X_train_count, y_train)

    # Testing the model
    X_test_count = cv.transform(X_test)
    accuracy = model.score(X_test_count, y_test)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Example usage for predicting new messages
    def predict(message):
        message_count = cv.transform([message])
        prediction = model.predict(message_count)
        return "Spam" if prediction[0] == 1 else "Not Spam"

    st.markdown("<marquee style='color: green; font-size: 32px; font-weight: bold;'>Spam Detection</marquee>", unsafe_allow_html=True)
    input_mess = st.text_input('Enter Message Here')

    if st.button('Validate'):
        output = predict(input_mess)
        if output == "Spam":
            st.markdown(f"<span style='color: red;'>{output}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color: blue;'>{output}</span>", unsafe_allow_html=True)


#streamlit run spam_app.py