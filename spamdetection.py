import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

def main():
    # Load the dataset
    df = pd.read_csv(r"D:\Downloads\spam.csv")

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

    st.header('Spam Detection')
    input_mess = st.text_input('Enter Message Here')

    if st.button('Validate'):
        output = predict(input_mess)
        st.markdown(output)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")


#streamlit run spamdetection.py