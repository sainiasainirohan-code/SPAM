import streamlit as st
import joblib

# 1. Load the saved model and vectorizer
model = joblib.load("spam.pkl")
cv = joblib.load("vectorizer.pkl")

# 2. Set up the Streamlit page header
st.set_page_config(page_title="SMS Spam Detector", page_icon="✉️", layout="centered")
st.title("✉️ SMS Spam Classification App")
st.write("Enter an SMS message below to check if it's **Ham** (Safe) or **Spam**.")

# 3. Create a text area input for the user
user_input = st.text_area("Enter your message here:", height=150, placeholder="Type your message...")

# 4. Predict button logic
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        data = cv.transform([user_input]).toarray()
        prediction = model.predict(data)[0]

        st.write("---")
        if prediction == 1:
            st.error("🚨 **Warning: This looks like a SPAM message!**")
        else:
            st.success("✅ **Safe: This looks like a HAM message.**")
