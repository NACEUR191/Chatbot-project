import streamlit as st
import speech_recognition as sr
import nltk
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')

# App setup
st.set_page_config(page_title="TF-IDF Chatbot", layout="centered")
st.title("🤖 Smarter Chatbot with Voice Input")
st.write("Upload a `.txt` file for the chatbot's knowledge. Then ask questions via voice or text.")

# Upload file
uploaded_file = st.file_uploader("📄 Upload a text file (.txt)", type=["txt"])

if uploaded_file:
    # Read text and tokenize
    raw_text = uploaded_file.read().decode("utf-8")
    sent_tokens = nltk.sent_tokenize(raw_text)

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sent_tokens)

    # Smart response function using cosine similarity
    def get_bot_response(user_input):
        input_vec = vectorizer.transform([user_input])
        similarities = cosine_similarity(input_vec, tfidf_matrix)
        idx = similarities.argsort()[0][-1]  # index of most similar sentence
        score = similarities[0][idx]
        if score < 0.2:  # if it's not similar enough
            return "🤔 I’m not sure how to respond to that. Try asking differently."
        return sent_tokens[idx]

    # Voice transcription
    def transcribe_speech():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now.")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                st.warning("❌ I couldn't understand you.")
            except sr.RequestError:
                st.error("⚠️ Error with the speech recognition service.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
        return ""

    # Text input
    text_input = st.text_input("💬 Type your message:")

    # Speech input
    if st.button("🎙️ Speak"):
        spoken_text = transcribe_speech()
        if spoken_text:
            st.success(f"🗣️ You said: {spoken_text}")
            response = get_bot_response(spoken_text)
            st.subheader("🤖 Chatbot says:")
            st.write(response)

    if text_input:
        response = get_bot_response(text_input)
        st.subheader("🤖 Chatbot says:")
        st.write(response)

else:
    st.warning("📂 Please upload a `.txt` file to activate the chatbot.")
