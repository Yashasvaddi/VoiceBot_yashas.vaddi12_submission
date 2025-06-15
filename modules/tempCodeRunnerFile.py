import streamlit as st
from response_gen import listen_once, generate_response, speak_with_polly
from collections import deque

# === Streamlit Setup ===
st.set_page_config(page_title="🎤 VoiceBot - Lenden Mitra", layout="centered")

# === Custom CSS to hide everything except user input and response ===
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
        margin: auto;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
        width: 100%%;
    }
    .response-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        font-size: 16px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# === Title ===
st.title("🎧 Lenden Mitra Voice Assistant")
st.caption("Speak your question and receive a response below. Only the latest exchange is shown.")

# === State for user input and response ===
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'last_response' not in st.session_state:
    st.session_state.last_response = ""

# === Voice input interaction ===
if st.button("🗣 Speak Now"):
    with st.spinner("Listening..."):
        query = listen_once()

    if not query:
        st.warning("Could not recognize your voice. Please try again.")
        speak_with_polly("Sorry, I didn’t catch that. Please try again.")
    else:
        st.session_state.last_question = query

        if query.lower() == "exit":
            speak_with_polly("Goodbye!")
            st.stop()

        answer, source, score = generate_response(query)
        st.session_state.last_response = answer
        speak_with_polly(answer)

# === Display user input and latest response ===
if st.session_state.last_question:
    st.markdown(f"**🧑 You asked:** {st.session_state.last_question}")

if st.session_state.last_response:
    st.markdown("<div class='response-box'>", unsafe_allow_html=True)
    st.markdown(f"**🤖 Lenden Mitra:** {st.session_state.last_response}")
    st.markdown("</div>", unsafe_allow_html=True)
