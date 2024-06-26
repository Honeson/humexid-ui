import streamlit as st
import requests
import uuid
import os
from main3 import get_answer
# Page configuration
st.set_page_config(page_title="Trustbreed Logo", page_icon="🤖", layout="wide")

# Custom CSS for dark theme
st.markdown("""
<style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .main {
        background-color: #2B2B2B;
        padding: 2rem;
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        background-color: #3C3C3C;
        color: #E0E0E0;
        border: 1px solid #555;
        border-radius: 10px;
        padding: 10px 15px;
        font-size: 16px;
    }
    .stButton > button {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
        background-color: #4CAF50;
        border: none;
    }
    .stButton > button:hover {
        transform: scale(1.03);
        background-color: #45A049;
    }
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        width: 80%;
        word-wrap: break-word;
    }
    .chat-message.user {
        background-color: #1C3B4C;
        margin-left: auto;
    }
    .chat-message.bot {
        background-color: #3C3C3C;
        margin-right: auto;
    }
    .chat-message p {
        margin: 0;
        line-height: 1.4;
    }
    .logo-container { display: flex; justify-content: center; align-items: center; margin-bottom: 2rem; }
    .logo { max-width: 200px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5); }   
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
if 'chat_histories' not in st.session_state:
    st.session_state['chat_histories'] = [{'id': st.session_state['session_id'], 'history': []}]
if 'question_asked' not in st.session_state:
    st.session_state['question_asked'] = False

# Get the absolute path to the 'static' directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

logo_path = os.path.join(static_dir, 'cren.svg')    



# Function to get current chat history
def get_current_chat_history():
    for chat in st.session_state['chat_histories']:
        if chat['id'] == st.session_state['session_id']:
            return chat['history']
    return []

# Function to update current chat history
def update_current_chat_history(history):
    for chat in st.session_state['chat_histories']:
        if chat['id'] == st.session_state['session_id']:
            chat['history'] = history
            return
    st.session_state['chat_histories'].append({'id': st.session_state['session_id'], 'history': history})

# Sidebar for starting a new conversation and insights
with st.sidebar:
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_column_width=True)
    else:
        st.sidebar.warning("Logo not found. Please add 'logo.png' to the 'static' folder.")
    if st.button("Start New Conversation"):
        st.session_state['session_id'] = str(uuid.uuid4())
        st.session_state['chat_histories'].append({'id': st.session_state['session_id'], 'history': []})
        st.session_state['question_asked'] = False
        st.experimental_rerun()

    st.subheader("Chat Insights")
    for i, chat in enumerate(reversed(st.session_state['chat_histories'])):
        if chat['history']:
            topics = set(word for message in chat['history'] for word in message['answer'].split()[:3])
            with st.expander(f"Conversation {len(st.session_state['chat_histories']) - i}"):
                st.write(f"💡 Topics: {', '.join(topics)}")
                st.write(f"🗨️ Exchanges: {len(chat['history'])}")
                if len(chat['history']) > 5:
                    st.success("🌟 Deep Dive!")
                if chat['id'] == st.session_state['session_id']:
                    st.info("Current Chat")

# Main chat interface
st.title("🤖 Trustly")

# Display current chat history
current_history = get_current_chat_history()
for chat in current_history:
    st.markdown(f'<div class="chat-message user"><p>{chat["question"]}</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-message bot"><p>{chat["answer"]}</p></div>', unsafe_allow_html=True)

# Input for new question
question = st.text_input("Hi, Do you have any complaint about any company (First Bank, MTN NG), I'm Trustly, I can asist you!", key="question_input")

# Handle question submission
if st.button("Send") and question:
    answer = get_answer(question, st.session_state['session_id'])
    current_history.append({"question": question, "answer": answer})
    update_current_chat_history(current_history)
    st.session_state['question_asked'] = True
    st.experimental_rerun()

# Reset question_asked state when the input changes
if question != st.session_state.get('last_question', ''):
    st.session_state['question_asked'] = False
    st.session_state['last_question'] = question

# Footer
st.markdown("---")
st.write("🌐 Connected to: Trustly - Your Smart AI")
st.write("ℹ️ Pro Tips:")
st.write("- This AI has memory and can remember you previous message'.")
st.write("- Need a fresh start? Hit 'Start New Conversation'.")
st.caption("Made with ❤️ for you!")
