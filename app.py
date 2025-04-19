import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

# API-Key aus Umgebungsvariablen (Streamlit Secrets)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# App-Titel
st.title("🤖 Mistral AI Chat")
st.caption("Web-App mit LangChain und Mistral")

# Initialisierung mit Session State
if "conversation" not in st.session_state:
    # LangChain Komponenten
    llm = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)
    memory = ConversationBufferMemory()
    st.session_state.conversation = ConversationChain(llm=llm, memory=memory)

# Chat-Historie anzeigen
for msg in st.session_state.get("messages", []):
    st.chat_message(msg["role"]).write(msg["content"])

# Benutzereingabe
if prompt := st.chat_input("Deine Nachricht"):
    # User-Nachricht speichern
    st.session_state.messages = st.session_state.get("messages", []) + [{"role": "user", "content": prompt}]
    st.chat_message("user").write(prompt)

    # Antwort generieren
    response = st.session_state.conversation.predict(input=prompt)
    
    # AI-Nachricht speichern
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
