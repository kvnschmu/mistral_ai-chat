import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

# API-Key aus Umgebungsvariablen (Streamlit Secrets)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Liste der verfügbaren Modelle
models = {
    "Mistral-large-latest": "mistral-large-latest",
    "Mistral-small-latest": "mistral-small-latest",
    "Pixtral": "pixtral-12b-2409",  # Beispiel für Pixtral-Modell
    "Mistral Nemo": "open-mistral-nemo",  # Beispiel für Mistral Nemo
    "Codestral Mamba": "open-codestral-mamba"  # Beispiel für Codestral Mamba
}

# App-Titel
st.title("🤖 Mistral AI Chat")
st.caption("Web-App mit LangChain und Mistral")

# Checkboxen für die Modell-Auswahl
selected_models = []
for model_name, model_id in models.items():
    if st.checkbox(model_name, key=model_name):  # Checkbox für jedes Modell
        selected_models.append(model_id)

if not selected_models:
    st.warning("Bitte wähle mindestens ein Modell aus.")
else:
    # Initialisierung mit Session State
    if "conversation" not in st.session_state:
        # Standardmodell verwenden, falls mehrere Modelle ausgewählt sind
        llm = ChatMistralAI(model=selected_models[0], api_key=MISTRAL_API_KEY)  # Erstes Modell in der Auswahl
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
