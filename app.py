import os
import tempfile
import streamlit as st
from openai import OpenAI

# setup client
def get_client():
    key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("No API key")
        st.stop()
    return OpenAI(api_key=key)

client = get_client()

# simple agent
class SimpleAgent:
    def __init__(self):
        self.memory = []  # store past chats

    # perceive step
    def perceive(self, text):
        intent = "chat"
        if "search" in text.lower():
            intent = "search"
        return {"intent": intent, "text": text}  # simple perception

    # reason step
    def reason(self, perception):
        goal = "Talk normally"
        if perception["intent"] == "search":
            goal = "Find info"
        plan = ["Understand user", "Generate answer"]
        return {"goal": goal, "plan": plan, "text": perception["text"]}

    # act step
    def act(self, reasoning):
        messages = [
            {"role": "system", "content": f"Goal: {reasoning['goal']}"},
            {"role": "user", "content": reasoning["text"]}
        ]
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )
        reply = response.choices[0].message.content
        self.memory.append({"user": reasoning["text"], "reply": reply})
        return reply

agent = SimpleAgent()

# page title
st.title("Voice AI")   # three-word comment: “Simple app header”

# text chat
st.subheader("Text Chat")  # three-word comment: “Text section header”
text_in = st.text_input("You:")  # three-word comment: “User text input”

if text_in:
    p = agent.perceive(text_in)        # three-word comment: “Detect intent basic”
    r = agent.reason(p)                # three-word comment: “Plan assistant behavior”
    reply = agent.act(r)               # three-word comment: “LLM final reply”
    st.write("Assistant:", reply)      # three-word comment: “Show text output”


# voice upload mode
st.subheader("Voice Input")  # three-word comment: “Voice section header”

audio_file = st.file_uploader("Upload audio file:", type=["mp3", "wav", "m4a"])  # upload

def transcribe(file):
    return client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=file
    ).text

def speak(text):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    )
    audio.stream_to_file(tmp)
    w
