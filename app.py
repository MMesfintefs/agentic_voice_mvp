import os
import streamlit as st
from openai import OpenAI

# Setup client
def get_client():
    key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("No API key")
        st.stop()
    return OpenAI(api_key=key)

client = get_client()

# Agent class
class SimpleAgent:
    def __init__(self):
        self.memory = []

    # Perceive
    def perceive(self, text: str):
        intent = "chat"
        if "search" in text.lower():
            intent = "search"
        return {"intent": intent, "text": text}

    # Reason
    def reason(self, perception):
        if perception["intent"] == "search":
            goal = "Find information"
        else:
            goal = "Have conversation"
        plan = ["Understand input", "Generate response"]
        return {"goal": goal, "plan": plan, "perception": perception}

    # Act
    def act(self, reasoning):
        user_text = reasoning["perception"]["text"]
        # For simplicity, we do not call a search tool here
        messages = [
            {"role": "system", "content": f"Goal: {reasoning['goal']}"},
            {"role": "user", "content": user_text}
        ]
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )
        reply = completion.choices[0].message.content
        # Save memory
        self.memory.append({"user": user_text, "reply": reply})
        return reply

agent = SimpleAgent()

# Streamlit UI
st.title("Agentic Voice AI – Simple Version")

user_input = st.text_input("You:")  # three-word comment: “Input user text”

if user_input:
    # run pipeline
    perception = agent.perceive(user_input)      # three-word comment: “Detect intent text”
    reasoning = agent.reason(perception)          # three-word comment: “Plan next step”
    reply = agent.act(reasoning)                  # three-word comment: “Call LLM reply”
    st.write("Assistant:", reply)                 # three-word comment: “Show assistant reply”
