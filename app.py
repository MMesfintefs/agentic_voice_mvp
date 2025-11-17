import streamlit as st
from openai import OpenAI

client = OpenAI()
st.write("OpenAI SDK version:", openai.__version__)
st.title("Ultra-Simple Agentic Assistant")

user_input = st.text_input("Say something:")

def agentic_reasoning(query):
    prompt = f"""
    You are an agentic reasoning engine.
    For the user's message: '{query}'
    1. Identify the intent.
    2. List the steps required.
    3. Perform the steps.
    4. Produce a final answer.

    Respond in JSON with:
    - intent
    - steps
    - answer
    """

    response = client.responses.create(
        model="gpt-5.1",
        input=prompt,
        reasoning={"effort": "medium"}
    )

    return response.output_text

if user_input:
    result = agentic_reasoning(user_input)
    st.write(result)
