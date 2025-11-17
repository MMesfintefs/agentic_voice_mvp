# app.py
import os
import tempfile
from typing import Any, Dict, List

import streamlit as st
from openai import OpenAI
from duckduckgo_search import DDGS

# =========================
# CONFIG & CLIENT
# =========================

def get_openai_client() -> OpenAI:
    """
    Get OpenAI client from Streamlit secrets or env.
    """
    api_key = None

    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "OPENAI_API_KEY not found. "
            "Add it to Streamlit secrets or environment variables."
        )
        st.stop()

    return OpenAI(api_key=api_key)


client = get_openai_client()

# =========================
# SIMPLE TOOL: WEB SEARCH
# =========================

def web_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Basic DuckDuckGo search as an example tool the agent can call.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {"title": r.get("title"), "href": r.get("href"), "body": r.get("body")}
            )
    return results


# =========================
# AUDIO HELPERS (PHASE 2B)
# =========================

def transcribe_audio_file(uploaded_file) -> str:
    """
    Send an uploaded audio file to OpenAI for transcription.
    Supports mp3, wav, m4a, etc. (whatever the API accepts).
    """
    # Streamlit's UploadedFile is file-like, we can pass it directly.
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=uploaded_file,
    )
    # API returns an object with .text
    return getattr(transcript, "text", str(transcript))


def synthesize_speech(text: str) -> bytes:
    """
    Turn reply text into speech using OpenAI TTS and return raw audio bytes.
    We write to a temp .mp3 then read it back for st.audio().
    """
    # Temp file path
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp.name
    tmp.close()

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    )

    # Official helper to write to file, then we read bytes
    response.stream_to_file(tmp_path)

    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()

    return audio_bytes


# =========================
# AGENT CLASS (PERCEIVE → REASON → ACT)
# =========================

class VoiceAgent:
    def __init__(self):
        # Very dumb memory list for now
        self.memory: List[Dict[str, Any]] = []

    # ---------- PERCEIVE ----------
    def perceive(self, text: str) -> Dict[str, Any]:
        lowered = text.lower()

        if any(w in lowered for w in ["summarize", "summary"]):
            intent = "summarize"
        elif any(w in lowered for w in ["calculate", "compute", "math", "number"]):
            intent = "calculate"
        elif any(w in lowered for w in ["search", "google", "web", "online"]):
            intent = "web_search"
        else:
            intent = "chat"

        perception = {
            "intent": intent,
            "entities": {"text": text},
            "sentiment": "neutral",
            "raw_text": text,
        }
        return perception

    # ---------- REASON ----------
    def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        intent = perception["intent"]
        text = perception["entities"]["text"]

        if intent == "summarize":
            goal = "Provide a clear, short summary of the user's text or topic."
            plan_steps = [
                "Identify the main topic or question.",
                "Extract key points.",
                "Condense into 2–4 concise sentences.",
            ]
        elif intent == "calculate":
            goal = "Perform a calculation or quantitative reasoning task."
            plan_steps = [
                "Identify the numerical values and relationships.",
                "Formulate the calculation.",
                "Compute the result and explain briefly.",
            ]
        elif intent == "web_search":
            goal = "Search the web for up-to-date or factual information."
            plan_steps = [
                "Turn the user's request into a concise search query.",
                "Call the web_search tool.",
                "Summarize the top results for the user.",
            ]
        else:
            goal = "Engage in helpful conversation and answer the user's question."
            plan_steps = [
                "Understand the user's question or request.",
                "Retrieve any relevant memory or context.",
                "Provide a helpful, direct answer.",
            ]

        reasoning = {
            "goal": goal,
            "plan": {"steps": plan_steps, "intent": intent, "original_text": text},
            "confidence": 0.8,  # placeholder
        }
        return reasoning

    # ---------- ACT ----------
    def act(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide whether to call tools and then call the LLM with full context.
        """
        intent = reasoning["plan"]["intent"]
        user_text = reasoning["plan"]["original_text"]

        tools_output = None

        if intent == "web_search":
            tools_output = web_search(user_text)

        system_prompt = (
            "You are an agentic voice assistant. You receive a perception and reasoning "
            "object and must respond to the user. "
            "Use the provided plan and any tool results. "
            "Be concise, clear, and actionable."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "system", "content": f"Perception & reasoning object: {reasoning}"},
        ]

        if tools_output is not None:
            messages.append(
                {
                    "role": "system",
                    "content": f"Tool results: {tools_output}",
                }
            )

        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )

        reply = completion.choices[0].message.content

        result = {
            "reply": reply,
            "tools_output": tools_output,
        }

        # Save to memory (very naive)
        self.memory.append(
            {
                "user": user_text,
                "reasoning": reasoning,
                "result": result,
            }
        )

        return result


# =========================
# STREAMLIT STATE & UI
# =========================

def init_session_state():
    if "agent" not in st.session_state:
        st.session_state.agent = VoiceAgent()
    if "history" not in st.session_state:
        st.session_state.history = []


def main():
    st.set_page_config(page_title="Agentic Voice Assistant", page_icon="🤖")
    st.title("🤖 Agentic Voice Assistant")

    init_session_state()
    agent: VoiceAgent = st.session_state.agent

    st.markdown(
        """
        This app runs an **agentic assistant** with:

        - Perceive → Reason → Act loop  
        - Optional web search tool  
        - **Text chat** and **voice via uploaded audio**  
        """
    )

    # -------------------------
    # TEXT CHAT (PHASE 1)
    # -------------------------
    st.subheader("💬 Text mode")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Type your message:", height=100)
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        with st.spinner("Thinking..."):
            perception = agent.perceive(user_input)
            reasoning = agent.reason(perception)
            result = agent.act(reasoning)

        st.session_state.history.append(
            {
                "mode": "text",
                "user": user_input,
                "perception": perception,
                "reasoning": reasoning,
                "result": result,
            }
        )

    # -------------------------
    # VOICE VIA UPLOADED AUDIO (PHASE 2B)
    # -------------------------
    st.subheader("🎤 Voice mode (upload audio file)")

    st.markdown(
        "Upload a short voice clip (mp3 / wav / m4a). "
        "I'll **transcribe → reason → reply → speak back**."
    )

    audio_file = st.file_uploader(
        "Upload audio",
        type=["mp3", "wav", "m4a", "mp4", "mpeg", "mpga", "webm"],
    )

    if st.button("Transcribe & respond (voice)"):

        if audio_file is None:
            st.warning("Upload an audio file first.")
        else:
            # 1) Transcribe
            with st.spinner("Transcribing your audio..."):
                transcript_text = transcribe_audio_file(audio_file)

            st.markdown("**Transcribed text:**")
            st.write(transcript_text)

            # 2) Run through agent pipeline
            with st.spinner("Thinking about your audio..."):
                perception_v = agent.perceive(transcript_text)
                reasoning_v = agent.reason(perception_v)
                result_v = agent.act(reasoning_v)

            reply_text = result_v["reply"]

            st.markdown("**Assistant (text reply):**")
            st.write(reply_text)

            # 3) Text-to-speech reply
            with st.spinner("Generating spoken reply..."):
                reply_audio = synthesize_speech(reply_text)

            st.markdown("**Assistant (voice reply):**")
            st.audio(reply_audio, format="audio/mp3")

            # Save in history
            st.session_state.history.append(
                {
                    "mode": "voice",
                    "user": f"[Voice] {audio_file.name}",
                    "transcript": transcript_text,
                    "perception": perception_v,
                    "reasoning": reasoning_v,
                    "result": result_v,
                }
            )

    # -------------------------
    # HISTORY & INTERNALS
    # -------------------------
    if st.session_state.history:
        st.subheader("Conversation & Agent Reasoning")

        for i, turn in enumerate(reversed(st.session_state.history), start=1):
            mode_label = "🎤 Voice" if turn.get("mode") == "voice" else "💬 Text"
            st.markdown(f"### Turn {i} {mode_label}")

            if turn.get("mode") == "voice":
                st.markdown(f"**You (audio):** {turn['user']}")
                st.markdown(f"**Transcript:** {turn.get('transcript', '')}")
            else:
                st.markdown(f"**You:** {turn['user']}")

            st.markdown(f"**Assistant:** {turn['result']['reply']}")

            with st.expander("Perception & Plan (agent internals)"):
                st.json(
                    {
                        "perception": turn["perception"],
                        "reasoning": turn["reasoning"],
                        "tools_output": turn["result"]["tools_output"],
                    }
                )


if __name__ == "__main__":
    main()
