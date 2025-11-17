# =========================
# Agentic Voice Assistant – Full App (Text + Upload Voice + Auto-Mic Mode)
# =========================

import os
import time
import tempfile
from typing import Any, Dict, List

import streamlit as st
from openai import OpenAI
from duckduckgo_search import DDGS

import numpy as np
import av
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# =========================
# CLIENT SETUP
# =========================

def get_openai_client() -> OpenAI:
    api_key = None

    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI_API_KEY not found in secrets.")
        st.stop()

    return OpenAI(api_key=api_key)


client = get_openai_client()


# =========================
# TOOLS
# =========================

def web_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {"title": r.get("title"), "href": r.get("href"), "body": r.get("body")}
            )
    return results


# =========================
# AUDIO HELPERS
# =========================

def transcribe_audio_file(uploaded_file) -> str:
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=uploaded_file,
    )
    return getattr(transcript, "text", str(transcript))


def synthesize_speech(text: str) -> bytes:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp.name
    tmp.close()

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    )
    response.stream_to_file(tmp_path)

    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()

    return audio_bytes


# =========================
# AGENT (PERCEIVE → REASON → ACT)
# =========================

class VoiceAgent:
    def __init__(self):
        self.memory: List[Dict[str, Any]] = []

    def perceive(self, text: str) -> Dict[str, Any]:
        lowered = text.lower()

        if any(w in lowered for w in ["summarize", "summary"]):
            intent = "summarize"
        elif any(w in lowered for w in ["calculate", "compute", "math"]):
            intent = "calculate"
        elif any(w in lowered for w in ["search", "google", "web"]):
            intent = "web_search"
        else:
            intent = "chat"

        return {
            "intent": intent,
            "entities": {"text": text},
            "sentiment": "neutral",
            "raw_text": text,
        }

    def reason(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        intent = perception["intent"]
        text = perception["entities"]["text"]

        if intent == "summarize":
            goal = "Provide a concise summary."
            steps = ["Find main idea", "Extract key points", "Summarize"]
        elif intent == "calculate":
            goal = "Perform numerical reasoning."
            steps = ["Identify numbers", "Compute", "Explain result"]
        elif intent == "web_search":
            goal = "Search the web."
            steps = ["Generate query", "Search", "Summarize output"]
        else:
            goal = "Help with natural conversation."
            steps = ["Understand request", "Use context", "Reply"]

        return {
            "goal": goal,
            "plan": {"steps": steps, "intent": intent, "original_text": text},
            "confidence": 0.8,
        }

    def act(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        intent = reasoning["plan"]["intent"]
        user_text = reasoning["plan"]["original_text"]

        tools_output = None
        if intent == "web_search":
            tools_output = web_search(user_text)

        system_prompt = (
            "You are an agentic assistant. Use the perception and reasoning objects "
            "to form a helpful response."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "system", "content": f"Perception & reasoning: {reasoning}"},
        ]

        if tools_output:
            messages.append({"role": "system", "content": f"Tool results: {tools_output}"})

        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )

        reply = completion.choices[0].message.content

        result = {"reply": reply, "tools_output": tools_output}

        self.memory.append(
            {
                "user": user_text,
                "reasoning": reasoning,
                "result": result,
            }
        )

        return result


# =========================
# SESSION STATE
# =========================

def init_state():
    if "agent" not in st.session_state:
        st.session_state.agent = VoiceAgent()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "voice_buffer" not in st.session_state:
        st.session_state.voice_buffer = []
    if "last_audio_time" not in st.session_state:
        st.session_state.last_audio_time = time.time()
    if "processing" not in st.session_state:
        st.session_state.processing = False


# =========================
# MAIN APP
# =========================

def main():
    st.set_page_config(page_title="Agentic Voice Assistant", page_icon="🤖")
    st.title("🤖 Agentic Voice Assistant")

    init_state()
    agent = st.session_state.agent

    st.markdown("Text chat, upload voice, and full auto-microphone mode.")

    # -------------------------
    # TEXT CHAT
    # -------------------------
    st.subheader("💬 Text Mode")

    with st.form("chat_form", clear_on_submit=True):
        text_in = st.text_area("Message:", height=100)
        send = st.form_submit_button("Send")

    if send and text_in.strip():
        perception = agent.perceive(text_in)
        reasoning = agent.reason(perception)
        result = agent.act(reasoning)

        st.session_state.history.append({
            "mode": "text",
            "user": text_in,
            "perception": perception,
            "reasoning": reasoning,
            "result": result,
        })

        st.write(result["reply"])

    # -------------------------
    # UPLOAD AUDIO MODE
    # -------------------------
    st.subheader("🎤 Upload Voice Clip")

    audio_file = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a", "webm"])

    if st.button("Transcribe & Respond (Upload)") and audio_file:
        transcript = transcribe_audio_file(audio_file)
        st.write("**You said:**", transcript)

        perception = agent.perceive(transcript)
        reasoning = agent.reason(perception)
        result = agent.act(reasoning)

        st.write("**Assistant:**", result["reply"])

        reply_audio = synthesize_speech(result["reply"])
        st.audio(reply_audio, format="audio/mp3")

    # -------------------------
    # AUTO VOICE MODE
    # -------------------------
    st.subheader("🎙️ Auto Voice Mode (Hands-Free)")

    st.markdown("Speak → stop → auto transcribe → auto reply → auto TTS.")

    def audio_frame_callback(frame):
        audio = frame.to_ndarray()
        rms = np.sqrt(np.mean(audio ** 2))

        if rms > 50:
            st.session_state.voice_buffer.append(
                audio.flatten().astype(np.int16).tobytes()
            )
            st.session_state.last_audio_time = time.time()

        return av.AudioFrame.from_ndarray(audio, layout="mono")

    webrtc_ctx = webrtc_streamer(
        key="auto-mic",
        mode=WebRtcMode.SENDRECV,
        audio_frame_callback=audio_frame_callback,
        media_stream_constraints={"audio": True, "video": False},
    )

    SILENCE_TIMEOUT = 1.5

    if webrtc_ctx and not st.session_state.processing:
        if (
            st.session_state.voice_buffer
            and time.time() - st.session_state.last_audio_time > SILENCE_TIMEOUT
        ):
            st.session_state.processing = True
            try:
                st.write("🧩 Processing speech...")

                raw = b"".join(st.session_state.voice_buffer)
                st.session_state.voice_buffer = []

                tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                with open(tmp_wav, "wb") as f:
                    f.write(raw)

                sound = AudioSegment.from_file(tmp_wav, format="wav")
                fixed_path = tmp_wav.replace(".wav", "_fixed.wav")
                sound.export(fixed_path, format="wav")

                with st.spinner("📝 Transcribing..."):
                    with open(fixed_path, "rb") as af:
                        transcript = transcribe_audio_file(af)

                st.write("**You said:**", transcript)

                with st.spinner("🤖 Thinking..."):
                    perception = agent.perceive(transcript)
                    reasoning = agent.reason(perception)
                    result = agent.act(reasoning)

                st.write("**Assistant:**", result["reply"])

                with st.spinner("🗣️ Speaking..."):
                    reply_audio = synthesize_speech(result["reply"])

                st.audio(reply_audio, format="audio/mp3")

            finally:
                st.session_state.processing = False

    # -------------------------
    # HISTORY
    # -------------------------
    st.subheader("Conversation Log")

    for i, turn in enumerate(reversed(st.session_state.history), start=1):
        st.markdown(f"### Turn {i} ({turn['mode']})")
        st.write("**You:**", turn["user"])
        st.write("**Assistant:**", turn["result"]["reply"])
        with st.expander("Internals"):
            st.json({
                "perception": turn["perception"],
                "reasoning": turn["reasoning"],
                "tools_output": turn["result"]["tools_output"],
            })


if __name__ == "__main__":
    main()
