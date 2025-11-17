# app.py
import os
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

    # Try Streamlit secrets first (recommended for Streamlit Cloud)
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
            # Simple tool call example
            tools_output = web_search(user_text)

        # Build system prompt to expose the structure
        system_prompt = (
            "You are an agentic voice assistant. You receive a perception and reasoning "
            "object and must respond to the user. "
            "Use the provided plan and any tool results. "
            "Be concise, clear, and actionable."
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_text,
            },
            {
                "role": "system",
                "content": f"Perception object: {reasoning}",
            },
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
# STREAMLIT UI
# =========================

def init_session_state():
    if "agent" not in st.session_state:
        st.session_state.agent = VoiceAgent()
    if "history" not in st.session_state:
        st.session_state.history = []


def main():
    st.set_page_config(page_title="Agentic Assistant – Phase 1", page_icon="🤖")
    st.title("🤖 Agentic Voice Assistant (Phase 1: Text Only)")

    init_session_state()

    st.markdown(
        """
        This is the **Phase 1** scaffold:
        - Perceive → Reason → Act loop
        - Optional web search tool
        - Text-only for now (voice comes later)
        """
    )

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Type your message:", height=100)
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        agent: VoiceAgent = st.session_state.agent

        with st.spinner("Thinking like an overworked intern..."):
            perception = agent.perceive(user_input)
            reasoning = agent.reason(perception)
            result = agent.act(reasoning)

        st.session_state.history.append(
            {
                "user": user_input,
                "perception": perception,
                "reasoning": reasoning,
                "result": result,
            }
        )

    # Show history
    if st.session_state.history:
        st.subheader("Conversation & Agent Reasoning")
        for i, turn in enumerate(reversed(st.session_state.history), start=1):
            st.markdown(f"### Turn {i}")
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
