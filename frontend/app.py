import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="CHARUSAT Support AI", page_icon="🤖", layout="centered")

st.title("CHARUSAT Support AI 🤖")
st.caption("Ask questions about admissions, courses, exams, and scholarships.")

if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the CHARUSAT student support assistant. How can I help you today?", "confidence": None, "confidence_level": None}
    ]

with st.sidebar:
    st.header("Session Management")
    st.write(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("New Conversation", type="primary"):
        st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = [
            {"role": "assistant", "content": "New session started. How can I help you?", "confidence": None, "confidence_level": None}
        ]
        st.rerun()

    st.divider()
    st.markdown("### Status")
    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if health.status_code == 200:
            st.markdown("🟢 Backend Online")
        else:
            st.markdown("🔴 Backend Offline")
    except:
        st.markdown("🔴 Backend Offline")

    st.divider()
    st.markdown("### Tips")
    st.markdown("- Ask in English, Hindi or Gujarati")
    st.markdown("- Click 'New Conversation' to reset memory")
    st.markdown("- Confidence shows answer reliability")


def show_confidence(confidence, confidence_level):
    if confidence is None:
        return
    conf_pct = round(confidence * 100)
    level = confidence_level or ""
    if conf_pct >= 60:
        st.caption(f":green[Confidence: {conf_pct}% ({level})]")
    elif conf_pct >= 30:
        st.caption(f":orange[Confidence: {conf_pct}% ({level})]")
    else:
        st.caption(f":red[Confidence: {conf_pct}% ({level})]")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_confidence(msg.get("confidence"), msg.get("confidence_level"))


if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "confidence": None,
        "confidence_level": None
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    API_URL,
                    json={"question": prompt, "session_id": st.session_state.session_id},
                    timeout=60
                )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "Sorry, I received an empty response.")
                confidence = data.get("confidence")
                confidence_level = data.get("confidence_level")

                message_placeholder.markdown(answer)
                show_confidence(confidence, confidence_level)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "confidence": confidence,
                    "confidence_level": confidence_level
                })
            else:
                error_msg = f"Error: Received status code {response.status_code}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "confidence": None,
                    "confidence_level": None
                })

        except requests.exceptions.RequestException:
            error_msg = "Could not connect to backend. Ensure FastAPI is running on http://127.0.0.1:8000"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "confidence": None,
                "confidence_level": None
            })