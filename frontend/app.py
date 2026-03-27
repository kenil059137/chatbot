import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="CHARUSAT Support AI", page_icon="🤖", layout="centered")

st.title("CHARUSAT Support AI 🤖")
st.caption("Ask questions about student support, eligibility criteria, and policies.")

# Initialize session state for session_id and messages
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the CHARUSAT student support assistant. How can I help you today?", "confidence": None}
    ]

# Sidebar for session management
with st.sidebar:
    st.header("Session Management")
    st.write(f"**Session ID:** `{st.session_state.session_id}`")
    
    if st.button("New Conversation", type="primary"):
        st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = [
            {"role": "assistant", "content": "New session started. How can I help you?", "confidence": None}
        ]
        st.rerun()
        
    st.divider()
    st.markdown("### Status")
    st.markdown("🟢 Backend System Online")

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("confidence") is not None:
            conf_pct = round(msg["confidence"] * 100)
            if conf_pct >= 60:
                color = "green"
            elif conf_pct >= 30:
                color = "orange"
            else:
                color = "red"
            st.caption(f":{color}[Confidence: {conf_pct}%]")

# Accept user input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "confidence": None})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    API_URL, 
                    json={"question": prompt, "session_id": st.session_state.session_id},
                    timeout=30
                )
                
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "Sorry, I received an empty response.")
                confidence = data.get("confidence")
                
                message_placeholder.markdown(answer)
                
                if confidence is not None:
                    conf_pct = round(confidence * 100)
                    if conf_pct >= 60:
                        color = "green"
                    elif conf_pct >= 30:
                        color = "orange"
                    else:
                        color = "red"
                    st.caption(f":{color}[Confidence: {conf_pct}%]")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "confidence": confidence
                })
            else:
                error_msg = f"Error: Received status code {response.status_code}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "confidence": None})
                
        except requests.exceptions.RequestException as e:
            error_msg = "Could not connect to the backend. Ensure FastAPI is running on `http://127.0.0.1:8000`."
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "confidence": None})
