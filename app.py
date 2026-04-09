
# --- THIS MUST BE THE FIRST THING IN YOUR SCRIPT ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END OF FIX ---
import streamlit as st
from streamlit import spinner
import time

from supporting_functions import(
    extract_video_id,
    get_transcript,
    translate_transcript,
    get_important_topics,
    generate_notes,
    create_chunks,
    create_vector_store,
    rag_answer,
    normal_chat,
    hybrid_chat
)

# ---------------- STREAMING FUNCTION ---------------- #
def stream_response(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.02)

#------Sidebar------#
with st.sidebar:
    st.title("🎬 Learnify AI")
    st.markdown("---")

    youtube_url = st.text_input("YouTube URL")
    language = st.text_input("Language", value="en")

    task_option = st.radio(
        "Choose Task",
        ["Chat with video", "Notes for you", "Chat with llm"]
    )

    submit_button = st.button("✨ Process")

st.title("YouTube Content Synthesizer")

# ---------------- PREPROCESSING ---------------- #
if submit_button and task_option != "Chat with llm":
    if youtube_url:
        video_id = extract_video_id(youtube_url)

        if video_id:
            with spinner("Fetching transcript..."):
                full_transcript = get_transcript(video_id, language)

                if language != 'en':
                    full_transcript = translate_transcript(full_transcript)

            # -------- NOTES -------- #
            if task_option == 'Notes for you':
                st.subheader("Important Topics")
                st.write(get_important_topics(full_transcript))

                st.subheader("Notes")
                st.write(generate_notes(full_transcript))

            # -------- VIDEO CHAT PREP -------- #
            if task_option == 'Chat with video':
                with st.spinner("Creating vector store..."):
                    chunks = create_chunks(full_transcript)
                    vectorstore = create_vector_store(chunks)

                    # 🔥 MULTI VIDEO SUPPORT
                    if "vector_store" not in st.session_state:
                        st.session_state.vector_store = vectorstore
                    else:
                        st.session_state.vector_store._collection.add(
                            documents=[doc.page_content for doc in chunks],
                            metadatas=[doc.metadata for doc in chunks]
                        )

                st.session_state.messages = []
                st.success("Video ready for chat!")

# ---------------- CHAT WITH VIDEO ---------------- #
if task_option == 'Chat with video' and "vector_store" in st.session_state:
    st.subheader("🎥 Chat with Video")

    mode = st.selectbox(
        "AI Mode",
        ["Auto (Smart)", "Video Only", "General Chat"]
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # show history
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    prompt = st.chat_input("Ask about the video...", key="video_input")

    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # prepare history
        history = []
        msgs = st.session_state.messages
        for i in range(0, len(msgs)-1, 2):
            if msgs[i]['role']=='user' and msgs[i+1]['role']=='assistant':
                history.append({
                    "user": msgs[i]['content'],
                    "assistant": msgs[i+1]['content']
                })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # 🔥 HYBRID SYSTEM
                if mode == "Auto (Smart)":
                    response = hybrid_chat(prompt, st.session_state.vector_store, history)
                elif mode == "Video Only":
                    response = rag_answer(prompt, st.session_state.vector_store)
                else:
                    response = normal_chat(prompt, history)

                # ⚡ STREAMING OUTPUT
                st.write_stream(stream_response(response))

        st.session_state.messages.append({'role': 'assistant', 'content': response})

# ---------------- CHAT WITH LLM ---------------- #
if task_option == 'Chat with llm':
    st.subheader("💬 Chat with LLM")

    if "llm_messages" not in st.session_state:
        st.session_state.llm_messages = []

    for msg in st.session_state.llm_messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    prompt = st.chat_input("Ask anything...", key="llm_input")

    if prompt:
        st.session_state.llm_messages.append({'role': 'user', 'content': prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # history
        history = []
        msgs = st.session_state.llm_messages
        for i in range(0, len(msgs)-1, 2):
            if msgs[i]['role']=='user' and msgs[i+1]['role']=='assistant':
                history.append({
                    "user": msgs[i]['content'],
                    "assistant": msgs[i+1]['content']
                })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = normal_chat(prompt, history)

                # ⚡ STREAMING OUTPUT
                st.write_stream(stream_response(response))

        st.session_state.llm_messages.append({'role': 'assistant', 'content': response})
