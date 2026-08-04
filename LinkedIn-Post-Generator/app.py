import streamlit as st
import uuid
from agents import PostRequest, app as graph_app
from langgraph.types import Command

st.set_page_config(page_title="LinkedIn Post Generator", layout="wide")
st.title("LinkedIn Post Generator")

if "phase" not in st.session_state:
    st.session_state.phase = "input"
if "config" not in st.session_state:
    st.session_state.config = None
if "request" not in st.session_state:
    st.session_state.request = None
if "post" not in st.session_state:
    st.session_state.post = None
if "error" not in st.session_state:
    st.session_state.error = None

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Search** - researches latest AI/tech news\n"
        "2. **Generate** - writes a post from the research\n"
        "3. **Review** - you see the result and accept or request changes\n"
        "4. **Refine** - if changes requested, regenerates with feedback"
    )
    if st.session_state.config:
        tid = st.session_state.config["configurable"]["thread_id"]
        st.caption(f"Thread: `{tid[:8]}...`")

# --- Input ---
if st.session_state.phase == "input":
    st.subheader("Create a new post")

    col1, col2 = st.columns(2)
    with col1:
        post_type = st.selectbox(
            "Post type",
            ["update", "meme", "explainer"],
            help="update = news summary, meme = humorous take, explainer = educational",
        )
    with col2:
        topic = st.text_input("Topic (optional)", placeholder="trending topic if empty")

    personal_angle = st.text_input("Personal angle (optional)", placeholder="your unique take")

    if st.button("Generate Post", type="primary"):
        st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        st.session_state.request = PostRequest(
            post_type=post_type,
            topic=topic if topic else None,
            personal_angle=personal_angle if personal_angle else None,
        )
        st.session_state.phase = "generating"
        st.rerun()

# --- Generating ---
if st.session_state.phase == "generating":
    status = st.status("Running workflow...", expanded=True)

    try:
        initial_state = {
            "request": st.session_state.request,
            "search_result": None,
            "post": None,
            "feedback": None,
        }
        for chunk in graph_app.stream(
            initial_state, config=st.session_state.config, stream_mode="updates"
        ):
            for node_name, node_output in chunk.items():
                if node_name == "search":
                    if node_output is None:
                        status.write("Skipping search (explainer with known topic)")
                    else:
                        sr = node_output.get("search_result", {})
                        ev = sr.event if hasattr(sr, "event") else str(sr)[:80]
                        status.write(f"Researched: {ev}...")
                elif node_name == "generate":
                    raw = node_output.get("post", "")
                    if isinstance(raw, list):
                        raw = raw[0].get("text", "")
                    st.session_state.post = raw
                    status.write(f"Post generated ({len(raw)} chars)")
                elif node_name == "review":
                    status.write("Waiting for your review...")
        st.session_state.phase = "review"
        st.rerun()
    except Exception as e:
        st.session_state.error = str(e)
        st.session_state.phase = "error"
        st.rerun()

# --- Review ---
if st.session_state.phase == "review":
    st.subheader("Generated post")
    st.markdown(st.session_state.post)
    st.divider()

    st.subheader("Review")
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Accept", type="primary"):
            try:
                for _ in graph_app.stream(
                    Command(resume="y"),
                    config=st.session_state.config,
                    stream_mode="updates",
                ):
                    pass
                st.session_state.phase = "done"
                st.rerun()
            except Exception as e:
                st.session_state.error = str(e)
                st.session_state.phase = "error"
                st.rerun()

    with col2:
        feedback = st.text_input("Or suggest changes", placeholder="e.g. make it more technical, shorter...")
        if st.button("Regenerate"):
            if feedback.strip():
                try:
                    for chunk in graph_app.stream(
                        Command(resume=feedback.strip()),
                        config=st.session_state.config,
                        stream_mode="updates",
                    ):
                        for node_name, node_output in chunk.items():
                            if node_name == "generate":
                                raw = node_output.get("post", "")
                                if isinstance(raw, list):
                                    raw = raw[0].get("text", "")
                                st.session_state.post = raw
                    st.rerun()
                except Exception as e:
                    st.session_state.error = str(e)
                    st.session_state.phase = "error"
                    st.rerun()
            else:
                st.warning("Enter feedback or click Accept.")

# --- Done ---
if st.session_state.phase == "done":
    st.subheader("Final post")
    st.markdown(st.session_state.post)
    st.divider()
    st.info("Copy the post above and share on LinkedIn!")
    if st.button("Create another post"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# --- Error ---
if st.session_state.phase == "error":
    st.error(f"Error: {st.session_state.error}")
    if st.button("Try again"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
