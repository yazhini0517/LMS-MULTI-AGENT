"""
main.py  —  Streamlit UI Entry Point  (Ollama Version)
=======================================================
Opens a browser UI on http://localhost:8501

Exactly follows PDF workflow:
  User Input → Writer Agent (RAG) → Reviewer Agent →
  Refinement → Final Tutorial

NO FastAPI. NO OpenAI. Pure Ollama + Streamlit.

Run:
    streamlit run main.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------
# Import service (compiles LangGraph)
# -----------------------------------------------------------------------
from app.services.tutorial_service import TutorialService


# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="LMS Multi-Agent Tutorial Generator",
    page_icon="📘",
    layout="wide"
)


# -----------------------------------------------------------------------
# Initialize service once using session state
# -----------------------------------------------------------------------
@st.cache_resource
def get_service():
    return TutorialService()


service = get_service()


# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.title("📘 LMS Multi-Agent Tutorial Generator")
st.markdown(
    f"""
    Powered by **Ollama** (`{os.getenv('OLLAMA_MODEL', 'llama3.2')}`)  
    RAG Server: `{os.getenv('RAG_SERVER_URL', 'http://localhost:8002')}`

    **Workflow:** Writer Agent (with RAG) → Reviewer Agent → Refinement → Final Tutorial
    """
)
st.divider()


# -----------------------------------------------------------------------
# Input Form — matches PDF Step 10.3 exactly
# -----------------------------------------------------------------------
st.subheader("📝 Tutorial Request")

col1, col2 = st.columns(2)

with col1:
    topic = st.text_input(
        label="Tutorial Topic",
        placeholder="e.g. Introduction to Python Functions",
        help="Enter the topic you want a tutorial generated on"
    )

    target_audience = st.text_input(
        label="Target Audience",
        value="students",
        placeholder="e.g. beginner programmers, developers, students",
        help="Who will be reading this tutorial?"
    )

with col2:
    difficulty_level = st.selectbox(
        label="Difficulty Level",
        options=["beginner", "intermediate", "advanced"],
        index=1,
        help="Choose the difficulty level for the tutorial"
    )

    max_iterations = st.slider(
        label="Max Refinement Iterations",
        min_value=1,
        max_value=5,
        value=2,
        help="How many times should the Writer refine based on Reviewer feedback?"
    )

st.markdown(
    f"""
    > **Selected:** Topic=`{topic or '(not set)'}` | 
    Audience=`{target_audience}` | 
    Difficulty=`{difficulty_level}` | 
    Max Iterations=`{max_iterations}`
    """
)

st.divider()


# -----------------------------------------------------------------------
# Generate Button
# -----------------------------------------------------------------------
generate_btn = st.button(
    "🚀 Generate Tutorial",
    type="primary",
    disabled=not topic.strip(),
    use_container_width=True
)

if not topic.strip():
    st.warning("⚠ Please enter a tutorial topic to proceed.")


# -----------------------------------------------------------------------
# Run the multi-agent workflow on button click
# -----------------------------------------------------------------------
if generate_btn and topic.strip():

    # Show progress
    with st.spinner(
        f"⏳ Generating tutorial on '{topic}'... "
        f"(Writer Agent → Reviewer Agent → Refinement)"
    ):
        result = service.generate_tutorial(
            topic            = topic.strip(),
            target_audience  = target_audience.strip() or "students",
            difficulty_level = difficulty_level,
            max_iterations   = max_iterations
        )

    # ---- Display result ----
    st.divider()

    if result.get("error") and not result.get("tutorial"):
        st.error(f"❌ Error: {result['error']}")

    else:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Approved",    str(result.get("is_approved", False)))
        col2.metric("🔄 Iterations",  str(result.get("iterations", 0)))
        col3.metric("📚 RAG Queries", str(len(result.get("rag_queries", []))))

        st.divider()

        # ---- Final Tutorial ----
        st.subheader("📘 Final Tutorial")
        st.markdown(result.get("tutorial", "No tutorial generated."))

        # ---- Download button ----
        st.download_button(
            label="⬇ Download Tutorial as Markdown",
            data=f"# {topic}\n\n{result.get('tutorial', '')}",
            file_name=f"tutorial_{topic.replace(' ', '_').lower()}.md",
            mime="text/markdown"
        )

        st.divider()

        # ---- Expandable details ----
        with st.expander("📝 View Initial Draft"):
            st.markdown(result.get("initial_draft", "None"))

        with st.expander("🔄 View Refined Draft"):
            refined = result.get("refined_draft", "")
            st.markdown(refined if refined else "No refinement was needed.")

        with st.expander("🔍 View Reviewer Feedback"):
            st.markdown(result.get("reviewer_feedback", "None"))

        with st.expander("📚 View RAG Context Used"):
            st.text(result.get("retrieved_context", "No RAG context retrieved."))

        with st.expander("🔎 View RAG Queries Sent"):
            queries = result.get("rag_queries", [])
            if queries:
                for q in queries:
                    st.code(q)
            else:
                st.text("No queries sent.")


# -----------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------
st.divider()
st.caption(
    "🤖 Multi-Agent Tutorial Generator | "
    f"Model: {os.getenv('OLLAMA_MODEL', 'llama3.2')} | "
    "Built with LangGraph + Ollama + Streamlit"
)