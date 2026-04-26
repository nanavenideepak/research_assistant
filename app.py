import os
import streamlit as st
from dotenv import load_dotenv

# Configure the Streamlit page — must be the first Streamlit command
st.set_page_config(page_title="Research Assistant Agent", layout="wide")

# Load environment variables from .env file (local dev)
load_dotenv(override=True)

# Support Streamlit Cloud secrets: copy them into env vars so agent.py picks them up
try:
    for key in ("GROQ_API_KEY", "TAVILY_API_KEY"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

# Validate required API keys before proceeding
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it to your .env file or Streamlit secrets.")
    st.stop()
if not os.getenv("TAVILY_API_KEY"):
    st.error("TAVILY_API_KEY not found. Add it to your .env file or Streamlit secrets.")
    st.stop()

from agent import decompose_question, search_web, decision_checkpoint, synthesize_report, AgentState

# Header section
st.title("Research Assistant Agent")
st.caption("Autonomous investigator for any topic.")

# Sidebar placeholder — will be populated after a run completes
st.sidebar.header("Session Info")

# Main input area
topic = st.text_input(
    "Enter your research topic",
    
)
run_button = st.button("Start Research", type="primary")

if run_button:
    if not topic:
        st.warning("Please enter a research topic")
        st.stop()

    try:
        # Initialize a fresh agent state
        state: AgentState = {
            "topic": topic,
            "sub_questions": [],
            "search_results": {},
            "coverage_sufficient": True,
            "followup_query": None,
            "report": ""
        }

        # STATUS BLOCK 1 — Decompose the research question
        with st.status("Breaking down research question...", expanded=True) as s1:
            state = decompose_question(state)
            st.write("Sub-questions identified:")
            for q in state["sub_questions"]:
                st.markdown(f"- {q}")
            s1.update(label="Question decomposed", state="complete", expanded=False)

        # STATUS BLOCK 2 — Search the web for each sub-question
        with st.status("Searching the web...", expanded=True) as s2:
            state = search_web(state)
            st.write(f"Searched {len(state['sub_questions'])} topics")
            for q in state["sub_questions"]:
                st.markdown(f"- Searched: _{q}_")
            s2.update(label=f"Web search complete — {len(state['search_results'])} results gathered", state="complete", expanded=False)

        # STATUS BLOCK 3 — Evaluate coverage sufficiency
        with st.status("Evaluating coverage...", expanded=True) as s3:
            state = decision_checkpoint(state)
            if state["coverage_sufficient"]:
                st.success("Coverage sufficient — proceeding to synthesis")
            else:
                st.warning(f"Gap detected — running follow-up search: '{state['followup_query']}'")
                st.write("Follow-up search complete")
            st.caption("Decision reason recorded in state")
            s3.update(label="Coverage evaluation complete", state="complete", expanded=False)

        # STATUS BLOCK 4 — Generate the final report
        with st.status("Generating report...", expanded=True) as s4:
            state = synthesize_report(state)
            s4.update(label="Report generated", state="complete", expanded=False)

        # Display sidebar metrics after successful run
        st.sidebar.metric("Searches Run", len(state["search_results"]))
        st.sidebar.metric("Sub-questions", len(state["sub_questions"]))

        st.sidebar.write("Coverage sufficient:", state["coverage_sufficient"])

        # Display the final report
        st.divider()
        st.subheader("Research Report")
        st.markdown(state["report"])
        st.download_button(
            label="Download Report (.md)",
            data=state["report"],
            file_name=f"research_{topic[:30].replace(' ', '_')}.md",
            mime="text/markdown"
        )

    except Exception as e:
        st.error(f"Agent error: {str(e)}")
        st.stop()
