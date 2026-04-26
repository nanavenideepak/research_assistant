import re
import json
from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from prompts import DECOMPOSER_PROMPT, DECISION_PROMPT, SYNTHESIZER_PROMPT

# Initialize the LLM with Groq's LLaMA 3.3 70B model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

# Initialize Tavily search tool with max 3 results per query
tavily = TavilySearchResults(max_results=3)


# Typed dictionary representing the agent's shared state across all steps
class AgentState(TypedDict):
    topic: str
    sub_questions: list[str]
    search_results: dict[str, str]
    coverage_sufficient: bool
    followup_query: str | None
    report: str


# Step 1: Break the research topic into focused sub-questions using the LLM
def decompose_question(state: AgentState) -> AgentState:
    prompt = DECOMPOSER_PROMPT.format(topic=state["topic"])
    response = llm.invoke([HumanMessage(content=prompt)])

    raw_lines = response.content.strip().split("\n")
    sub_questions = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # Keep lines that start with a digit or a dash
        if line[0].isdigit() or line.startswith("-"):
            # Strip leading numbering like "1. " "2. " or "- "
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            cleaned = re.sub(r"^-\s*", "", cleaned)
            cleaned = cleaned.strip()
            if cleaned:
                sub_questions.append(cleaned)

    # Take at most 4 sub-questions
    state["sub_questions"] = sub_questions[:4]
    return state


# Step 2: Run Tavily web search for each sub-question and collect results
def search_web(state: AgentState) -> AgentState:
    for sub_question in state["sub_questions"]:
        results = tavily.invoke(sub_question)
        concatenated = ""
        for r in results:
            title = r.get("title", "N/A")
            content = r.get("content", "N/A")
            url = r.get("url", "N/A")
            concatenated += f"Title: {title}\nContent: {content}\nURL: {url}\n\n"
        state["search_results"][sub_question] = concatenated
    return state


# Step 3: LLM evaluates if gathered info is sufficient or needs a follow-up search
def decision_checkpoint(state: AgentState) -> AgentState:
    context = "\n".join(state["search_results"].values())
    prompt = DECISION_PROMPT.format(topic=state["topic"], context=context)
    response = llm.invoke([HumanMessage(content=prompt)])

    response_text = response.content.strip()
    # Strip markdown code fences if present
    response_text = re.sub(r"```json|```", "", response_text).strip()

    try:
        parsed = json.loads(response_text)
    except Exception:
        parsed = {"sufficient": True, "reason": "Defaulting to sufficient", "followup_query": None}

    state["coverage_sufficient"] = bool(parsed.get("sufficient", True))
    state["followup_query"] = parsed.get("followup_query", None)

    # If coverage is not sufficient and a follow-up query exists, run additional search
    if not state["coverage_sufficient"] and state["followup_query"] is not None:
        followup_results = tavily.invoke(state["followup_query"])
        concatenated = ""
        for r in followup_results:
            title = r.get("title", "N/A")
            content = r.get("content", "N/A")
            url = r.get("url", "N/A")
            concatenated += f"Title: {title}\nContent: {content}\nURL: {url}\n\n"
        state["search_results"]["followup"] = concatenated

    return state


# Step 4: Synthesize all findings into a structured markdown report
def synthesize_report(state: AgentState) -> AgentState:
    context = "\n".join(state["search_results"].values())
    prompt = SYNTHESIZER_PROMPT.format(topic=state["topic"], context=context)
    response = llm.invoke([HumanMessage(content=prompt)])
    state["report"] = response.content
    return state


# Main orchestrator that runs all four steps sequentially and returns final state
def run_research_agent(topic: str) -> AgentState:
    state: AgentState = {
        "topic": topic,
        "sub_questions": [],
        "search_results": {},
        "coverage_sufficient": True,
        "followup_query": None,
        "report": ""
    }
    state = decompose_question(state)
    state = search_web(state)
    state = decision_checkpoint(state)
    state = synthesize_report(state)
    return state
