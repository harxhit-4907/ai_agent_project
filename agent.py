import operator
from typing import TypedDict, Literal, List, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from tools import web_search, summarize_articles, generate_html_newsletter, simulate_send
from config import MODEL_NAME, OUTPUT_DIR

# ───────── State ─────────
class AgentState(TypedDict):
    goal: str
    mode: Literal["autonomous", "hitl"]
    plan: str
    raw_articles: list
    summaries: list
    newsletter_draft: str
    critique: str
    newsletter_final: str
    send_result: str
    human_feedback: str
    step_log: list

# ───────── LLM ─────────
llm = ChatOllama(model=MODEL_NAME, temperature=0.3)

# ───────── Nodes ─────────
def planner(state):
    prompt = f"Create a step-by-step plan for: {state['goal']}"
    res = llm.invoke(prompt)
    return {**state, "plan": res.content, "step_log": state["step_log"] + ["[PLAN]"]}

def researcher(state):
    # Only search ONE query to save tokens
    queries = ["latest AI agents news 2026"] 
    articles = []
    for q in queries:
        articles += web_search(q)
        
    # ONLY take the top 3 articles instead of 7
    return {"raw_articles": articles[:3], "step_log": ["[RESEARCH]"]}

def summarizer(state):
    summaries = summarize_articles(state["raw_articles"], llm)
    return {**state, "summaries": summaries, "step_log": state["step_log"] + ["[SUMMARIZE]"]}

def writer(state):
    draft = generate_html_newsletter(state["summaries"], state["goal"], llm)
    return {**state, "newsletter_draft": draft, "step_log": state["step_log"] + ["[WRITE]"]}

def critic(state):
    prompt = f"Critique this newsletter:\n{state['newsletter_draft'][:2000]}"
    res = llm.invoke(prompt)
    return {**state, "critique": res.content, "step_log": state["step_log"] + ["[CRITIQUE]"]}

def hitl(state):
    # This acts as a placeholder; app.py overrides this at runtime
    return {**state}

def revisor(state):
    prompt = f"""Improve newsletter based on critique and feedback:
Feedback: {state['human_feedback']}
Critique: {state['critique']}
Draft: {state['newsletter_draft']}"""
    res = llm.invoke(prompt)
    return {**state, "newsletter_final": res.content, "step_log": state["step_log"] + ["[REVISE]"]}

def sender(state):
    result = simulate_send(state["newsletter_final"], OUTPUT_DIR)
    return {**state, "send_result": result, "step_log": state["step_log"] + ["[SEND]"]}

def router(state):
    return "hitl" if state["mode"] == "hitl" else "revisor"

# ───────── Graph ─────────
def build_graph(checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node("planner", planner)
    g.add_node("researcher", researcher)
    g.add_node("summarizer", summarizer)
    g.add_node("writer", writer)
    g.add_node("critic", critic)
    g.add_node("hitl", hitl)
    g.add_node("revisor", revisor)
    g.add_node("sender", sender)

    g.set_entry_point("planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "summarizer")
    g.add_edge("summarizer", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", router, {"hitl": "hitl", "revisor": "revisor"})
    g.add_edge("hitl", "revisor")
    g.add_edge("revisor", "sender")
    g.add_edge("sender", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())