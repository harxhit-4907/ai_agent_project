# 🗞️ Autonomous AI Newsletter Agent

An intelligent, multi-step AI workflow that autonomously researches, summarizes, self-critiques, and drafts responsive HTML newsletters. Built with **LangGraph** and **FastAPI**, and powered entirely by local LLMs via **Ollama**—meaning zero API costs and zero rate limits.

## ✨ Features
* **Multi-Step Reasoning (LangGraph):** The agent autonomously navigates a complex state graph: Planning ➔ Research ➔ Summarization ➔ Drafting ➔ Critique ➔ Revision ➔ Publishing.
* **Live Web Research:** Uses DuckDuckGo search to pull real-time news and data.
* **Human-in-the-Loop (HITL):** A built-in toggle allows human reviewers to pause the agent, review its self-critique, and inject feedback before the final HTML is generated.
* **Real-Time UI Streaming:** A vanilla HTML/JS frontend that consumes Server-Sent Events (SSE) to display live terminal logs and the final rendered newsletter.
* **100% Local AI:** Powered by Llama 3 via Ollama for complete privacy and unlimited usage.

## 🛠️ Tech Stack
* **Logic/Orchestration:** LangGraph, LangChain (`langchain-ollama`)
* **Backend:** FastAPI, Uvicorn, Python `asyncio`
* **Frontend:** HTML5, CSS3, Vanilla JavaScript
* **LLM:** Ollama (Llama 3)
* **Tools:** DuckDuckGo Search, BeautifulSoup

## 🚀 Quick Start

### 1. Prerequisites
* Python 3.9+
* [Ollama](https://ollama.com/) installed on your machine.
* Pull the Llama 3 model by running this in your terminal:
  ```bash
  ollama run llama3-latest