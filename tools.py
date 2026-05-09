import datetime
import re
import time  # <-- Added for rate limiting
from pathlib import Path
from typing import Any
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage 

def web_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append({
                    "title":     r.get("title", ""),
                    "url":       r.get("url", ""),
                    "snippet":   r.get("body", ""),
                    "source":    r.get("source", ""),
                    "published": r.get("date", ""),
                })
        return results
    except Exception as e:
        print(f"[web_search] Warning: {e}")
        return []

def summarize_articles(articles: list[dict], llm: Any) -> list[dict]:
    summaries = []
    for article in articles:
        prompt = f"""Summarize the following news article in exactly 2-3 sentences.
Title: {article['title']}
Source: {article['source']}
Content: {article['snippet']}"""
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            summary = response.content.strip()
        except Exception:
            summary = article["snippet"][:300]
            
        summaries.append({**article, "summary": summary})
        
        # <-- THE RATE LIMIT FIX: Wait 4 seconds to respect API limits
        print(f"Summarized '{article['title']}'. Waiting 4 seconds...")
        time.sleep(10) 
        
    return summaries

def generate_html_newsletter(summaries: list[dict], goal: str, llm: Any) -> str:
    today = datetime.date.today().strftime("%B %d, %Y")
    articles_text = "\n\n".join([
        f"[{i+1}] {s['title']}\nSource: {s['source']} | {s['published']}\nSummary: {s['summary']}\nURL: {s['url']}"
        for i, s in enumerate(summaries)
    ])

    prompt = f"""Create a complete, responsive HTML email newsletter.
Goal: {goal}
Date: {today}
Articles to include:
{articles_text}
Requirements: Inline CSS only, dark navy header, each article as a card. Subject line as an HTML comment: <!-- SUBJECT: ... -->. Return ONLY HTML."""
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Clean up markdown code blocks if the LLM includes them
    content = response.content.strip()
    if content.startswith("```html"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()

def simulate_send(newsletter_html: str, output_dir: str = "output") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/newsletter_{timestamp}.html"

    subject = "AI Agent Weekly"
    match = re.search(r"<!--\s*SUBJECT:\s*(.+?)\s*-->", newsletter_html, re.IGNORECASE)
    if match:
        subject = match.group(1).strip()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(newsletter_html)

    return f"Newsletter saved to '{filename}'. Subject: '{subject}'. Send simulated successfully."