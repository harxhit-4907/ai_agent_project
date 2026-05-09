import asyncio
import json
import threading
import uuid
from pathlib import Path
from typing import Literal, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver 

import agent as ag
from agent import AgentState
from config import OUTPUT_DIR

app = FastAPI(title="Newsletter Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
runs: Dict[str, Dict[str, Any]] = {}

class RunRequest(BaseModel):
    goal: str
    mode: Literal["autonomous", "hitl"] = "autonomous"

class FeedbackRequest(BaseModel):
    run_id: str
    feedback: str = ""

def _push_log(loop, queue, message, log_type="step"):
    asyncio.run_coroutine_threadsafe(
        queue.put({"type": log_type, "message": message}), loop
    )

def _wrap_node(original_fn, label, loop, queue):
    def wrapper(state):
        _push_log(loop, queue, f"Running: {label}…")
        result = original_fn(state)
        _push_log(loop, queue, f"✅ {label} complete")
        return result
    return wrapper

def _hitl_node_factory(run_id, loop, queue):
    def hitl_web(state):
        _push_log(loop, queue, "Human review required. Waiting for feedback…")
        run_data = runs[run_id]
        
        ok = run_data["feedback_event"].wait(timeout=60 * 60 * 24 * 7)
        
        feedback = run_data["feedback_value"].strip() if ok else ""
        if ok:
            _push_log(loop, queue, "Feedback received. Resuming agent…")
            
        return {**state, "human_feedback": feedback}
    return hitl_web

async def _run_agent_async(run_id: str, req: RunRequest):
    loop = asyncio.get_running_loop()
    run_data = runs[run_id]
    queue: asyncio.Queue = run_data["log_queue"]

    def sync_run():
        original_nodes = {}
        node_names = ["planner", "researcher", "summarizer", "writer", "critic", "hitl", "revisor", "sender"]

        try:
            # Save originals
            for name in node_names:
                original_nodes[name] = getattr(ag, name)

            # Patch nodes for frontend streaming
            for name in node_names:
                if name == "hitl" and req.mode == "hitl":
                    setattr(ag, name, _hitl_node_factory(run_id, loop, queue))
                else:
                    original = original_nodes[name]
                    setattr(ag, name, _wrap_node(original, name.title(), loop, queue))

            # THE FIX: Always use a checkpointer and pass the thread_id
            checkpointer = MemorySaver()
            graph = ag.build_graph(checkpointer=checkpointer)

            initial_state: AgentState = {
                "goal": req.goal,
                "mode": req.mode,
                "plan": "",
                "raw_articles": [],
                "summaries": [],
                "newsletter_draft": "",
                "critique": "",
                "newsletter_final": "",
                "send_result": "",
                "human_feedback": "",
                "step_log": [],
            }

            # THE FIX: This prevents the checkpointer crash
            config = {"configurable": {"thread_id": run_id}}
            final_state = graph.invoke(initial_state, config=config)
            return final_state

        finally:
            # Always restore original functions to prevent state contamination
            for name, original in original_nodes.items():
                setattr(ag, name, original)

    try:
        final_state = await asyncio.to_thread(sync_run)
        run_data["state"] = final_state
        run_data["done"] = True
        await queue.put({"type": "done", "message": "Newsletter agent finished successfully."})

    except Exception as e:
        run_data["error"] = str(e)
        run_data["done"] = True
        await queue.put({"type": "error", "message": str(e)})

@app.post("/run")
async def start_run(req: RunRequest):
    run_id = str(uuid.uuid4())[:8]
    log_queue: asyncio.Queue = asyncio.Queue()

    runs[run_id] = {
        "state": None,
        "log_queue": log_queue,
        "done": False,
        "error": None,
        "feedback_event": threading.Event(),
        "feedback_value": "",
    }

    asyncio.create_task(_run_agent_async(run_id, req))
    return {"run_id": run_id}

@app.get("/stream/{run_id}")
async def stream_run(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_gen():
        q: asyncio.Queue = runs[run_id]["log_queue"]
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                yield "data: {\"type\": \"ping\"}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    if req.run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")

    runs[req.run_id]["feedback_value"] = req.feedback
    runs[req.run_id]["feedback_event"].set()
    return {"status": "feedback received"}

@app.get("/output/{run_id}")
async def get_output(run_id: str):
    run = runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404)
    if not run["done"]:
        raise HTTPException(status_code=202)
    if run["error"]:
        raise HTTPException(status_code=500, detail=run["error"])

    state = run["state"] or {}
    return {
        "newsletter_html": state.get("newsletter_final", ""),
        "plan":            state.get("plan", ""),
        "critique":        state.get("critique", ""),
        "send_result":     state.get("send_result", ""),
        "step_log":        state.get("step_log", []),
    }

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")

app.mount("/static", StaticFiles(directory="frontend"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)