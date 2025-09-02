from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import asyncio

from serve import run_model_prompt_queries  # your async query runner

app = FastAPI()

# Shared state
status = {
    "is_running": False,
    "done": 0,
    "total": 0,
}

status_lock = asyncio.Lock()
lock = asyncio.Lock()
running_task: Optional[asyncio.Task] = None  # Reference to the active task

class QueryRequest(BaseModel):
    models: List[str]
    prompts: List[str]
    titles: Optional[List[str]]
    num_elems: List[int]
    num_repeats: Union[int, List[List[int]]]

@app.post("/run")
async def run_queries(req: QueryRequest):
    global running_task

    if lock.locked():
        raise HTTPException(status_code=400, detail="Server is already processing a request.")

    async def run_and_track():
        async with lock:
            status["is_running"] = True
            status["done"] = 0
            status["total"] = len(req.models) * len(req.prompts) * req.num_repeats

            async def progress_callback():
                 async with status_lock:
                    status["done"] += 1

            try:
                await run_model_prompt_queries(**req.model_dump(), progress_cb=progress_callback)
            except asyncio.CancelledError:
                print("Task was cancelled.")
                raise
            finally:
                status["is_running"] = False

    # Launch task and store reference
    running_task = asyncio.create_task(run_and_track())
    return {"status": "started"}

@app.post("/cancel")
async def cancel_query():
    global running_task

    if running_task is None or running_task.done():
        return {"status": "no task running"}

    running_task.cancel()
    try:
        await running_task
    except asyncio.CancelledError:
        pass

    return {"status": "cancelled"}

@app.get("/status")
async def get_status():
    return status
