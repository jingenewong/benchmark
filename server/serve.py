import asyncio
from tqdm import tqdm
from query import ask_model    # now an async function
from db import append_result, safe_key
import os
import json
import aiofiles


TASKS_PER_MIN = 50
DELAY_BETWEEN_TASKS = 60 / TASKS_PER_MIN  

async def run_model_prompt_queries(
    models: list,
    prompts: list,
    titles: list,
    num_elems: list,
    num_repeats,
    max_workers: int = 100000,
    progress_cb=None,
):
    prompt_to_title = dict(zip(prompts, titles))
    cache = {}

    # semaphore to cap concurrent ask_model calls
    sem = asyncio.Semaphore(max_workers)

    async def bound_ask(model, prompt, num_elem):
        async with sem:
            try:
                out = await ask_model(model, prompt, cache, num_elem)
            except Exception as e:
                print(f"Error for model={model}, prompt={prompt}: {e}")
                return None, model, prompt
            return out, model, prompt

    # prepare all (model, prompt, num_elem) triples
    if isinstance(num_repeats, list):
        all_tasks = [
            (m, p, n)
            for j, (p, n) in enumerate(zip(prompts, num_elems))  
            for i, m in enumerate(models)                        
            for _ in range(num_repeats[i][j])
        ]
    else:
        all_tasks = [
        (m, p, n)
        for p, n in zip(prompts, num_elems)
        for _ in range(num_repeats)
        for m in models
    ]

    tasks = []
    for idx, (model, prompt, num_elem) in enumerate(all_tasks):
        await asyncio.sleep(DELAY_BETWEEN_TASKS)
        tasks.append(asyncio.create_task(bound_ask(model, prompt, num_elem)))


    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
        output, model, prompt = await fut
        if output:
            title = prompt_to_title[prompt]
            
            try:
                # Firebase
                append_result(title, model, output)

                # Local disk
                save_dir = "./outputs"
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{title}-{safe_key(model)}.jsonl"
                filepath = os.path.join(save_dir, filename)

                async with aiofiles.open(filepath, mode="a") as f:
                    await f.write(json.dumps(output) + "\n")

            except Exception as e:
                print(f"Error saving output for {title} - {model}: {e}")
                
        if progress_cb:
            await progress_cb()

    return cache
