import requests
import json
import os

DATABASE_URL = 'https://physics-benchmark-default-rtdb.firebaseio.com/'

def safe_key(text):
    return text.replace('.', '_')

# --------------- Save results into local JSONL file ----------------

def save_result(title, model, save_file=None):
    key = f"{title}-{safe_key(model)}"
    url = f"{DATABASE_URL}/results/{key}.json"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch result: {response.status_code}, {response.text}")
    
    results = response.json()
    res = list(results.values()) if results else []

    if save_file:
        with open(save_file, "w") as f:
            for result in res:
                f.write(json.dumps(result) + "\n")
    
    return res

# --------------- Get all results for a list of models ----------------

def get_all_results(title, model_list):
    return [{model: save_result(title, model, None)} for model in model_list]

# --------------- Save all results into local JSONL files ----------------

def save_all_results(titles, models, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    for title in titles:
        for model in models:
            filename = f"{safe_key(model)}-{title}.jsonl"
            filepath = os.path.join(save_dir, filename)
            save_result(title, model, filepath)

# --------------- Get filename for model ----------------

def get_filename(titles, models, save_dir):
    for title in titles:
        for model in models:
            filename = f"{safe_key(model)}-{title}.jsonl"
    return filename

import os

def find_incomplete_prompts(save_dir, n):
    model_to_titles = {}

    for filename in os.listdir(save_dir):
        if not filename.endswith('.jsonl'):
            continue

        file_base = filename[:-6]  # remove '.jsonl'

        q_idx = file_base.rfind('-Q')
        if q_idx == -1:
            continue  

        model_name = file_base[:q_idx]
        title = file_base[q_idx+1:] 

        filepath = os.path.join(save_dir, filename)
        try:
            with open(filepath, 'r') as f:
                num_lines = sum(1 for _ in f)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        # Each line = one entry
        if num_lines < n:
            if model_name not in model_to_titles:
                model_to_titles[model_name] = []
            model_to_titles[model_name].append(title)
        
        for model in model_to_titles:
            model_to_titles[model].sort(key=lambda x: int(x[1:]))

    return model_to_titles

def get_attempts_left(models, titles, save_dir, n):
    attempts_left = []

    for model in models:
        model_attempts = []
        for title in titles:
            filename = f"{safe_key(model)}-{title}.jsonl"
            filepath = os.path.join(save_dir, filename)

            # Count number of attempts already saved
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        num_lines = sum(1 for _ in f)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    num_lines = 0
            else:
                num_lines = 0

            # Compute how many more are needed
            remaining = max(0, n - num_lines)
            model_attempts.append(remaining)
        
        attempts_left.append(model_attempts)

    return attempts_left

# a = get_attempts_left(
#     models=["o1-mini"],
#     titles=["Q" + str(i) for i in range(1, 136)],
#     save_dir="/home/joseph_tennyson/physics/physics-benchmark/client/Responses_small",
#     n=30
# )
