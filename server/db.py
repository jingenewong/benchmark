import firebase_admin
from firebase_admin import credentials, db
import json
import os

cred = credentials.Certificate("db_admin.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'YOUR DATABASE URL HERE'
})

def safe_key(text):
    return text.replace('.', '_')

def append_result(title, model, result):
    key = f"{title}-{safe_key(model)}"
    data = result
    ref = db.reference(f"results/{key}")
    ref.push(data)

# save with jsonl format
def save_result(title, model, save_file=None):
    key = f"{title}-{safe_key(model)}"
    ref = db.reference(f"results/{key}")
    results = ref.get()
    
    res = list(results.values()) if results else []
    if save_file:
        with open("results.jsonl", "w") as f:
            for result in res:
                f.write(json.dumps(result) + "\n")
    
    return res

def get_all_results(title, model_list):
    return [{model: save_result(title, model, None)} for model in model_list]

def save_all_results(titles, models, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    for title in titles:
        for model in models:
            filename = f"{title}-{safe_key(model)}.jsonl"
            filepath = os.path.join(save_dir, filename)
            save_result(title, model, filepath)
        


def reset_db():
    ref = db.reference("results")
    ref.delete()

def delete_model_entries(model):
    ref = db.reference("results")
    all_entries = ref.get()
    
    if not all_entries:
        print("No entries found.")
        return
    
    target_suffix = safe_key(model)
    keys_to_delete = []
    
    for key in all_entries.keys():
        if key.endswith(f"-{target_suffix}"):
            keys_to_delete.append(key)
    
    for key in keys_to_delete:
        print(f"Deleting {key}")
        db.reference(f"results/{key}").delete()



