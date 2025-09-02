import requests
import time

BASE_URL = "https://b5b74dacaecf.ngrok-free.app"

def run(models, prompts, titles, num_elems, num_repeats=1):
    payload = {
        "models": models,
        "prompts": prompts,
        "titles": titles,
        "num_elems": num_elems,
        "num_repeats": num_repeats
    }
    r = requests.post(f"{BASE_URL}/run", json=payload)
    return r.json()

def status():
    r = requests.get(f"{BASE_URL}/status")
    return r.json()

def cancel():
    r = requests.post(f"{BASE_URL}/cancel")
    return r.json()

