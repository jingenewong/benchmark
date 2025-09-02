from serve import run_model_prompt_queries
import random, numpy as np
from db import get_all_results, reset_db, save_result
import asyncio

def array2string(array):
    return '\n'.join(['\t'.join(map(str, row)) for row in array])

E_pnts = 51
k_pnts = 101

E_min_0 = 2; E_max_0 = 3
k_min_0 = -1; k_max_0 = 1
tau_0 = 3
noise_0 = 0.3

E_min_1 = 5; E_max_1 = 7
k_min_1 = -1; k_max_1 = 1
tau_1 = 1
noise_1 = 0.2

dE_0 = (E_max_0 - E_min_0)/(E_pnts - 1)
dk_0 = (k_max_0 - k_min_0)/(k_pnts - 1)
Ex_0 = 2.5

dE_1 = (E_max_1 - E_min_1)/(E_pnts - 1)
dk_1 = (k_max_1 - k_min_1)/(k_pnts - 1)
Ex_1 = 5.0

E_array_0 = np.linspace(E_min_0, E_max_0, E_pnts, endpoint = True)
k_array_0 = np.linspace(k_min_0, k_max_0, k_pnts, endpoint = True)

E_array_1 = np.linspace(E_min_1, E_max_1, E_pnts, endpoint = True)
k_array_1 = np.linspace(k_min_1, k_max_1, k_pnts, endpoint = True)

spectrum_array_0 = np.zeros((E_pnts, k_pnts))
spectrum_array_1 = np.zeros((E_pnts, k_pnts))

def Ek_0(k, Ex):
    E = k + Ex
    return E

def Ek_1(k, Ex):
    E = 3*k + Ex
    return E

def Ak_0(E, k, tau, Ex):
    A = 1/(1 + ((E - Ek_0(k, Ex))**2)*((2*tau)**2))
    return A

def Ak_1(E, k, tau, Ex):
    A = 1/(1 + ((E - Ek_1(k, Ex))**2)*((2*tau)**2))
    return A

k_dispersion_array_0 = np.zeros(E_pnts)
k_dispersion_array_1 = np.zeros(E_pnts)
k_trace_array_1 = np.zeros(E_pnts)

for num in range(E_pnts):
    k_dispersion_array_0[num] = E_array_0[num] - Ex_0
    k_dispersion_array_1[num] = (E_array_1[num] - Ex_1)/3

for E_num in range(E_pnts):
    k_trace_array_1[E_num] = (E_array_1[E_num] - Ex_1)/3
    for k_num in range(k_pnts):
        spectrum_array_0[E_num,k_num] = Ak_0(E_array_0[E_num], k_array_0[k_num], tau_0, Ex_0) + random.random()*noise_0
        spectrum_array_1[E_num,k_num] = Ak_1(E_array_1[E_num], k_array_1[k_num], tau_1, Ex_1) + random.random()*noise_1

temp_0 = np.array(E_array_0)[:, np.newaxis]
temp_1 = np.array(E_array_1)[:, np.newaxis]

data_0 = array2string(np.concatenate((np.round(temp_0, decimals = 3), np.round(spectrum_array_0, decimals = 3)), axis = 1))
data_1 = array2string(np.concatenate((np.round(temp_1, decimals = 3), np.round(spectrum_array_1, decimals = 3)), axis = 1))

content = "Dataset A:\nEnergy (eV) \t Intensity \n" + "Spectrum:\t" + '\t'.join(map(str, k_array_1)) + "\n" + data_1 + "Dataset B:\nEnergy (eV) \t Intensity \n" + "Spectrum:\t" + '\t'.join(map(str, k_array_0)) + "\n" + data_0

prompt = 'You are given two ARPES datasets in the form of tables.'  \
            'Their bandstructure is the set of momenta corresponding to the maximum spectral intensity at each energy. Here is an example. '\
            'Read Dataset A. Its bandstructure is given by the array: [' + ','.join(str(x) for x in k_trace_array_1) + f']. '\
            'Now read Dataset B. What are the momenta of the bandstructure for each energy? '\
            'Do not print any files read, or code written. Return only an array of {E_pnts} numbers}, with no text.'

message = [f"{content}\n\n{prompt}"]
num_elems = [E_pnts] 
message = [f"return [2 for _ in range(3)] as array of three numbers.]"]
num_elems = [3]     

title = ["test"]

models = [
    # "o1-low",
    # "o1-medium",
    # "o1-high",
    # "o3-low",
    # "o3-medium",
    # "o3-high",
    # "o3-mini-low",
    # "o3-mini-medium",
    # "o3-mini-high",
    # "o4-mini-low",
    # "o4-mini-medium",
    # "o4-mini-high",
    # "o1-mini",
    # "deepseek-reasoner",
    # "gemini-2.0-flash",
    # "gemini-1.5-flash",
    # "gemini-2.5-pro-exp-03-25",
    # "gemini-2.0-flash-lite-preview-02-05",
    # "claude-3-7-sonnet-20250219"
]


# num_repeats = 1

# cache = asyncio.run(
#         run_model_prompt_queries(
#             models=models,
#             prompts=message,
#             titles=title,
#             num_elems=num_elems,
#             num_repeats=[[1]]
#         )
#     )


# cache = asyncio.run(
#         run_model_prompt_queries(
#             models=models,
#             prompts=message,
#             titles=titles,
#             num_elems=num_elems,
#             num_repeats=1
#         )
#     )
# reset_db()
    # now we know it’s done:


# print(get_all_results("test", models))
# for model in models:
#     get_results("ARPES Bandstructure", model, save_file="model.jsonl")


