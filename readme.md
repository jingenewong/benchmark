# SciPro (Scientific Process) Arena

## Installation and API usage

Before using the client:

1. Install the required packages:
   ```bash
   pip install -r client_requirements.txt
2. In `server/db.py`, write in your database URL after `databaseURL`. Likewise for the variable `DATABASE_URL` in `client/call_db.py`.
3. Ensure that the correct API site is reflected in the variable `BASE_URL` of `client/call_server.py`.

<br>
<br>

## General notes, and structure of code & data

- The benhcmark codebase has two domains: the scientific-facing side, and an API-facing side.
- Reflecting this dichotomy, the codebase is split into two main folders, `client` (which prepares benchmark questions, sends questions to the server, reads model responses, evaluates/plots the responses, and generates leaderboards), and `server` (which manages the database in which responses are kept, and how models are queried through APIs).
- Note the five scripts in the `client` folder. `call_server.py` and `call_db.py` queries the server and pulls results respectively (next two sections). `init.py` contains the bulk of ARPES-specific code and is understandable only by those well-versed in that subfield of condensed matter; contact the authors for more detail. Both `ask.py` and `eval.py` call upon `init.py` for their operation; `ask.py` asks the server questions and the database for responses, while `eval.py` scores these responses. These will be elaborated in later subsections.
- Let us now go through the various subfolders of the `client` folder. Note that `_small`, `_med`, and `_large` appended after folders refers to the resolution sizes of questions asked; these are folders for which all questions are posed to all models. `_single` refers to single questions posed to single models, while one parameters (noise or resolution) is varied. Note that the leaderboard ranks the scores of models to `small` resolution questions only, as this was the largest approxciamte set of resolutions that fit within the smallest context windows of all models tested (somewhere on the order 100k tokens once three-shot prompting has been taken into account).
- The folder `evaluation` contains summaries of all results (such as leaderboards) as well as results for individual models.
- The folder `Evaluation_plots` contains the plots of all results obtained by all models, where rhe numbering of questions runs from `Q1` to `Q135` covering all categories in order (see appendix C) with five noise levels per question. Its subfolder `Evaluation_single` collects the plots of individual questions, where `_r[...]` refers to the resolution of that dataset and `_n[...]` its noise level (in percent). Here, questions are labelled by category rather than question number.
- The folder `Evaluation_single` (not to be confused with the prior subfolder) contains plots for single questions with varying noise/resolution.
- `Plots` contain plots for all questions, listed under their respective categories, in both .txt (spectra) and .png (images of spectra) formats. Note the format of their names, which applies in general elsewhere. Examples given here will suffice. `A1a` refers to question A1, example a (for three-shot questions, examples a, b, c and are given with answers, and d as the actual question to be asked; for one-shot, only one example a is given, and b is the actual question). `_r25` refers to a resolution of 100 pixels along the horizontal axis, `_n20` to a noise level of 20%, `_k5_e3` to a momentum convolution of 0.005 inverse Angstrom and 0.003 electron-Volts (these are the same for all questions in our benchmark, but may in principle be varied). Lastly, note that questions in the D1 category have an additional parameter in their title referring to phononic coupling strength: `_l05`, `_l075`, `_l10`, `_l20`, and `_l50` refer to coupling strengths of 0.5, 0.75, 1.0, 2.0, and 5.0 respectively.
- The various folders beginning with `Prompts` contain the actual prompts given to the question (.txt files ending with `_Q`, for 'question') and their respective solutions (ending with `_S`, for 'solution'). The previous comments on resolution sizes (`_small`, `_medium`, `_large`) and single prompts (`_single`) applies.
- Likewise, folders beginning with `Responses` list the responses of various models to all questions.
- Users wishing to use apply the benchmark to new models may modify code in `client/ask.py` and `client/eval.py`. However, those wishing to extend existing benchmark questions to other domains of physics or science will have to modify `client/init.py`; for this purpose it is best to contact the authors for more complete documentation and an elaboration of their philosophy.

<br>
<br>

## Backend Server Setup

### Database Setup (Firebase)

1. Create a Firebase project and enable Firestore
2. Download your Firebase service account admin file and save it to  `db_admin.json` in the `server/` directory

### Server Setup

1. Install server dependencies:
   ```bash
   cd server
   pip install -r ../serve_requirements.txt
   ```

2. Set up ngrok for external access:
   ```bash
   chmod +x setup_ngrok.sh
   ./setup_ngrok.sh
   ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
   ```

3. Start the server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. In a separate terminal, start ngrok:
   ```bash
   ngrok http 8000
   ```

### Environment Variables

Create a `.env` file in the root directory with the following variables:

#### Example API Keys
```bash
# OpenAI API (for GPT models, o1 models)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API (for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google AI API (for Gemini models)
GOOGLE_API_KEY=your_google_api_key_here

# DeepSeek API (for DeepSeek models)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# xAI API (for Grok models)
XAI_API_KEY=your_xai_api_key_here

# OpenRouter API (for various models)
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

#### Required URLs
```bash
# Firebase Database URL
FIREBASE_DB_URL=https://your-project-id.firebaseio.com

# Server Base URL (your ngrok URL)
BASE_URL=https://your-ngrok-url.ngrok.io
```
<br>
<br>

## Querying the Server (`client/call_server.py`)

### `run(models, prompts, titles, num_elems, num_repeats=1)`
Starts a new experiment by sending prompts to the server. Only one job can run at once; additional attempts are blocked.

- `models`: list of model names to query
- `prompts`: list of prompts
- `titles`: list of unique titles for each prompt
- `num_elems`: list of values to extract per prompt (0 for a scalar, 1+ for an array with the specified length)
- `num_repeats`: how many times to repeat each query (default = 1, meaning it isn't repeated, only done once)
   - NOTE: can also pass in a 2D array from calling `client/call_db.py` get_attempts_left so that each model and prompt
   pair is repeated until a certain amount of attempts are completed

For the ith prompt in prompts, we expect it to have a title of titles[i] and to have an array length of `num_elems[i]` (or to be a scalar if `num_elems[i] == 0`)

### `status()`
Returns the current server status, including whether a job is running and progress info.

### `cancel()`
Sends a request to cancel the currently running job.

<br>
<br>

## Pulling Results (`client/call_db.py`)

### `save_result(title, model, save_file=None)`
Fetches all results for a given title and model.  
If `save_file` is provided, saves the results to a `.jsonl` file, with one line per attempt.

### `get_all_results(title, model_list)`
Fetches all results for a given title across multiple models.  
Returns a list of dictionaries, each containing model results.

### `save_all_results(titles, models, save_dir)`
Saves results for **every** `(title, model)` pair into separate `.jsonl` files inside the specified `save_dir`.  
Each file is named as `title-model.jsonl`.

###  `find_incomplete_prompts(find_incomplete_prompts(save_dir, n)`
Takes in a save_dir where the output files reside. For every model, find prompts in which the number of recorded entries is less than a specified threshold `n`.

### `get_attempts_left(models, titles, save_dir, n):`
Get the number of attempts left to get n total attempts for each title. Looks within save_directory. Outputs a 2d array where
output[`i`][`j`] is how many attempts model i has on title j to reach the target number. 

<br>
<br>

## Asking questions (`client/ask.py`)

Skipping past numerous subsidiary functions, these are the ones that matter and the user is expected to call.

### `ask_batch(size, retrieve, attempts)`

- Asks all questions to all models listed.
- Models to be called should be listed in the line `models = [...]`.
- `size` indicates the resolution of the questions given, where 1 refers to small, 2 medium, and 3 large. Recall that the standard leaderboard reflects answers to small resolution questions only (`size=1`).
- `retrieve` toggles whether previously generated spectra (saved under `Prompts_[...]`) are to be reused for the current set of questions; this frees the program from having to generate all 135 questions again from scratch (which may take several minutes), which is especially useful for debugging. Ça va sans dire that these questions should have been generated beforehand. `retrieve=0` to generate questions from scratch, and `retrieve=1` to reuse previously generated questions.
- `attempts` indicates the number of times each question to be posed to each model; that is, the sample size. Note that we can 'top up' responses such that if `n` responses have already been collected for some question where `attempts = m > n`, then only `m-n` more responses will be requested for that question. This requires the existing responses to be loaded into the appropriate `client/Responses_[...]` folder using the `save_batch()` function (next point).

### `save_batch(size, attempts)`

- Saves responses for the listed models (under `model=[...]`) from the database.
- `size` and `attempts` are the same parameters as before.

### `ask_resolution_iter(question_category, resolution_array, noise_ratio, attempts)` and `ask_noise_iter(question_category, resolution, noise_ratio_array, attempts)`

- Asks a single question (listed under the text variable `question_category`) for models listed in `models=[...]` for a total number of `attempts` stated.
- Available text strings for `question_category` are listed within the previous function in the script, `get_single_prompt`.
- Resolution sizes should be either given as an array (`resolution_array` in the first function) or fixed (`resolution` in the second function). Resolution is defined as the number of pixels along the horizontal axis of spectra.
- Noise level (as a fraction of the maximum signal intensity) should be fixed (`noise_ratio` in the first function) or given as an array (`noise_ratio_array` in the second).

### `save_resolution_iter(question_category, resolution_array, noise_ratio, attempts)` and `save_noise_iter(question_category, resolution, noise_ratio_array, attempts)`

- These save responses to questions asked by the previous two functions.
- Their input variables are equivalent to the ones described above.

<br>
<br>

## Evaluating responses and ranking models (`client/eval.py`)

Skipping past numerous subsidiary functions, these are the ones that matter and the user is expected to call.

### `score_models()`

- After testing all 135 questions a list of models in `client/ask.py` using the function `ask_batch()` and saving them from the databased using the function `save_batch()`, this function scores all models in the list `model_list_1` with accompanying names in `model_name_1`.
- Responses are read off from the relevant `Responses_[...]` folder (in this case, only low resolution questions were scored).
- Leaderbaords and detailed breakdowns are generated and saved under the relevant folders.

### `plot_models()`

- Similar function to the above, calls a set of `plot_model_responses(model_name, size)` for various models stated in `model_name` of resolution sizes listed (`size = 1,2,3` for small, medium, and large).
- Responses are read off from the relevant `Responses_[...]` folder

Note that the following functions were written to score a small number of question categories over a range of resolutions/frequencies; the user may extend these functions at will. 

### `score_A1_B1_single_resolution_iter(model, resolution_array_A1, resolution_array_B1, noise_ratio)`

- For a given `model`, plots how scores for question categories A1 and B1 vary with resolution (stated in the two respective resolution arrays), for a fixed `noise_ratio`.

### `score_four_single_noise_iter(model, resolution, noise_array)`

- For a given `model` and `resolution`, plots how scores for A1, B1, B2, and D1 (with coupling strength 1) vary with noise (stated in `noise_array`).

### `plot_noise_responses(model, resolution, noise_array, question_category)`

- Analogous to `plot_models()` but applies to noise-dependent questions given by the given function.
- Input variables are self-explanatory.



