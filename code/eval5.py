# Use LLMs (GPT4o and Claude) to get top diagnoses based on symptoms from CSV file
# Now with OMIM‐ID‐based matching via omim_full.json + fuzzy matching via rapidfuzz

import os
import csv
import re
import json
from typing import Optional, Dict
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from huggingface_hub import InferenceClient

# ─── We add rapidfuzz for fuzzy string matching ───
from rapidfuzz import fuzz

# ─── Load environment variables ───
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment or .env file")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("Please set ANTHROPIC_API_KEY in your environment or .env file")
if not HUGGINGFACE_API_KEY:
    raise RuntimeError("Please set HUGGINGFACE_API_KEY in your environment or .env file")

# ─── Initialize LLM clients ───
gpt_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── Model configuration ───
MODEL_CONFIG = {
    'gpt-4o': {
        'provider': 'openai',
        'model': 'gpt-4o'
    },
    'claude-3-5-sonnet-latest': {
        'provider': 'anthropic',
        'model': 'claude-3-5-sonnet-latest'
    }
}
MODEL_DISPLAY = {
    'gpt-4o': 'GPT4o',
    'claude-3-5-sonnet-latest': 'Claude'
}

PROMPT_TEMPLATE = """
You are a geneticist, neurologist and advanced clinical NLP model. Based only on the following list of patient symptoms, list the {top_n} most likely genetic *neurological* disorder diagnoses.
Only return diagnoses that exactly match **official disorder names listed in the OMIM (Online Mendelian Inheritance in Man) database**.

Symptoms:
{symptoms}

Return a numbered list of {top_n} diagnoses, one per line. Each diagnosis should match OMIM nomenclature exactly.
An example of the desired output format is shown below:
"
1. ALOPECIA-EPILEPSY-OLIGOPHRENIA SYNDROME OF MOYNAHAN
2. CEREBELLAR HYPOPLASIA/ATROPHY, EPILEPSY, AND GLOBAL DEVELOPMENTAL DELAY; CHEGDD
3. EPILEPSY, EARLY-ONSET, 4, VITAMIN B6-DEPENDENT; EPEO4
...
"""

# ─── Functions to parse LLM output ───
def extract_diagnoses_from_response(response, top_n):
    """
    Parse exactly top_n diagnoses from an LLM response formatted as:
      1. Diagnosis A
      2. Diagnosis B
      ...
    Returns a list of diagnosis strings (all lowercased).
    """
    raw_lines = []
    if isinstance(response, str):
        raw_lines = response.splitlines()
    elif isinstance(response, list):
        # Anthropic .messages.create() → response.content is a list of TextBlock-like objects
        for block in response:
            if hasattr(block, 'text'):
                text = block.text
            elif isinstance(block, dict) and 'text' in block:
                text = block['text']
            else:
                text = str(block)
            raw_lines.extend(str(text).splitlines())
    else:
        raw_lines = str(response).splitlines()

    diagnoses = []
    numbered_pattern = re.compile(r'^\s*(\d+)\.\s*(.+)$')
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        m = numbered_pattern.match(line)
        if m:
            diag_text = m.group(2).strip()
            diagnoses.append(diag_text.lower())
        if len(diagnoses) >= top_n:
            break

    return diagnoses[:top_n]

def normalize_label(s: str) -> str:
    """
    Lowercase, remove punctuation (except spaces), collapse multiple spaces.
    Used to normalize both OMIM names and LLM guesses for lookup.
    """
    s = s.lower()
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ─── Load OMIM JSON and build exact mapping → {"normalized OMIM title": mimNumber} ───
def load_omim_mapping(json_path: str) -> Dict[str, int]:
    """
    Load omim_full.json, which is a list of entries like:
      {
        "mimNumber": 203600,
        "preferredTitle": "ALOPECIA-EPILEPSY-OLIGOPHRENIA SYNDROME OF MOYNAHAN",
        "clinicalSynopsis": [ ... ]
      }
    Build a dictionary that maps normalized(preferredTitle) → mimNumber.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        omim_list = json.load(f)

    mapping: Dict[str, int] = {}
    for entry in omim_list:
        title = entry.get("preferredTitle", "")
        mim_num = entry.get("mimNumber", None)
        if title and mim_num is not None:
            norm = normalize_label(title)
            mapping[norm] = mim_num

    return mapping

OMIM_JSON_PATH = "omim_full.json"
omim_name_to_id = load_omim_mapping(OMIM_JSON_PATH)

def get_omim_id(label: str) -> Optional[int]:
    """
    Exact lookup: normalize the label and return the MIM number if found.
    """
    norm = normalize_label(label)
    return omim_name_to_id.get(norm)

# ─── Fuzzy lookup: compare the normalized guess against all OMIM titles, pick best score ───
def fuzzy_find_omim_id(label: str, mapping: Dict[str,int], threshold: int = 95) -> Optional[int]:
    """
    Given a label string (LLM guess), normalize it, then compute fuzz.ratio(...)
    against every key in mapping. Return the mimNumber of the best‐scoring
    title if that score ≥ threshold, else None.
    """
    guess_norm = normalize_label(label)
    best_score = 0
    best_mim = None

    for title_norm, mim_num in mapping.items():
        # Compute Levenshtein‐ratio between guess_norm and each OMIM title_norm
        score = fuzz.ratio(guess_norm, title_norm)
        if score > best_score:
            best_score = score
            best_mim = mim_num

    if best_score >= threshold:
        return best_mim
    return None

def get_omim_id_fuzzy(label: str) -> Optional[int]:
    """
    First try exact lookup; if not found, fall back to fuzzy matching.
    """
    exact = get_omim_id(label)
    if exact is not None:
        return exact
    return fuzzy_find_omim_id(label, omim_name_to_id, threshold=80)


# ─── Function to send prompt to LLM and return raw text ───
def get_top_diagnoses(symptoms: str, top_n: int, llm_key: str) -> str:
    cfg = MODEL_CONFIG.get(llm_key)
    if cfg is None:
        raise ValueError(f"Unknown LLM '{llm_key}'. Choose from {list(MODEL_CONFIG)}")

    provider = cfg['provider']
    model_id = cfg['model']
    prompt = PROMPT_TEMPLATE.format(top_n=top_n, symptoms=symptoms)

    if provider == 'openai':
        resp = gpt_client.chat.completions.create(
            model=model_id,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()

    elif provider == 'anthropic':
        resp = anthropic_client.messages.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0
        )
        blocks = resp.content
        full_text = "".join(block.text for block in blocks)
        return full_text.strip()

    else:
        raise ValueError(f"Unsupported provider '{provider}' for LLM '{llm_key}'")


# ─── Main processing loop ───
def main():
    input_csv = "results_final/clinical-cases.csv"
    output_csv = "results_final/clinical-cases-results-llm.csv"
    # input_csv = "results_final/pubmed_cases.csv"
    # output_csv = "results_final/pubmed-cases-results-llm.csv"

    print(f"▶ Starting processing. Reading from '{input_csv}'.")
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = [
            'symptoms',
            'diagnosis',
            'GPT4o top 10',
            'GPT4o top 20',
            'GPT4o match top 10',
            'GPT4o match top 20',
            'Claude top 10',
            'Claude top 20',
            'Claude match top 10',
            'Claude match top 20'
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Count total cases for progress reporting
        total_cases = sum(1 for _ in open(input_csv, 'r', encoding='utf-8')) - 1
        print(f"ℹ️  Detected {total_cases} cases (rows). Now iterating...\n")

        # infile.seek(0)
        # next(reader)  # skip header in DictReader

        for idx, row in enumerate(reader, start=1):
            symptoms = row['symptoms'].strip()
            true_diag = row['diagnosis'].strip()
            # true_diag = row['true diagnosis'].strip()
            norm_true = normalize_label(true_diag)

            print("─" * 60)
            print(f"▶ Case {idx}/{total_cases}: \"{true_diag}\"")
            print(f"   Symptoms: {symptoms[:60]}{'...' if len(symptoms) > 60 else ''}")

            output_row = {
                'symptoms': symptoms,
                'diagnosis': true_diag
            }

            # ─── Look up OMIM ID for the true diagnosis, using fuzzy matching ───
            true_id = get_omim_id_fuzzy(true_diag)

            for llm_key in ['gpt-4o', 'claude-3-5-sonnet-latest']:
                disp_name = MODEL_DISPLAY[llm_key]

                for top_n in [10, 20]:
                    col_prefix = f"{disp_name} top {top_n}"
                    match_col = f"{disp_name} match top {top_n}"

                    try:
                        print(f"   • [{disp_name} top {top_n}] Sending prompt to model...")
                        raw_resp = get_top_diagnoses(symptoms, top_n, llm_key)
                        top_list = extract_diagnoses_from_response(raw_resp, top_n)

                        # Join into single string for CSV
                        joined_list = ",".join(top_list)
                        output_row[col_prefix] = joined_list

                        # ─── For each guessed diagnosis, do fuzzy lookup ───
                        match_flag = 0
                        guess_ids = []
                        for guess in top_list:
                            guess_id = get_omim_id_fuzzy(guess)
                            guess_ids.append(guess_id)
                            if true_id is not None and guess_id is not None and true_id == guess_id:
                                match_flag = 1
                                break

                        # If still no match (true_id or guess_id remains None),
                        # fall back to your old substring‐based check:
                        if (true_id is None) or all(gid is None for gid in guess_ids):
                            for g_norm in [normalize_label(g) for g in top_list]:
                                if norm_true == g_norm or norm_true in g_norm or g_norm in norm_true:
                                    match_flag = 1
                                    break

                        output_row[match_col] = match_flag

                        print(f"     → Received {len(top_list)} diagnoses. Match={'Yes' if match_flag else 'No'}")
                        # print(f"     → True OMIM ID (fuzzy→exact): {true_diag}")
                        # print(f"     → Top guesses' OMIM IDs (fuzzy→exact): {guess_ids}")
                        # print(f"     → Top {top_n} diagnoses: {', '.join(joined_list)}")
                    except Exception as e:
                        output_row[col_prefix] = ""
                        output_row[match_col] = 0
                        print(f"   [!] Error for Case {idx}, {disp_name} top {top_n}: {type(e).__name__}: {e}")

            writer.writerow(output_row)
            print(f"✅ Finished Case {idx}/{total_cases}\n")

    print("─" * 60)
    print(f"✅ All done. Results written to '{output_csv}'")


if __name__ == "__main__":
    main()
