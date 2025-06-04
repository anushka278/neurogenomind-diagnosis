# Extracts symptoms and diagnosis from full text PubMed clinical cases using GPT4o
# and standardizes diagnoses to match OMIM entries (including OMIM ID)

from openai import OpenAI
import os
import json
import re
from rapidfuzz import fuzz

# ─── Load OMIM JSON and build mappings ───
OMIM_JSON_PATH = "omim_full.json"

def normalize_label(s: str) -> str:
    """
    Lowercase, remove non-alphanumeric characters (except spaces), collapse multiple spaces.
    Used to normalize both OMIM titles and LLM outputs for lookup.
    """
    s = s.lower()
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_omim_mappings(json_path: str):
    """
    Load omim_full.json, which is a list of entries like:
      {
        "mimNumber": 203600,
        "preferredTitle": "ALOPECIA-EPILEPSY-OLIGOPHRENIA SYNDROME OF MOYNAHAN",
        "clinicalSynopsis": [ ... ]
      }
    Build:
      - norm_to_id:    normalized(preferredTitle) → mimNumber
      - norm_to_title: normalized(preferredTitle) → preferredTitle (original)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        omim_list = json.load(f)

    norm_to_id = {}
    norm_to_title = {}
    for entry in omim_list:
        title = entry.get("preferredTitle", "")
        mim_num = entry.get("mimNumber", None)
        if title and (mim_num is not None):
            norm = normalize_label(title)
            norm_to_id[norm] = mim_num
            norm_to_title[norm] = title

    return norm_to_id, norm_to_title

# Load mappings once at import time
omim_norm_to_id, omim_norm_to_title = load_omim_mappings(OMIM_JSON_PATH)

def get_omim_match(label: str, threshold: int = 80):
    """
    Attempt to match a given label string to an OMIM entry.
    First try exact normalized match; if not found, do fuzzy search over all OMIM titles.
    Returns: (matched_title, mimNumber) or (None, None) if no match above threshold.
    """
    norm_label = normalize_label(label)
    # Exact match
    if norm_label in omim_norm_to_id:
        return omim_norm_to_title[norm_label], omim_norm_to_id[norm_label]

    # Fuzzy match: iterate over all normalized OMIM titles
    best_score = 0
    best_norm = None
    for omim_norm in omim_norm_to_id:
        score = fuzz.ratio(norm_label, omim_norm)
        if score > best_score:
            best_score = score
            best_norm = omim_norm

    if best_score >= threshold:
        return omim_norm_to_title[best_norm], omim_norm_to_id[best_norm]

    return None, None

# ─── Initialize OpenAI client ───
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def extract_symptoms_and_diagnosis(text):
    # Get symptom list
    symptom_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert clinical NLP model."},
            {"role": "user", "content": f"""Given the following clinical case, extract ONLY the list of observed clinical symptoms. Provide 3-10.
                                            Do NOT include any mention of diagnosis or genetic findings.
                                            Output a Python-style list of 3-10 symptoms.
                                            Make sure to return symptoms *exactly* consistent with OMIM (Online Mendelian Inheritance in Man) database nomenclature.
            CASE:
            {text}
            """}
        ],
        temperature=0
    )
    symptoms = symptom_response.choices[0].message.content.strip()

    # Get final diagnosis
    diagnosis_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert clinical NLP model."},
            {"role": "user", "content": f"""Given the following clinical case, extract ONLY the final diagnosis of the patient.
                                            If there are multiple possibilities, choose the most likely one. Output: One-line diagnosis.
                                            Make sure to return a diagnosis that exactly matches **official disorder names listed in the OMIM (Online Mendelian Inheritance in Man) database**.
                                            Do not mention genetic findings, specific genes, or OMIM ID. Keep your format concise as shown below.
            CASE:
            {text}
            """}
        ],
        temperature=0
    )
    diagnosis = diagnosis_response.choices[0].message.content.strip()

    return symptoms, diagnosis

def run_extraction_on_folder(input_dir="fulltext", output_path="extracted_cases_final.jsonl"):
    output_data = []
    files = os.listdir(input_dir)

    for filename in files:
        if not filename.endswith(".txt"):
            continue

        pmid = filename.replace(".txt", "")
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Processing PMID {pmid}...")
        try:
            symptoms_raw, diagnosis_raw = extract_symptoms_and_diagnosis(text)

            # Parse symptoms list if possible
            if symptoms_raw.strip().startswith("["):
                try:
                    symptoms_list = eval(symptoms_raw)
                except Exception:
                    symptoms_list = symptoms_raw
            else:
                symptoms_list = symptoms_raw

            # Attempt to match the raw LLM diagnosis to an OMIM entry
            matched_title, matched_id = get_omim_match(diagnosis_raw)
            if matched_id is None:
                print(f"   [!] Warning: Could not match diagnosis '{diagnosis_raw}' to any OMIM entry.")
                # You may choose to skip this case or keep the original text; here, we'll keep original
                matched_title = diagnosis_raw
                matched_id = None

            case_entry = {
                "pmid": pmid,
                "symptoms": symptoms_list,
                "true diagnosis": matched_title,
                "omim_id": matched_id
            }
            output_data.append(case_entry)

        except Exception as e:
            print(f"[!] Failed to process PMID {pmid}: {e}")

    # Save all entries to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(output_data)} case entries to {output_path}")

# Run this
if __name__ == "__main__":
    run_extraction_on_folder()
