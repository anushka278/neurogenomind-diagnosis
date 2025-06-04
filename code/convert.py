# Convert .jsonl to .csv

import json
import csv
import ast

def convert_jsonl_to_csv(input_path, output_path):
    with open(input_path, 'r') as jsonl_file, open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['pmid', 'symptoms', 'true diagnosis']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for line in jsonl_file:
            entry = json.loads(line)
            raw_symptoms = entry.get('symptoms', '')

            # Extract the list from the formatted string block
            try:
                cleaned = raw_symptoms.strip("```python\n").strip("```").strip()
                symptom_list = ast.literal_eval(cleaned)
            except Exception as e:
                print(f"Error parsing symptoms for PMID {entry.get('pmid', 'unknown')}: {e}")
                symptom_list = []

            writer.writerow({
                # 'pmid': entry.get('pmid', ''),
                'symptoms': "; ".join(symptom_list),
                'true diagnosis': entry.get('true diagnosis', '')
            })

if __name__ == "__main__":
    input_file = "extracted_cases_final.jsonl"   # Change this to your actual file path
    output_file = "pubmed_cases.csv"
    convert_jsonl_to_csv(input_file, output_file)
    print(f"CSV saved to {output_file}")