import argparse
import os
import json
import re
import requests
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils.dataframe import dataframe_to_rows
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3-flash-preview:generateContent"
)

DEFAULT_BATCH_SIZE = 10

# file to extract a dataframe from the excel file indicated as the input.
def read_excel(path):
    try:
        df = pd.read_excel(path).head(10) # limit to 10 rows for testing, I guess this makes batch size 10 irrelevant but can change later
        print(f"Read input file: {path}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file: {e}")

# write the confusion matrix to a new excel file with some basic formatting.
def write_confusion_excel(matrix_df, output_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Confusion Matrix"

    ws["A1"] = "Confusion Matrix"
    ws["A1"].font = Font(bold=True)

    ws.append([])

    for row in dataframe_to_rows(matrix_df, index=True, header=True):
        ws.append(row)

    wb.save(output_path)
    print(f"Wrote output file: {output_path}")

def extract_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    return text

# batch the dataframe into smaller chunks to speed up processing and avoid hitting API limits.
def batch_dataframe(df, batch_size):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]

# send a batch of data to Gemini and get true and predicted labels. Parse the response as a json.
def call_gemini(batch_df):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    prompt = f"""
You are given tabular data in JSON format.
For EACH row, produce a JSON object with:
- true_label
- predicted_label

Return ONLY valid JSON in this exact format:
[
  {{"true_label": "...", "predicted_label": "..."}}
]

Data:
{batch_df.to_json(orient="records")}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(
        f"{GEMINI_MODEL_URL}?key={api_key}",
        json=payload,
        timeout=90
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API error {response.status_code}: {response.text}"
        )

    raw_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    clean_text = extract_json(raw_text)

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Invalid JSON returned by Gemini:\n{clean_text}"
    )

# uses pd.crosstab to build a confusion matrix of true vs predicted.
def build_confusion_matrix(results):
    df = pd.DataFrame(results)
    return pd.crosstab(
        df["true_label"],
        df["predicted_label"],
        rownames=["Actual"],
        colnames=["Predicted"]
    )

# main function to read command line input, batch the data, and write the confusion matrix to a new excel file.
def main():
    parser = argparse.ArgumentParser(
        description="Generate a confusion matrix from Excel using Google Gemini"
    )
    parser.add_argument("input_file", help="Path to input Excel file")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of rows per Gemini request"
    )

    args = parser.parse_args()

    df = read_excel(args.input_file)

    all_results = []
    for batch in batch_dataframe(df, args.batch_size):
        print(f"Processing batch of {len(batch)} rows...")
        batch_results = call_gemini(batch)
        all_results.extend(batch_results)

    matrix = build_confusion_matrix(all_results)

    output_file = f"confusion_matrix_test.xlsx"

    write_confusion_excel(matrix, output_file)

    print(f"Confusion matrix saved to: {output_file}")


if __name__ == "__main__":
    main()