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
from collections import defaultdict
from openai import OpenAI

load_dotenv()

GEMINI_MODEL_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3-flash-preview:generateContent"
)

DEFAULT_BATCH_SIZE = 5

# file to extract a dataframe from the excel file indicated as the input.
def read_excel(path):
    try:
        df = pd.read_excel(path).head(100)
        print(f"Read input file: {path}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file: {e}")

# write the confusion matrix to a new excel file with some basic formatting.
def write_confusion_excel(matrix_df, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Deleted existing file: {output_path}")

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

# send a batch of data to GPT-4 and get true and predicted labels. Parse the response as a json.
def call_gpt(batch_df):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are a veterinary radiology classification system.

Your task is to classify a canine thoracic radiology report into predefined binary labels.

Only use the report text in the columns labeled "Findings (original radiologist report)" and "Conclusions (original radiologist report)" to determine the presence of abnormalities. Do not use any other columns for classification.

OUTPUT FORMAT REQUIREMENTS:

You are a veterinary radiology classification system.

Your task is to classify a canine thoracic radiology report into predefined binary labels.

Return one JSON object per input case, matching the number of input rows.

Do not return more or fewer JSON objects than the number of input cases. Each JSON object must have a "CaseID" field that matches the "CaseID" from the input data, along with the classification labels.

Only use the report text in the columns labeled
"Findings (original radiologist report)"
and
"Conclusions (original radiologist report)"

Definitions:
- Abnormal = Abnormal finding present
- Normal = No abnormal finding mentioned

label_guidance = {{
    "perihilar_infiltrate": "Increased opacity centered around the lung hilus; perihilar interstitial or alveolar pattern.",
    "pneumonia": "Alveolar lung pattern, air bronchograms, focal or lobar consolidation, especially cranioventral.",
    "bronchitis": "Thickened bronchial walls, donut or tramline signs, bronchial pulmonary pattern.",
    "interstitial": "Diffuse hazy lung opacity without full alveolar consolidation; reticular pattern.",
    "diseased_lungs": "Generalized abnormal pulmonary pattern; lungs not radiographically normal.",
    "hypo_plastic_trachea": "Uniformly narrowed tracheal lumen compared to expected diameter.",
    "cardiomegaly": "Enlarged cardiac silhouette; increased vertebral heart score (VHS).",
    "pulmonary_nodules": "Discrete round soft tissue opacities within lung fields.",
    "pleural_effusion": "Fluid in pleural space; retracted lung lobes; scalloped lung margins; obscured cardiac silhouette.",
    "rtm": "Right middle lung lobe consolidation or focal alveolar pattern.",
    "focal_caudodorsal_lung": "Localized opacity in caudodorsal lung fields.",
    "focal_perihilar": "Localized opacity centered at or near lung hilus.",
    "pulmonary_hypoinflation": "Reduced lung volume; crowded pulmonary vessels; elevated diaphragm.",
    "right_sided_cardiomegaly": "Enlargement of right atrium or ventricle; cranial cardiac border rounding.",
    "pericardial_effusion": "Globoid cardiac silhouette; enlarged heart with sharp margins; possible pleural effusion.",
    "bronchiectasis": "Dilated bronchi; lack of tapering bronchial walls; visible to lung periphery.",
    "pulmonary_vessel_enlargement": "Enlarged pulmonary arteries or veins relative to adjacent bronchi.",
    "left_sided_cardiomegaly": "Enlarged left atrium or ventricle; caudal cardiac border bulging.",
    "thoracic_lymphadenopathy": "Enlarged mediastinal or hilar lymph nodes; widened cranial mediastinum.",
    "esophagitis": "Esophageal wall thickening or gas dilation; possible megaesophagus signs."
}}

Data:
{batch_df.to_json(orient="records")}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "radiology_classification",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "CaseID": {"type": "string"},
                                        "perihilar_infiltrate": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "pneumonia": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "bronchitis": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "interstitial": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "diseased_lungs": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "hypo_plastic_trachea": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "cardiomegaly": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "pulmonary_nodules": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "pleural_effusion": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "rtm": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "focal_caudodorsal_lung": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "focal_perihilar": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "pulmonary_hypoinflation": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "right_sided_cardiomegaly": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "pericardial_effusion": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "bronchiectasis": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "pulmonary_vessel_enlargement": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "left_sided_cardiomegaly": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "thoracic_lymphadenopathy": {"type": "string", "enum": ["Normal", "Abnormal"]},
                                        "esophagitis": {"type": "string", "enum": ["Normal", "Abnormal"]}
                                    },
                                    "required": [
                                        "perihilar_infiltrate",
                                        "pneumonia",
                                        "bronchitis",
                                        "interstitial",
                                        "diseased_lungs",
                                        "hypo_plastic_trachea",
                                        "cardiomegaly",
                                        "pulmonary_nodules",
                                        "pleural_effusion",
                                        "rtm",
                                        "focal_caudodorsal_lung",
                                        "focal_perihilar",
                                        "pulmonary_hypoinflation",
                                        "right_sided_cardiomegaly",
                                        "pericardial_effusion",
                                        "bronchiectasis",
                                        "pulmonary_vessel_enlargement",
                                        "left_sided_cardiomegaly",
                                        "thoracic_lymphadenopathy",
                                        "esophagitis"
                                    ]
                                }
                            }
                        },
                        "required": ["results"]
                    }
                }
            },
        )

        raw = response.choices[0].message.content
        # print(f"Raw API response:\n{raw}")
        parsed = json.loads(raw)
        results = parsed["results"]

        print(f"Batch returned {len(results)} lines.")

        return results

    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")

# Function to extract ground truth labels from specific cells in the Excel file
def extract_ground_truth_labels(path):
    try:
        ground_truth_df = pd.read_excel(
            path, usecols="K:AD", skiprows=0, nrows=100
        )
        label_names = [
        "perihilar_infiltrate",
        "pneumonia",
        "bronchitis",
        "interstitial",
        "diseased_lungs",
        "hypo_plastic_trachea",
        "cardiomegaly",
        "pulmonary_nodules",
        "pleural_effusion",
        "rtm",
        "focal_caudodorsal_lung",
        "focal_perihilar",
        "pulmonary_hypoinflation",
        "right_sided_cardiomegaly",
        "pericardial_effusion",
        "bronchiectasis",
        "pulmonary_vessel_enlargement",
        "left_sided_cardiomegaly",
        "thoracic_lymphadenopathy",
        "esophagitis"
        ]

        ground_truth_df.columns = label_names
        print("Extracted ground truth labels from input file.")
        return ground_truth_df
    except Exception as e:
        raise RuntimeError(f"Failed to extract ground truth labels: {e}")

# Function to load results from an Excel file instead of calling the LLM
def load_results_from_excel(file_path):
    try:
        results_df = pd.read_excel(file_path)
        print(f"Loaded results from: {file_path}")
        return results_df.to_dict(orient="records")
    except Exception as e:
        raise RuntimeError(f"Failed to load results from Excel file: {e}")

def call_gemini(batch_df):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    prompt = f"""
You are a veterinary radiology classification system.

Your task is to classify a canine thoracic radiology report into predefined binary labels.

Only use the report text in the colomns labeled "Findings (original radiologist report)" and "Conclusions (original radiologist report)" to determine the presence of abnormalities. Do not use any other columns for classification.

You must:
- Return ONLY valid JSON.
- Use EXACT label names provided.
- Output "Abnormal" for any abnormal finding.
- Output "Normal" for no abnormal finding mentioned.
- Include ALL labels.
- Do NOT include explanations.
- Do NOT include extra fields.
- Do NOT include markdown formatting.

Definitions:
- Abnormal = Abnormal finding present
- Normal = No abnormal finding mentioned

label_guidance = {{
    "perihilar_infiltrate": "Increased opacity centered around the lung hilus; perihilar interstitial or alveolar pattern.",
    
    "pneumonia": "Alveolar lung pattern, air bronchograms, focal or lobar consolidation, especially cranioventral.",
    
    "bronchitis": "Thickened bronchial walls, donut or tramline signs, bronchial pulmonary pattern.",
    
    "interstitial": "Diffuse hazy lung opacity without full alveolar consolidation; reticular pattern.",
    
    "diseased_lungs": "Generalized abnormal pulmonary pattern; lungs not radiographically normal.",
    
    "hypo_plastic_trachea": "Uniformly narrowed tracheal lumen compared to expected diameter.",
    
    "cardiomegaly": "Enlarged cardiac silhouette; increased vertebral heart score (VHS).",
    
    "pulmonary_nodules": "Discrete round soft tissue opacities within lung fields.",
    
    "pleural_effusion": "Fluid in pleural space; retracted lung lobes; scalloped lung margins; obscured cardiac silhouette.",
    
    "rtm": "Right middle lung lobe consolidation or focal alveolar pattern.",
    
    "focal_caudodorsal_lung": "Localized opacity in caudodorsal lung fields.",
    
    "focal_perihilar": "Localized opacity centered at or near lung hilus.",
    
    "pulmonary_hypoinflation": "Reduced lung volume; crowded pulmonary vessels; elevated diaphragm.",
    
    "right_sided_cardiomegaly": "Enlargement of right atrium or ventricle; cranial cardiac border rounding.",
    
    "pericardial_effusion": "Globoid cardiac silhouette; enlarged heart with sharp margins; possible pleural effusion.",
    
    "bronchiectasis": "Dilated bronchi; lack of tapering bronchial walls; visible to lung periphery.",
    
    "pulmonary_vessel_enlargement": "Enlarged pulmonary arteries or veins relative to adjacent bronchi.",
    
    "left_sided_cardiomegaly": "Enlarged left atrium or ventricle; caudal cardiac border bulging.",
    
    "thoracic_lymphadenopathy": "Enlarged mediastinal or hilar lymph nodes; widened cranial mediastinum.",
    
    "esophagitis": "Esophageal wall thickening or gas dilation; possible megaesophagus signs."
}}

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
        timeout=180
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


def to_binary(x):
    return 1 if x == "Abnormal" else 0

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
    batch_index = 1
    # for batch in batch_dataframe(df, args.batch_size):
    #     print(f"Processing batch {batch_index} of {len(batch)} rows...")
    #     # print(f"Batch input data:\n{batch}")


    #     for attempt in range(3):  # Retry up to 3 times, had trouble with gpt returning number of lines not matching input rows
    #         try:
    #             batch_results = call_gpt(batch)
    #             if len(batch_results) == len(batch):
    #                 break
    #             else:
    #                 print(f"Retrying batch {batch_index}, attempt {attempt + 1}")
    #         except Exception as e:
    #             print(f"Error on attempt {attempt + 1}: {e}")

    #     all_results.extend(batch_results)
    #     batch_index += 1

    all_results = load_results_from_excel("all_results.xlsx")

    # Save all_results to an Excel file
    # all_results_df = pd.DataFrame(all_results)
    # all_results_output_file = "all_results.xlsx"
    # all_results_df.to_excel(all_results_output_file, index=False)
    # print(f"All results saved to: {all_results_output_file}")

    # print(all_results)

    ground_truth_labels = extract_ground_truth_labels(args.input_file)
    # print(ground_truth_labels.head())

    labels = [k for k in all_results[0].keys() if k != "CaseID"]

    conf_matrix = {
        label: {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        for label in labels
    }
    global_counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for i in range(len(all_results)):

        ai_case = all_results[i]
        gt_row = ground_truth_labels.iloc[i]

        for label in labels:

            y_pred = to_binary(ai_case[label])
            y_true = to_binary(gt_row[label])

            if y_true == 1 and y_pred == 1:
                conf_matrix[label]["TP"] += 1
                global_counts["TP"] += 1

            elif y_true == 0 and y_pred == 0:
                conf_matrix[label]["TN"] += 1
                global_counts["TN"] += 1

            elif y_true == 0 and y_pred == 1:
                conf_matrix[label]["FP"] += 1
                global_counts["FP"] += 1

            elif y_true == 1 and y_pred == 0:
                conf_matrix[label]["FN"] += 1
                global_counts["FN"] += 1

    # Calculate sensitivity and specificity for each label
    metrics = {}
    for label, counts in conf_matrix.items():
        tp = counts["TP"]
        tn = counts["TN"]
        fp = counts["FP"]
        fn = counts["FN"]

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics[label] = {
            "Sensitivity": sensitivity,
            "Specificity": specificity
        }

    # print("Label-wise Sensitivity and Specificity:")
    # for label, metric in metrics.items():
    #     print(f"{label}: Sensitivity = {metric['Sensitivity']:.2f}, Specificity = {metric['Specificity']:.2f}")

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={"index": "Label"}, inplace=True)

    conf_matrix_df = pd.DataFrame.from_dict(conf_matrix, orient="index")
    combined_df = conf_matrix_df.merge(metrics_df, left_index=True, right_on="Label")

    combined_output_file = "combined_results.xlsx"
    write_confusion_excel(combined_df, combined_output_file)

    print(f"Combined confusion matrix and metrics saved to: {combined_output_file}")

if __name__ == "__main__":
    main()