import os, json, random, time
from pathlib import Path
import pandas as pd
import re
import time
import numpy as np
from openai import OpenAI

# Read specific columns from a CSV file, returning a DataFrame and a JSON list.
def read_csv_columns(file_path, columns):
    """
    Read specific columns from a CSV file and return a DataFrame and JSON list.
    """
    df = pd.read_csv(file_path, usecols=columns, encoding='utf-8')
    json_list = df.to_dict('records')
    return df, json_list

# Read a txt file and return its content.
def read_txt_file(file_path):
    """
    Read a txt file and return its content.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# Load environment variables from a .env file.
def load_env_file(env_path):
    if not env_path.exists():
        return
    with open(env_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), value)

# Call the API to get the model response.
def get_evalue_analysis(system_prompt, row_data, model='gpt-4o', base_url=None, api_key=None):
    """
    Call the OpenAI API for E-value calculation and analysis, returning the response text.
    """
    user_message = f"""Please provide professional analysis based on the following research data:

Study Information:
- Exposure: {row_data.get('exposure', 'N/A')}
- Outcome: {row_data.get('outcome', 'N/A')}  
- Measured confounders: {row_data.get('measured confounders', 'N/A')}
- Effect size: {row_data.get('Effect size', 'N/A')}

CRITICAL INTERPRETATION PRINCIPLES:
- **E-value interpretation**: A LARGER E-value indicates the association is MORE robust to unmeasured confounding (i.e., a stronger unmeasured confounder would be required to explain away the observed effect, making it LESS LIKELY to be due to unmeasured confounding)
- **Cornfield's inequality**: Evaluates whether any single unmeasured confounder could plausibly have sufficient strength of association with both exposure and outcome to explain away the observed effect

Based on your professional epidemiological judgment, please:
1. Calculate the E-value using the appropriate formula
2. Evaluate from Cornfield inequality perspective: Consider (a) whether any single unmeasured confounder could possibly have the required strength of association with both exposure and outcome, (b) plausibility of such confounders in this specific context, (c) if any known strong confounders in this context have already been measured. Provide your analysis (1-2 sentences)
3. Evaluate from E-value perspective: Consider (a) the magnitude of the calculated E-value , (b) whether an unmeasured confounder with such strength is plausible given the exposure-outcome relationship. Provide your analysis (1-2 sentences)
4. Please consider BOTH Cornfield inequality and E-value evaluations above, and draw a conclusion: conclude whether unmeasured confounding is "unlikely", "possibly", or "highly likely" to explain away the observed association. Provide a comprehensive reason (2-3 sentences) that synthesizes both perspectives
5. Identify 3 potential unmeasured confounding variables relevant to this specific exposure-outcome relationship

IMPORTANT: Return ONLY the JSON format below, without any additional text, explanation, or thinking process before or after the JSON:

```json
{{
    "E-value": <calculated E-value>,
    "Cornfield's inequality evaluation": "<your analysis without conclusion>",
    "E-value evaluation": "<your analysis without conclusion>",
    "Final conclusion": "<unlikely|possibly|highly likely>",
    "Reason for final conclusion": "<comprehensive reason synthesizing both Cornfield's inequality and E-value>",
    "Potential unmeasured confounders": [
        "<confounder 1>",
        "<confounder 2>",
        "<confounder 3>"
    ]
}}
```
"""
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_completion_tokens=2000,
        temperature=0.0
    )
    response_text = response.choices[0].message.content.strip()
    return response_text

def extract_json_from_response(response_text):
    """
    Extract a JSON structure from the response text and return a DataFrame.
    Use multiple strategies to improve extraction success for model outputs.
    """
    # Strategy 1: match ```json ... ``` code blocks.
    code_block = re.search(r"```json\s*([\s\S]*?)```", response_text, re.IGNORECASE)
    if code_block:
        json_str = code_block.group(1).strip()
        try:
            response_json = json.loads(json_str)
            return pd.DataFrame([response_json])
        except Exception as e:
            print(f"Strategy 1 failed: {e}")
    
    # Strategy 2: match any ``` ... ``` code block.
    code_block_any = re.search(r"```\s*([\s\S]*?)```", response_text)
    if code_block_any:
        json_str = code_block_any.group(1).strip()
        # Remove possible language identifiers.
        json_str = re.sub(r'^(json|JSON)\s*', '', json_str)
        try:
            response_json = json.loads(json_str)
            return pd.DataFrame([response_json])
        except Exception as e:
            print(f"Strategy 2 failed: {e}")
    
    # Strategy 3: find the last complete JSON object (models often output JSON at the end).
    json_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text))
    if json_matches:
        # Try matches from the end.
        for match in reversed(json_matches):
            json_str = match.group()
            try:
                response_json = json.loads(json_str)
                # Validate required fields.
                if "E-value" in response_json or "Final conclusion" in response_json:
                    return pd.DataFrame([response_json])
            except:
                continue
    
    # Strategy 4: match a looser JSON block containing "E-value".
    match = re.search(r'\{[\s\S]*?"E-value"[\s\S]*?\}', response_text)
    if match:
        json_str = match.group()
        # Try to find the complete JSON structure.
        brace_count = 0
        end_pos = 0
        for i, char in enumerate(json_str):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        if end_pos > 0:
            json_str = json_str[:end_pos]
            try:
                response_json = json.loads(json_str)
                return pd.DataFrame([response_json])
            except Exception as e:
                print(f"Strategy 4 failed: {e}")
    
    # All strategies failed.
    print(f"Valid JSON not found; original content (first 500 chars): {response_text[:500]}...")
    print(f"Original content (last 500 chars): ...{response_text[-500:]}")
    return None

def save_to_csv(df, output_path):
    """
    Combine a list of DataFrames (with the same columns) and save to a CSV file.
    """
    combined_df = pd.concat(df, ignore_index=True)
    combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return combined_df


# Main function.
def main():
    
    model = 'claude-opus-4-1'
    """gpt-4o, claude-opus-4-1-20250805, DeepSeek-V3, gemini-2.5-pro, gpt-5-mini"""
    system_prompt = read_txt_file('system_prompt_v1.txt')
    filepath = 'sensi_data_v5.csv'
    sensi_df, sensi_json = read_csv_columns(filepath, ['exposure','outcome','measured confounders','Effect size'])
    results = []
    failed_rows = []  # Track failed rows.

    env_path = Path(__file__).resolve().parent / '.env'
    load_env_file(env_path)
    api_base_url = os.getenv('API_BASE_URL')
    api_key = os.getenv('API_KEY')
    if not api_base_url or not api_key:
        raise ValueError('Missing API_BASE_URL or API_KEY in .env')

    for idx, row in sensi_df.iterrows():
        print(f"Processing row {idx+1}/{len(sensi_df)}")
        row_data = sensi_json[idx]
        
        # Add retry logic.
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response_text = get_evalue_analysis(
                    system_prompt,
                    row_data,
                    model=model,
                    base_url=api_base_url,
                    api_key=api_key
                )
                result_df = extract_json_from_response(response_text)
                
                if result_df is not None:
                    results.append(result_df)
                    break  # Success: exit retry loop.
                elif attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries - 1}...")
                    time.sleep(3)  # Wait longer before retrying.
                else:
                    print(f"  Row {idx+1} failed after max retries.")
                    failed_rows.append(idx+1)
            except Exception as e:
                print(f"  Error processing row {idx+1}: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries - 1}...")
                    time.sleep(3)
                else:
                    failed_rows.append(idx+1)
        
        time.sleep(2)
    output_path = f"{model}_evalue_analysis_results.csv"
    if results:
        combined_df = save_to_csv(results, output_path)
        print(f"Results saved to {output_path}")
        print(f"Successfully processed: {len(results)}/{len(sensi_df)} rows")
        if failed_rows:
            print(f"Failed rows: {failed_rows}")
    else:
        print("No valid analysis results; CSV not generated.")
    
    # Merge original table and results.
    if results:
        merged_df = pd.concat([sensi_df.reset_index(drop=True), combined_df.reset_index(drop=True)], axis=1)
        merged_output_path = f"{model}_merged_evalue_analysis_results.csv"
        merged_df.to_csv(merged_output_path, index=False, encoding='utf-8-sig')
        print(f"Merged results saved to {merged_output_path}")

if __name__ == "__main__":
    main()
