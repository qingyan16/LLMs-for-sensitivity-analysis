# LLMs-for-sensitivity-analysis

The core implementation of this study is included  in the script `all.py`.

- Input data are provided in `sensi_data_v5.csv.` This file includes extracted information for each observational study. The script will read each row of this file as model input.

- Initialization prompt is loaded from `system_prompt_v1.txt`.

- Full prompts are defined within the script.

- The output is a structured JSON result, which is further converted to .csv for reporting the final results.
