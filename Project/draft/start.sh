#For generating LLM reports using Gemini-2.5-flash
python generate_llm_report_text_gemini.py \
  --main_csv /home/24chuong.ta/anaconda3/chun/development_space/MammoCLIP_ExperimentSetup_Fewshot/data/cmmd_metadata.csv \
  --radiomics_csv /home/24chuong.ta/anaconda3/chun/development_space/MammoCLIP_ExperimentSetup_Fewshot/data/final_radiomics_features.csv \
  --patient_col PatientID \
  --out_reports_csv /home/24chuong.ta/anaconda3/chun/development_space/MammoCLIP_ExperimentSetup_Fewshot/data/patient_llm_reports.csv \
  --out_merged_csv /home/24chuong.ta/anaconda3/chun/development_space/MammoCLIP_ExperimentSetup_Fewshot/data/cmmd_metadata_with_llm.csv \
  --backend sdk \
  --model gemini-2.5-flash \
  --temperature 0.0 \
  --max_output_tokens 256
