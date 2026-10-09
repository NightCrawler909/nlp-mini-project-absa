# Aspect-Based Sentiment Analysis (ABSA) Mini-Project

This project implements a complete Aspect-Based Sentiment Analysis pipeline on product and app reviews. It evaluates rule-based, classical ML (CRF, LR/SVM), and Transformer approaches.

## Dataset
- **SemEval-2014 Task 4**: Laptop and Restaurant domains. The datasets contain aspect terms, their offsets, and polarity (positive/negative/neutral).
- **Amazon Reviews**: Used as an unlabeled corpus for insights and word cloud generation.

## Setup
1. Ensure Python 3.10+ is installed.
2. Run `python -m venv venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Spacy model: `pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz`
   
## Running the Pipeline
Simply execute:
```bash
python run_all.py
```
This will:
- Parse the cached datasets into JSON format and perform EDA (`src/data.py` and `src/preprocess.py`)
- Train all Aspect Term Extraction (ATE) models
- Train all Aspect Sentiment Classification (ASC) models
- Perform end-to-end evaluation
- Save models, comparison tables, and figures in the `results/` folder.

## Demo App
A Streamlit app is provided in `app/app.py`.
Run it using:
```bash
streamlit run app/app.py
```
The app has two tabs:
1. **Single Review**: Paste a review and see aspect highlights.
2. **Bulk Upload**: Upload a CSV to view an interactive dashboard of aspect frequencies and sentiments.

## Results
A summary of the model evaluation is saved to `results/tables/model_comparison.csv` and an error analysis is generated in `results/tables/error_analysis.md`.
