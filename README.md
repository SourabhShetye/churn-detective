Live demo: [URL]
Repo: [URL]
Demo video: [URL]

Churn Detective applies a LightGBM churn classifier with SHAP explainability, KMeans customer segmentation, and an S-Learner uplift model in a Streamlit app.

## How to run locally
pip install -r requirements.txt
streamlit run app/streamlit_app.py
python src/evaluate.py
python src/segment.py

## Stack
- Python
- Streamlit
- LightGBM
- scikit-learn
- SHAP
- pandas, numpy

## What's NOT done
- No production monitoring or alerting pipeline
- No automated model retraining
- No live deployment or authentication layer
- No true randomized uplift experiment in the current dataset

## In production, I would also add
- CI/CD deployment with a managed cloud service and secure API access
- automated data validation, feature pipeline, and drift detection
- a model registry and scheduled retraining pipeline
- campaign measurement and ROI tracking for uplift offers

![Screenshot](link)

This project is released under the MIT License.
