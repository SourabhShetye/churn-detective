## Assumptions I made
- The available churn dataset is representative of the customer base and labels are accurate.
- Customer behavior and campaign impact can be approximated with features available in the current data.
- The uplift objective is best served by a single model architecture for the current prototype.

## Trade-offs
Choice | Alternative | Why I picked this
LightGBM | Random forest or logistic regression | LightGBM balances strong tabular accuracy, fast training, and explanation compatibility.
SHAP explainability | LIME or feature importance only | SHAP gives consistent local and global insights that are easy to present to business stakeholders.
KMeans segmentation | Manual buckets or hierarchical clustering | KMeans is simple, reproducible, and provides clear groups for customer targeting.
S-Learner uplift model | Two-model approach or X-Learner | S-Learner is easier to implement and maintain with the current dataset size and structure.

## What I de-scoped and why
- Live production deployment and authentication, because the priority was a working predictive and uplift prototype.
- Automated model retraining and drift monitoring, because the current submission focuses on analysis and explanation.
- Full experimental campaign design, because the dataset does not include true randomized treatment labels.

## What I'd do differently with another day
- Add a data validation and model drift pipeline to keep predictions stable over time.
- Deploy the Streamlit app on a managed service with secure access and logging.
- Expand uplift validation with a true randomized control experiment or X-Learner approach.
- Build a lightweight campaign ROI dashboard to connect model insights to business decisions.
