# Case 4: Churn Detective 
**An AI-Powered Telecom Retention Intelligence Platform**

## 1. Executive Summary & Business Impact

**Context:** 
Our mid-sized telecom operation has been experiencing a critical customer retention issue, losing postpaid customers at an alarming rate of 2.3% monthly. This figure sits significantly above the industry benchmark of 1.5% (a 50% underperformance), threatening long-term revenue stability and driving up customer acquisition costs.

**The Solution:** 
To combat this, we developed and deployed **Churn Detective**, an AI-powered Telecom Retention Intelligence Platform. This solution moves beyond descriptive analytics by proactively answering three strategic questions for the business:
1. *Who will churn?*
2. *Why are they leaving?*
3. *Who can be saved by an offer?*

**Business Impact & ROI:** 
The platform's core predictive engine, a LightGBM classifier, achieved an impressive AUC-ROC of 0.856. By operationalizing these predictions and targeting the top 2,100 customers sorted by their uplift score, we expect to save approximately 630 customers. Assuming an offer cost of $15 and a 30% retention rate among targeted persuadable customers, this intervention strategy is projected to generate **$817,000 in net value**, delivering a substantial **254% Return on Investment (ROI)**.

## 2. Features & Technology Stack

The platform is built on a robust, state-of-the-art machine learning stack designed for both predictive power and business interpretability:

*   **LightGBM (Classification):** Serves as the core predictive model to identify customers with a high probability of churning. Its gradient-boosting framework handles complex, non-linear relationships efficiently.
*   **TreeSHAP (Interpretability):** Used to demystify the LightGBM model. SHAP (SHapley Additive exPlanations) values provide local and global interpretability, allowing us to explain the exact drivers behind every individual churn prediction to stakeholders.
*   **KMeans (Segmentation):** Applied to group the customer base into distinct behavioral clusters, enabling highly personalized marketing and retention strategies.
*   **S-Learner (Uplift Modelling):** A meta-learning approach used to estimate the Individual Treatment Effect (ITE). Instead of just predicting who will churn, the S-Learner predicts who is most likely to change their decision *because* of a retention offer (identifying the "Persuadables").
*   **Streamlit (Dashboard):** Powers the interactive front-end application, bridging the gap between raw ML outputs and business decision-makers through an intuitive UI.

**Optimization Strategy:** 
Crucially, due to a moderate class imbalance in our dataset (36.2% churn rate), the model was explicitly optimized for **AUC-PR (Area Under the Precision-Recall Curve)** rather than just AUC-ROC. Achieving an AUC-PR of **0.804** ensures that our model minimizes false positives, meaning our retention budget is spent efficiently on actual at-risk customers.

## 3. Streamlit Dashboard Walkthrough

The Churn Detective application is structured into four primary interactive modules designed for executive and operational use:

### Page 1 - Churn Overview
This landing page provides an immediate, high-level pulse on the telecom's retention health.
*   **KPI Cards:** Displays top-level metrics instantly summarizing the current churn landscape.
*   **Demographic Breakdown:** Interactive charts detail churn rates distributed by contract type, customer tenure, and monthly charges.
*   **TreeSHAP Global Drivers:** A global bar chart exposes the overarching reasons customers are leaving. It highlights critical friction points, most notably: the type of **Contract**, the early **tenure cliff** (where new customers drop off), and the **4+ support calls cliff** (a strong leading indicator of frustration and imminent churn).

### Page 2 - Customer Segments
This page translates KMeans clustering results into actionable marketing personas, utilizing intuitive persona cards, radar charts, and a scatter map. We identified 4 distinct segments:
*   **Price-Sensitive Shoppers:** Highly responsive to pricing changes and competitive offers.
*   **Frustrated Early Adopters:** Customers who have recently joined but exhibit high support needs and dissatisfaction.
*   **Quietly Disengaging Veterans:** Long-term customers whose usage has slowly dropped, signaling a risk of passive churn.
*   **At-Risk Budget Customers:** Lower-tier plan subscribers struggling with cost or perceived value.

### Page 3 - Retention Simulator
A highly interactive financial modeling tool built directly for the CMO.
*   **Interactive Sliders:** Users can dynamically adjust the *Offer Cost*, *Retention Rate*, and the *Targeting Threshold*.
*   **Dynamic ROI Curves:** As sliders are moved, the ROI and Expected Net Value curves update in real-time. This allows business leaders to simulate different campaign strategies, visually pinpoint the optimal financial threshold, and confidently allocate the retention budget.

### Page 4 - Uplift Targeting
The operational heart of the platform, leveraging the S-Learner meta-learner to maximize campaign efficiency.
*   **Four-Quadrant Distribution:** Visualizes the customer base categorized into *Persuadables* (savable with an offer), *Lost Causes* (will leave regardless), *Sure Savers* (will stay without an offer), and *Sleeping Dogs* (an offer might trigger them to leave).
*   **Qini & Decile Charts:** Demonstrates the cumulative uplift gained by targeting our model's recommendations versus a randomized targeting approach.
*   **Actionable Output:** Provides a downloadable CSV list of ranked "Persuadables", empowering the retention team to execute the campaign immediately.

## 4. Future Plans & Roadmap

To evolve Churn Detective from an analytical tool into an automated, real-time retention engine, we have defined the following roadmap:

*   **Immediate (0-30 days):** Execute the first live campaign. We will run an A/B test with a 20% holdout control group to empirically validate our assumed 30% retention rate and measure the true Area Under the Uplift Curve (AUUC).
*   **Short-Term (1-3 months):** Establish engineering rigor. We will deploy the model behind a REST API (`POST /score`), implement `CalibratedClassifierCV` to ensure our predicted probabilities align with true churn likelihoods, and configure automated Apache Airflow DAGs to trigger model retraining automatically if the Population Stability Index (PSI) exceeds 0.20 (indicating data drift).
*   **Long-Term (3-12 months):** Transition to real-time, event-driven intelligence. This involves integrating an Online Feature Store (such as Feast or Tecton), streaming real-time customer support events via Kafka to trigger preemptive interventions *before* a customer hits the critical "4-call cliff", and building direct API integrations to automatically push the generated "Persuadables" list into CRMs like Salesforce or HubSpot.
