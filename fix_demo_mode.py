import sys
import re

# 1. Update streamlit_app.py
content = open('app/streamlit_app.py', 'r', encoding='utf-8').read()

c1 = """    if missing:
        artifacts["DEMO_MODE"] = True
        artifacts["_missing"]  = missing

    return artifacts"""
c1_new = """    if missing:
        import streamlit as st
        st.error(f"Missing models: {missing}. Run notebooks to generate them.")
        st.stop()

    return artifacts"""

content = content.replace(c1, c1_new)

c2 = """    if test_path.exists():
        df = pd.read_parquet(test_path)
        return df, False   # (dataframe, is_demo)"""
c2_new = """    if test_path.exists():
        df = pd.read_parquet(test_path)
        return df
    else:
        import streamlit as st
        st.error(f"Data not found at {test_path}. Please run notebooks first.")
        st.stop()"""
content = content.replace(c2, c2_new)

content = re.sub(r'    # ── Demo mode: generate synthetic data matching schema ───────────────.*?    return df, True   # \(dataframe, is_demo\)\n', '', content, flags=re.DOTALL)

content = content.replace('df, is_demo = load_data()\n    st.session_state.df      = df\n    st.session_state.is_demo = is_demo', 'st.session_state.df = load_data()\n    st.session_state.is_demo = False')

c3 = """    # Demo mode indicator
    if st.session_state.get("is_demo", True):
        st.markdown(\"\"\"
        <div class="warning-box">
            ⚠️ <strong>Demo mode</strong><br>
            Displaying synthetic data. Drop your CSV into
            <code>data/raw/</code> and run the notebooks to activate
            live model outputs.
        </div>
        \"\"\", unsafe_allow_html=True)
    else:
        st.success("✅ Live model — real data loaded")"""
c3_new = """    st.success("✅ Live model — real data loaded")"""
content = content.replace(c3, c3_new)

open('app/streamlit_app.py', 'w', encoding='utf-8').write(content)


# 2. Update churn_overview.py
content = open('app/pages/churn_overview.py', 'r', encoding='utf-8').read()

content = re.sub(r'    if os\.path\.exists\(shap_fig_path\):\n        st\.image\(shap_fig_path, caption="SHAP Global Feature Importance \(test set\)"\)\n    else:\n.*?        if is_demo:\n            st\.caption\("⚠️ Demo mode — synthetic SHAP values shown\. "\n                       "Run notebook 02_baseline_model\.ipynb to generate real SHAP output\."\)\n', r'''    if os.path.exists(shap_fig_path):
        st.image(shap_fig_path, caption="SHAP Global Feature Importance (test set)")
    else:
        import streamlit as st
        st.error(f"SHAP figure not found at {shap_fig_path}. Run notebooks.")
''', content, flags=re.DOTALL)

open('app/pages/churn_overview.py', 'w', encoding='utf-8').write(content)

# 3. Update retention_simulator.py
content = open('app/pages/retention_simulator.py', 'r', encoding='utf-8').read()

content = re.sub(r'    if "churn_proba" not in df\.columns or df\.empty:\n        st\.info\("Run notebook 02_baseline_model\.ipynb to generate churn probabilities\."\)\n        # Synthetic sweep for demo.*?    else:\n        proba  = df\["churn_proba"\]\.values\n        y_true = df\["churned"\]\.values if "churned" in df\.columns else \(proba > 0\.5\)\.astype\(int\)\n        thresholds = np\.linspace\(0\.10, 0\.90, 161\)\n', r'''    if "churn_proba" not in df.columns or df.empty:
        import streamlit as st
        st.error("Run notebook 02_baseline_model.ipynb to generate churn probabilities.")
        st.stop()
        
    proba  = df["churn_proba"].values
    y_true = df["churned"].values if "churned" in df.columns else (proba > 0.5).astype(int)
    thresholds = np.linspace(0.10, 0.90, 161)
''', content, flags=re.DOTALL)

open('app/pages/retention_simulator.py', 'w', encoding='utf-8').write(content)

# 4. Update uplift_targeting.py
content = open('app/pages/uplift_targeting.py', 'r', encoding='utf-8').read()
content = content.replace("    In demo mode, synthetic scores are displayed on the home page only.\n", "")
open('app/pages/uplift_targeting.py', 'w', encoding='utf-8').write(content)

print("Streamlit demo mode stripped.")
