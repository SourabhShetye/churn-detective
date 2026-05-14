
import json

nb = {
 "cells": [],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

# Paste cell sources here
cells = [
  ("markdown", ["# 05 · Cost-Aware Threshold Tuning & CMO Deck Data\n", "\n",
    "**Inputs:** `data/processed/test.parquet`\n", "\n",
    "**Outputs:** `outputs/reports/cmo_deck_data.json`, figures, models"]),

  ("code", [
    "import sys, os, json\n",
    "sys.path.insert(0, os.path.abspath('..'))\n",
    "import warnings; warnings.filterwarnings('ignore')\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import joblib\n",
    "\n",
    "from src.evaluate import (\n",
    "    BusinessParams, threshold_sweep, optimal_threshold,\n",
    "    full_scorecard, revenue_at_risk_summary, build_action_table,\n",
    ")\n",
    "from src.segment import segment_summary\n",
    "\n",
    "NAVY='#111e35'; TEAL='#2dd4bf'; AMBER='#fbbf24'; ROSE='#fb7185'; SLATE='#94a3b8'\n",
    "plt.rcParams.update({\n",
    "    'figure.facecolor':NAVY,'axes.facecolor':NAVY,'axes.edgecolor':'#1a2d4e',\n",
    "    'axes.labelcolor':SLATE,'xtick.color':SLATE,'ytick.color':SLATE,\n",
    "    'text.color':'#cbd5e1','grid.color':'#1a2d4e','grid.linestyle':'--','grid.alpha':0.4,'figure.dpi':130,\n",
    "})\n",
    "FIG_DIR='../outputs/figures'; REPORT_DIR='../outputs/reports'; MODEL_DIR='../models'\n",
    "os.makedirs(FIG_DIR,exist_ok=True); os.makedirs(REPORT_DIR,exist_ok=True)\n",
    "print('Libraries loaded \u2713')"
  ]),

  ("code", [
    "test_df = pd.read_parquet('../data/processed/test.parquet')\n",
    "required = ['churned','churn_proba','segment_label','monthly_charges']\n",
    "missing = [c for c in required if c not in test_df.columns]\n",
    "if missing:\n",
    "    raise RuntimeError(f'Missing columns: {missing}. Run notebooks 02-04 first.')\n",
    "y_true  = test_df['churned'].values\n",
    "y_proba = test_df['churn_proba'].values\n",
    "has_uplift = 'uplift_score' in test_df.columns\n",
    "uplift_scores = test_df['uplift_score'].values if has_uplift else None\n",
    "print(f'Test set: {len(test_df):,} rows')\n",
    "print(f'Uplift scores available: {has_uplift}')\n",
    "print(f'Churn rate: {y_true.mean():.3f}')"
  ]),

  ("code", [
    "params = BusinessParams(\n",
    "    avg_monthly_revenue      = float(test_df['monthly_charges'].mean()),\n",
    "    avg_customer_lifetime_mo = 24,\n",
    "    retention_offer_cost     = 15.00,\n",
    "    retention_success_rate   = 0.30,\n",
    "    false_alarm_cost         = 15.00,\n",
    "    discount_rate_annual     = 0.10,\n",
    ")\n",
    "print(f'CLV: ${params.customer_lifetime_value:,.2f}')\n",
    "print(f'Avg monthly revenue: ${params.avg_monthly_revenue:.2f}')"
  ]),

  ("code", [
    "sweep_df = threshold_sweep(y_true, y_proba, params=params, n_steps=200)\n",
    "opt_t    = optimal_threshold(sweep_df, metric='net_value')\n",
    "opt_row  = sweep_df.loc[sweep_df['threshold'].sub(opt_t).abs().idxmin()]\n",
    "print(f'Optimal threshold   : {opt_t:.3f}')\n",
    "print(f'Customers contacted : {int(opt_row[\"n_contacted\"]):,}')\n",
    "print(f'Expected net value  : ${opt_row[\"net_value\"]:,.0f}')\n",
    "print(f'Revenue saved       : ${opt_row[\"revenue_saved\"]:,.0f}')\n",
    "print(f'Campaign cost       : ${opt_row[\"campaign_cost\"]:,.0f}')\n",
    "print(f'ROI                 : {opt_row[\"roi_pct\"]:.0f}%')"
  ]),

  ("code", [
    "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n",
    "fig.patch.set_facecolor(NAVY)\n",
    "ax = axes[0]; ax.set_facecolor(NAVY)\n",
    "ax.plot(sweep_df['threshold'], sweep_df['net_value']/1000, color=TEAL, lw=2.5)\n",
    "ax.fill_between(sweep_df['threshold'], sweep_df['net_value']/1000, 0,\n",
    "                where=sweep_df['net_value']>0, alpha=0.10, color=TEAL)\n",
    "ax.axvline(opt_t, color=ROSE, lw=1.5, linestyle='--', label=f'Optimal={opt_t:.2f}')\n",
    "ax.axvline(0.50, color=AMBER, lw=1.2, linestyle=':', label='Default=0.50')\n",
    "ax.axhline(0, color=SLATE, lw=0.8)\n",
    "ax.set_xlabel('Threshold'); ax.set_ylabel('Net Value ($K)')\n",
    "ax.set_title('Net Business Value vs Threshold', color='#f8fafc', fontsize=11)\n",
    "ax.legend(fontsize=9); ax.grid(alpha=0.3)\n",
    "ax2 = axes[1]; ax2.set_facecolor(NAVY)\n",
    "ax2.plot(sweep_df['threshold'], sweep_df['precision'], color=TEAL, lw=2, label='Precision')\n",
    "ax2.plot(sweep_df['threshold'], sweep_df['recall'], color=ROSE, lw=2, label='Recall')\n",
    "ax2.plot(sweep_df['threshold'], sweep_df['f1'], color=AMBER, lw=2, linestyle='--', label='F1')\n",
    "ax2.axvline(opt_t, color='#f8fafc', lw=1.2, linestyle='--', alpha=0.6)\n",
    "ax2.set_xlabel('Threshold'); ax2.set_ylabel('Score')\n",
    "ax2.set_title('Precision / Recall Trade-off', color='#f8fafc', fontsize=11)\n",
    "ax2.legend(fontsize=9); ax2.grid(alpha=0.3); ax2.set_ylim(0,1)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f'{FIG_DIR}/business_value_curve.png', bbox_inches='tight', dpi=130, facecolor=NAVY)\n",
    "plt.show()\n",
    "print('Saved business_value_curve.png')"
  ]),

  ("code", [
    "churners_test = test_df[(test_df['churned']==1) & (test_df['churn_proba']>=opt_t)].copy()\n",
    "seg_impact = churners_test.groupby('segment_label').agg(\n",
    "    n_targeted=('customer_id','count'),\n",
    "    monthly_rev_at_risk=('monthly_charges','sum'),\n",
    "    avg_monthly_rev=('monthly_charges','mean'),\n",
    "    avg_churn_proba=('churn_proba','mean'),\n",
    ").reset_index()\n",
    "seg_impact['expected_saves']  = (seg_impact['n_targeted'] * params.retention_success_rate).round(0)\n",
    "seg_impact['revenue_saved']   = (seg_impact['expected_saves'] * seg_impact['avg_monthly_rev'] * params.avg_customer_lifetime_mo).round(0)\n",
    "seg_impact['campaign_cost']   = (seg_impact['n_targeted'] * params.retention_offer_cost).round(0)\n",
    "seg_impact['net_value']       = (seg_impact['revenue_saved'] - seg_impact['campaign_cost']).round(0)\n",
    "seg_impact['roi_pct']         = (seg_impact['net_value'] / seg_impact['campaign_cost'] * 100).round(0)\n",
    "print(seg_impact[['segment_label','n_targeted','expected_saves','revenue_saved','net_value','roi_pct']].to_string(index=False))"
  ]),

  ("code", [
    "scorecard    = full_scorecard(y_true, y_proba, threshold=opt_t, model_name='LightGBM')\n",
    "rev_summary  = revenue_at_risk_summary(\n",
    "    test_df, churn_proba_col='churn_proba',\n",
    "    monthly_charges_col='monthly_charges',\n",
    "    threshold=opt_t, params=params\n",
    ")\n",
    "action_table = build_action_table(\n",
    "    df_meta       = test_df[['customer_id','monthly_charges','segment_label']].copy(),\n",
    "    y_proba       = y_proba,\n",
    "    uplift_scores = uplift_scores,\n",
    "    threshold     = opt_t,\n",
    "    params        = params,\n",
    ")\n",
    "print(f'Action table: {len(action_table):,} rows')\n",
    "action_table.head()"
  ]),

  ("code", [
    "def safe(val):\n",
    "    import numpy as np\n",
    "    if isinstance(val, (np.integer,)): return int(val)\n",
    "    if isinstance(val, (np.floating,)): return float(round(val,4))\n",
    "    if isinstance(val, (np.bool_,)): return bool(val)\n",
    "    return val\n",
    "\n",
    "deck_data = {\n",
    "    'metadata': {'model':'LightGBM v1','test_set_rows':len(test_df)},\n",
    "    'slide_1_problem': {\n",
    "        'churn_rate_pct'          : safe(y_true.mean()*100),\n",
    "        'customers_at_risk'       : safe(rev_summary['n_at_risk']),\n",
    "        'monthly_revenue_at_risk' : safe(rev_summary['monthly_revenue_at_risk']),\n",
    "        'annual_revenue_at_risk'  : safe(rev_summary['annual_revenue_at_risk']),\n",
    "    },\n",
    "    'slide_5_measurement': {\n",
    "        'model_auc_roc'       : safe(scorecard['auc_roc'].iloc[0]),\n",
    "        'model_auc_pr'        : safe(scorecard['auc_pr'].iloc[0]),\n",
    "        'optimal_threshold'   : safe(opt_t),\n",
    "        'campaign_net_value'  : safe(opt_row['net_value']),\n",
    "        'campaign_roi_pct'    : safe(opt_row['roi_pct']),\n",
    "        'expected_saves'      : safe(rev_summary['expected_saves']),\n",
    "    },\n",
    "    'business_params': {\n",
    "        'avg_monthly_revenue'     : safe(params.avg_monthly_revenue),\n",
    "        'customer_lifetime_value' : safe(params.customer_lifetime_value),\n",
    "        'retention_offer_cost'    : params.retention_offer_cost,\n",
    "        'retention_success_rate'  : params.retention_success_rate,\n",
    "    },\n",
    "}\n",
    "\n",
    "out_path = f'{REPORT_DIR}/cmo_deck_data.json'\n",
    "with open(out_path, 'w') as f:\n",
    "    json.dump(deck_data, f, indent=2, default=safe)\n",
    "print(f'Saved {out_path}')"
  ]),

  ("code", [
    "joblib.dump(sweep_df,     f'{MODEL_DIR}/threshold_sweep_v1.pkl')\n",
    "joblib.dump(action_table, f'{MODEL_DIR}/action_table_v1.pkl')\n",
    "print('Saved threshold_sweep_v1.pkl')\n",
    "print('Saved action_table_v1.pkl')\n",
    "print('\\n' + '\u2501'*50)\n",
    "print('\u2713 Notebook 05 complete.')\n",
    "print('  Launch: streamlit run app/streamlit_app.py')\n",
    "print('\u2501'*50)"
  ]),
]

for ctype, source in cells:
    nb["cells"].append({
        "cell_type": ctype,
        "metadata": {},
        "source": source,
        **({"outputs": [], "execution_count": None} if ctype == "code" else {})
    })

with open('notebooks/05_cost_threshold.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Notebook rebuilt cleanly.')
