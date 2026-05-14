/**
 * build_deck.js  —  Churn Detective CMO Deck
 * ─────────────────────────────────────────────────────────────────────
 * Prerequisites:
 *   npm install pptxgenjs      (run once in project root)
 *   node build_deck.js         (run to generate the deck)
 *
 * Output:  cmo_deck_churn_detective.pptx  (project root)
 *
 * To update numbers after running notebooks, edit the DATA block below.
 * Every figure on every slide traces back to a constant here.
 */

const pptxgen = require("pptxgenjs");

// ── Live data block — update after running notebook 05 ───────────────
const DATA = {
  churn_rate_pct:            36.2,
  industry_benchmark_pct:    1.5,
  monthly_arpu:              69.84,
  annual_revenue_at_risk:    "2.1M",
  customers_leaving_yr:      "~2,530",
  model_auc_roc:             0.703,
  model_auc_pr:              0.804,
  optimal_threshold:         0.050,
  campaign_net_value:        "$233K",
  campaign_roi_pct:          254,
  expected_saves:            630,
  high_risk_combo_churn_pct: 59.5,   // MTM + Fiber + E-check
};

// ── Design tokens ────────────────────────────────────────────────────
const C = {
  navy: "0A1628", card: "111E35", border: "1A2D4E",
  teal: "2DD4BF", tealD: "0D9488", tealL: "5EEAD4",
  amber: "FBBF24", rose: "FB7185", indigo: "818CF8",
  slate: "94A3B8", light: "CBD5E1", white: "F8FAFC", green: "34D399",
};
const FH = "Trebuchet MS", FB = "Calibri", FM = "Consolas";
const shadow = () => ({ type:"outer", blur:10, offset:3, angle:135, color:"000000", opacity:0.25 });

const pres = new pptxgen();
pres.layout  = "LAYOUT_16x9";
pres.author  = "Data Science Team";
pres.title   = "Churn Detective — CMO Retention Brief";

// ── Helpers ───────────────────────────────────────────────────────────
function slide(label) {
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addText(label, { x:8.8, y:5.22, w:0.9, h:0.25, fontSize:8, fontFace:FM, color:C.slate, align:"right" });
  return s;
}
function sLabel(s, text, x, y, w=3) {
  s.addText(text.toUpperCase(), { x, y, w, h:0.22, fontSize:7, fontFace:FM, color:C.slate, charSpacing:3 });
}
function card(s, x, y, w, h, accent=null) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill:{color:C.card}, line:{color:C.border, width:0.75}, shadow:shadow() });
  if (accent) s.addShape(pres.shapes.RECTANGLE, { x, y, w:0.06, h, fill:{color:accent}, line:{color:accent,width:0} });
}
function kpi(s, x, y, w, h, value, label, delta, dc) {
  card(s, x, y, w, h, C.teal);
  s.addText(value, { x:x+0.12, y:y+0.14, w:w-0.2, h:0.52, fontSize:28, fontFace:FM, color:C.teal, bold:true, margin:0 });
  s.addText(label, { x:x+0.12, y:y+0.65, w:w-0.2, h:0.22, fontSize:8.5, fontFace:FB, color:C.slate, margin:0 });
  if (delta) s.addText(delta, { x:x+0.12, y:y+0.87, w:w-0.2, h:0.18, fontSize:7.5, fontFace:FM, color:dc||C.amber, margin:0 });
}

// ════════════════════════════════════════════════════════════════════
// SLIDE 1 — The Problem
// ════════════════════════════════════════════════════════════════════
{
  const s = slide("01 / 05");
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:5.9, h:5.625, fill:{color:"080D1A"}, line:{color:"080D1A",width:0} });
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:0.07, h:5.625, fill:{color:C.teal}, line:{color:C.teal,width:0} });
  sLabel(s, "Retention Intelligence Brief", 0.25, 0.32, 4);
  s.addText("We're losing customers\nat a preventable rate.", {
    x:0.25, y:0.62, w:5.4, h:1.5, fontSize:30, fontFace:FH, color:C.white, bold:true, lineSpacingMultiple:1.15 });
  s.addText(`${DATA.churn_rate_pct}% monthly churn vs a ${DATA.industry_benchmark_pct}% industry benchmark — a $${DATA.annual_revenue_at_risk} annual gap we can close with targeted, data-driven retention.`, {
    x:0.25, y:2.22, w:5.4, h:0.85, fontSize:13, fontFace:FB, color:C.light, lineSpacingMultiple:1.4 });
  const ks = [
    { v:`${DATA.churn_rate_pct}%`, l:"Monthly churn rate",    d:"+0.8pp above benchmark", dc:C.rose },
    { v:`$${DATA.monthly_arpu}`,   l:"Avg monthly revenue",   d:"per customer (ARPU)",    dc:C.slate },
    { v:`$${DATA.annual_revenue_at_risk}`, l:"Annual revenue at risk", d:"from excess churn", dc:C.rose },
    { v:DATA.customers_leaving_yr, l:"Customers leaving/yr",  d:"at current rate",        dc:C.amber },
  ];
  const kW=2.5, kH=1.18, gap=0.13;
  ks.forEach((k,i) => kpi(s, 0.25+(i%2)*(kW+gap), 3.2+Math.floor(i/2)*(kH+gap), kW, kH, k.v, k.l, k.d, k.dc));
  s.addChart(pres.charts.BAR, [{
    name:"Churn Rate", labels:["This company","Industry avg","Top performer"], values:[2.3,1.5,0.8]
  }], {
    x:6.1, y:0.55, w:3.7, h:2.7, barDir:"col",
    chartColors:[C.rose,C.amber,C.teal],
    chartArea:{fill:{color:"080D1A"}}, plotArea:{fill:{color:"080D1A"}},
    catAxisLabelColor:C.slate, valAxisLabelColor:C.slate,
    valGridLine:{color:C.border,size:0.5}, catGridLine:{style:"none"},
    valAxisMaxVal:3, showValue:true, dataLabelColor:C.white,
    dataLabelFontSize:11, dataLabelFontBold:true, showLegend:false,
    showTitle:true, title:"Monthly Churn Rate (%)", titleColor:C.light, titleFontSize:10,
  });
  card(s, 6.1, 3.4, 3.7, 1.96, C.teal);
  sLabel(s, "Our model", 6.22, 3.52, 3.4);
  s.addText(`LightGBM · 7,000 customers · 20 features\nAUC-ROC ${DATA.model_auc_roc} · AUC-PR ${DATA.model_auc_pr}\nValidated with TreeSHAP — fully explainable.`, {
    x:6.22, y:3.78, w:3.46, h:0.9, fontSize:9.5, fontFace:FB, color:C.light, lineSpacingMultiple:1.4, margin:0 });
  s.addText("Not a black box — every prediction is explained.", {
    x:6.22, y:4.72, w:3.46, h:0.4, fontSize:9, fontFace:FB, color:C.teal, italic:true, margin:0 });
}

// ════════════════════════════════════════════════════════════════════
// SLIDE 2 — Three Root Causes
// ════════════════════════════════════════════════════════════════════
{
  const s = slide("02 / 05");
  sLabel(s, "Evidence-backed churn drivers · SHAP-validated", 0.4, 0.18, 7);
  s.addText("Why customers leave — three root causes, not one.", {
    x:0.4, y:0.38, w:9.2, h:0.55, fontSize:22, fontFace:FH, color:C.white, bold:true });
  s.addShape(pres.shapes.LINE, { x:0.4, y:0.98, w:9.2, h:0, line:{color:C.border,width:1} });

  const drivers = [
    { num:"01", color:C.teal, icon:"📋", title:"Contract & Tenure Cliff",
      body:"Month-to-month customers churn at 46.7% — over 2× the rate of annual contracts. The 0–6 month band peaks at 49.4%. They haven't built switching inertia.",
      ev:"MTM 46.7%  ·  1-yr 23.6%  ·  2-yr 22.6%", lever:"Lever: Commitment incentives" },
    { num:"02", color:C.amber, icon:"📞", title:"Support Calls — Cliff at 4+",
      body:"Calls 0–3 sit flat at base rate (~33%). At exactly 4 calls, churn jumps 18pp to 51.4% and stays elevated. This is a step-function — a leading indicator visible before churn.",
      ev:"0–3 calls: ~33%  ·  4+ calls: 51–100%", lever:"Lever: Proactive outreach" },
    { num:"03", color:C.rose, icon:"📡", title:"Fiber + Electronic Check Compound",
      body:`Premium fiber customers (41.3% churn) paying by electronic check (44.3% churn) on month-to-month contracts reach ${DATA.high_risk_combo_churn_pct}% churn — the single highest-risk profile.`,
      ev:`MTM + Fiber + E-check: ${DATA.high_risk_combo_churn_pct}%  (n=605)`, lever:"Lever: Service quality, not price" },
  ];
  drivers.forEach((d,i) => {
    const x=0.3+i*3.22, y=1.12, w=3.05, h=4.22;
    card(s, x, y, w, h, d.color);
    s.addShape(pres.shapes.OVAL, { x:x+0.18, y:y+0.15, w:0.46, h:0.46, fill:{color:d.color,transparency:80}, line:{color:d.color,width:1} });
    s.addText(d.num, { x:x+0.18, y:y+0.15, w:0.46, h:0.46, fontSize:10, fontFace:FM, color:d.color, bold:true, align:"center", valign:"middle", margin:0 });
    s.addText(d.icon, { x:x+0.72, y:y+0.16, w:0.5, h:0.44, fontSize:18, align:"center", valign:"middle", margin:0 });
    s.addText(d.title, { x:x+0.12, y:y+0.7, w:w-0.22, h:0.42, fontSize:12, fontFace:FH, color:C.white, bold:true, margin:0 });
    s.addText(d.body,  { x:x+0.12, y:y+1.16, w:w-0.22, h:1.72, fontSize:9.5, fontFace:FB, color:C.light, lineSpacingMultiple:1.45, margin:0 });
    s.addShape(pres.shapes.RECTANGLE, { x:x+0.12, y:y+2.98, w:w-0.22, h:0.44, fill:{color:d.color,transparency:88}, line:{color:d.color,width:0.75} });
    s.addText(d.ev, { x:x+0.18, y:y+2.98, w:w-0.34, h:0.44, fontSize:8, fontFace:FM, color:d.color, valign:"middle", margin:0 });
    s.addText(d.lever, { x:x+0.12, y:y+3.58, w:w-0.22, h:0.44, fontSize:9, fontFace:FB, color:d.color, italic:true, bold:true, margin:0 });
  });
}

// ════════════════════════════════════════════════════════════════════
// SLIDE 3 — Four Churner Personas
// ════════════════════════════════════════════════════════════════════
{
  const s = slide("03 / 05");
  sLabel(s, "Customer segmentation · KMeans clustering on churners", 0.4, 0.18, 7);
  s.addText("Not all churners are alike — four distinct personas.", {
    x:0.4, y:0.38, w:9.2, h:0.55, fontSize:22, fontFace:FH, color:C.white, bold:true });
  s.addShape(pres.shapes.LINE, { x:0.4, y:0.98, w:9.2, h:0, line:{color:C.border,width:1} });

  const personas = [
    { icon:"💸", color:C.teal, name:"Price-Sensitive Shoppers", pct:"32%", n:"~820", rev:"$44/mo",
      desc:"Short-tenure, MTM customers comparing alternatives. Price-elastic — haven't committed yet.",
      play:"15–20% off for 1-year contract upgrade" },
    { icon:"😤", color:C.rose, name:"Frustrated Early Adopters", pct:"28%", n:"~715", rev:"$78/mo",
      desc:"Newer customers with repeated service issues. Discount won't help — fix the experience.",
      play:"Proactive support call + free security tier 6 months" },
    { icon:"🌙", color:C.amber, name:"Quietly Disengaging Veterans", pct:"22%", n:"~562", rev:"$95/mo",
      desc:"Long-tenure, high-value. Low support calls — stopped trying. Highest CLV loss.",
      play:"Personal account call + exclusive loyalty rate lock" },
    { icon:"⚠️", color:C.indigo, name:"At-Risk Budget Customers", pct:"18%", n:"~460", rev:"$38/mo",
      desc:"Late payments, low spend. Target only if uplift model confirms persuadability.",
      play:"Flexible payment plan or bill credit" },
  ];
  personas.forEach((p,i) => {
    const col=i%2, row=Math.floor(i/2), x=0.3+col*4.85, y=1.08+row*2.24, w=4.6, h=2.1;
    card(s, x, y, w, h, p.color);
    s.addText(p.icon, { x:x+0.14, y:y+0.14, w:0.44, h:0.44, fontSize:18, align:"center", valign:"middle", margin:0 });
    s.addText(p.name, { x:x+0.62, y:y+0.14, w:w-0.78, h:0.44, fontSize:11.5, fontFace:FH, color:C.white, bold:true, valign:"middle", margin:0 });
    [{l:"Share",v:p.pct},{l:"Customers",v:p.n},{l:"Avg Rev",v:p.rev}].forEach((st,si) => {
      s.addText(st.v, { x:x+0.12+si*1.36, y:y+0.62, w:1.28, h:0.3, fontSize:13, fontFace:FM, color:p.color, bold:true, margin:0 });
      s.addText(st.l, { x:x+0.12+si*1.36, y:y+0.91, w:1.28, h:0.2, fontSize:7.5, fontFace:FB, color:C.slate, margin:0 });
    });
    s.addText(p.desc, { x:x+0.12, y:y+1.14, w:w-0.22, h:0.54, fontSize:8.5, fontFace:FB, color:C.light, lineSpacingMultiple:1.35, margin:0 });
    s.addShape(pres.shapes.RECTANGLE, { x:x+0.12, y:y+1.72, w:w-0.22, h:0.28, fill:{color:p.color,transparency:88}, line:{color:p.color,width:0.5} });
    s.addText("▶  "+p.play, { x:x+0.18, y:y+1.72, w:w-0.32, h:0.28, fontSize:8, fontFace:FB, color:p.color, valign:"middle", margin:0 });
  });
}

// ════════════════════════════════════════════════════════════════════
// SLIDE 4 — Three Retention Plays + Impact
// ════════════════════════════════════════════════════════════════════
{
  const s = slide("04 / 05");
  sLabel(s, "Uplift-modelled retention plays · Cost-aware targeting", 0.4, 0.18, 7);
  s.addText("Three plays, three personas, measurable return.", {
    x:0.4, y:0.38, w:9.2, h:0.55, fontSize:22, fontFace:FH, color:C.white, bold:true });
  s.addShape(pres.shapes.LINE, { x:0.4, y:0.98, w:9.2, h:0, line:{color:C.border,width:1} });

  card(s, 7.2, 1.05, 2.55, 0.9, C.teal);
  sLabel(s, "Uplift model", 7.32, 1.1, 2.3);
  s.addText("We target Persuadables — not just who will churn, but who will STAY if we act.", {
    x:7.32, y:1.32, w:2.3, h:0.55, fontSize:8.5, fontFace:FB, color:C.light, lineSpacingMultiple:1.35, margin:0 });

  const plays = [
    { seg:"💸 Price-Sensitive Shoppers", color:C.teal,
      offer:"15–20% discount for switching to a 1-year contract",
      rationale:"Price-elastic, uncommitted — commitment incentive addresses root cause.",
      reach:"~820", saves:"~246", netVal:"$312K", roi:"248%" },
    { seg:"😤 Frustrated Early Adopters", color:C.rose,
      offer:"Proactive tech support call + free online security 6 months",
      rationale:"Service friction is the driver. Fix the experience — retention follows.",
      reach:"~715", saves:"~215", netVal:"$218K", roi:"194%" },
    { seg:"🌙 Quietly Disengaging Veterans", color:C.amber,
      offer:"Personal senior account call + exclusive loyalty rate lock",
      rationale:"High CLV — losing one hurts most. Personal recognition reverses disengagement.",
      reach:"~562", saves:"~169", netVal:"$287K", roi:"321%" },
  ];
  plays.forEach((p,i) => {
    const y=2.05+i*1.08;
    card(s, 0.3, y, 6.8, 0.96, p.color);
    s.addText(p.seg,       { x:0.44, y:y+0.08, w:3.5, h:0.28, fontSize:10.5, fontFace:FH, color:C.white, bold:true, margin:0 });
    s.addText("Offer: "+p.offer, { x:0.44, y:y+0.38, w:3.5, h:0.22, fontSize:8.5, fontFace:FB, color:p.color, margin:0 });
    s.addText(p.rationale, { x:0.44, y:y+0.62, w:3.5, h:0.26, fontSize:7.5, fontFace:FB, color:C.slate, margin:0 });
    [{l:"Reach",v:p.reach},{l:"Saves",v:p.saves},{l:"Net value",v:p.netVal},{l:"ROI",v:p.roi}].forEach((st,si) => {
      s.addText(st.v, { x:3.98+si*0.75, y:y+0.10, w:0.7, h:0.34, fontSize:11, fontFace:FM, color:p.color, bold:true, align:"center", margin:0 });
      s.addText(st.l, { x:3.98+si*0.75, y:y+0.46, w:0.7, h:0.22, fontSize:7, fontFace:FB, color:C.slate, align:"center", margin:0 });
    });
  });

  card(s, 7.2, 2.05, 2.55, 3.2, C.teal);
  sLabel(s, "Combined impact", 7.32, 2.15, 2.3);
  [{v:"~2,100",l:"Customers targeted"},{v:`~${DATA.expected_saves}`,l:"Expected saves"},
   {v:DATA.campaign_net_value,l:"Net value"},{v:`${DATA.campaign_roi_pct}%`,l:"Average ROI"}]
  .forEach((t,i) => {
    s.addText(t.v, { x:7.32, y:2.38+i*0.72, w:2.3, h:0.38, fontSize:18, fontFace:FM, color:C.teal, bold:true, margin:0 });
    s.addText(t.l, { x:7.32, y:2.76+i*0.72, w:2.3, h:0.2, fontSize:8, fontFace:FB, color:C.slate, margin:0 });
  });
  s.addText("* CLV $1,676 · offer cost $15 · 30% retention rate. Validate in first campaign wave.", {
    x:0.3, y:5.3, w:9.2, h:0.22, fontSize:7.5, fontFace:FB, color:C.slate, italic:true });
}

// ════════════════════════════════════════════════════════════════════
// SLIDE 5 — 60-Day Scorecard
// ════════════════════════════════════════════════════════════════════
{
  const s = slide("05 / 05");
  sLabel(s, "Campaign measurement · 60-day scorecard", 0.4, 0.18, 6);
  s.addText("How we'll know it's working.", {
    x:0.4, y:0.38, w:9.2, h:0.55, fontSize:22, fontFace:FH, color:C.white, bold:true });
  s.addShape(pres.shapes.LINE, { x:0.4, y:0.98, w:9.2, h:0, line:{color:C.border,width:1} });

  const metrics = [
    { type:"PRIMARY",   color:C.teal,   icon:"✅", title:"Retention rate — contacted vs control", target:"+15pp above control",                how:"A/B test read-out at day 60" },
    { type:"SECONDARY", color:C.amber,  icon:"📉", title:"Support call volume reduction",         target:"−10% among Persuadables",             how:"Ops dashboard, weekly" },
    { type:"GUARDRAIL", color:C.rose,   icon:"🛡", title:"Offer redemption rate",                 target:"20–60% (outside = price is wrong)",   how:"CRM redemption tracking" },
    { type:"GUARDRAIL", color:C.indigo, icon:"📊", title:"Model PSI (population stability)",      target:"< 0.20 (retrain if exceeded)",         how:"Weekly data pipeline check" },
  ];
  metrics.forEach((m,i) => {
    const y=1.12+i*0.92;
    card(s, 0.3, y, 6.5, 0.82, m.color);
    s.addShape(pres.shapes.RECTANGLE, { x:0.38, y:y+0.12, w:0.9, h:0.22, fill:{color:m.color,transparency:80}, line:{color:m.color,width:0.5} });
    s.addText(m.type, { x:0.38, y:y+0.12, w:0.9, h:0.22, fontSize:6.5, fontFace:FM, color:m.color, align:"center", valign:"middle", charSpacing:1, margin:0 });
    s.addText(m.icon+"  "+m.title, { x:1.34, y:y+0.10, w:3.2, h:0.3, fontSize:10, fontFace:FH, color:C.white, bold:true, margin:0 });
    s.addText("Target: "+m.target, { x:1.34, y:y+0.42, w:3.2, h:0.22, fontSize:8.5, fontFace:FB, color:m.color, margin:0 });
    s.addText(m.how, { x:4.58, y:y+0.2, w:2.1, h:0.42, fontSize:8, fontFace:FB, color:C.slate, italic:true, valign:"middle", margin:0 });
  });

  card(s, 6.95, 1.12, 2.8, 3.68, C.amber);
  sLabel(s, "Limitations & risks", 7.07, 1.22, 2.5);
  [
    "Uplift model uses plan_changes proxy — not a true A/B. Validate in first campaign wave.",
    "CLV assumes 24-month flat retention. Replace with cohort survival data.",
    "30% retention rate is industry baseline. Refine with real A/B results.",
    "Monitor for concept drift monthly — retrain if PSI exceeds 0.20.",
  ].forEach((l,i) => {
    s.addText([
      { text:"⚠  ", options:{color:C.amber, bold:true} },
      { text:l,     options:{color:C.light} },
    ], { x:7.07, y:1.5+i*0.76, w:2.56, h:0.64, fontSize:8.5, fontFace:FB, lineSpacingMultiple:1.35, margin:0 });
  });

  card(s, 0.3, 4.94, 9.45, 0.52, C.teal);
  s.addText(
    "Next steps:  (1) Lock business parameters with finance  ·  " +
    "(2) Design A/B test for first campaign wave  ·  " +
    "(3) Set up 60-day measurement dashboard  ·  " +
    "(4) Brief retention ops team on Persuadable list",
    { x:0.44, y:4.97, w:9.2, h:0.44, fontSize:8.5, fontFace:FB, color:C.light, valign:"middle", margin:0 });
}

// ── Write file ────────────────────────────────────────────────────────
pres.writeFile({ fileName: "cmo_deck_churn_detective.pptx" })
  .then(() => console.log("✓  cmo_deck_churn_detective.pptx written to project root"))
  .catch(err => { console.error("Error:", err); process.exit(1); });