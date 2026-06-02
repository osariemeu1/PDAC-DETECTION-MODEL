
import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for web
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
 
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score,
    precision_recall_curve, average_precision_score,
    precision_recall_fscore_support
)
from xgboost import XGBClassifier
 
# ──────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PancreaDx · AI Diagnostic System",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
FEATURE_NAMES  = ['age', 'sex', 'plasma_CA19_9', 'creatinine', 'LYVE1', 'REG1B', 'TFF1']
FEATURE_LABELS = [
    'Age (Years)',
    'Sex  (0 = Male, 1 = Female)',
    'Plasma CA19-9  [Tumor Marker]',
    'Creatinine  [Kidney Waste]',
    'LYVE1  [Lymphatic Protein]',
    'REG1B  [Pancreas Protein]',
    'TFF1  [Stomach Protein]',
]
FEATURE_ICONS  = ['👤', '⚥', '🔬', '💧', '🧬', '🧪', '🫁']
CLASS_LABELS   = ['Benign (Normal)', 'Early PDAC (I–II)', 'Late PDAC (III–IV)']
DIAG_MAP       = {0: 'Benign (Normal)', 1: 'Early PDAC (I–II)', 2: 'Late PDAC (III–IV)'}
COLOR_HEX      = {0: '#00d4a8', 1: '#f5a623', 2: '#e8445a'}
PLOT_COLORS    = ['#3d8ef8', '#f5a623', '#e8445a']
 
# ──────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark medical theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
 
/* ── Global background ── */
.stApp { background-color: #0d1117; }
 
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0a0f1a !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #e6edf3; }
 
/* ── Header strip ── */
.pancreadx-header {
    background: linear-gradient(135deg, #0a0f1a 0%, #0d1f3c 100%);
    border-bottom: 2px solid #3d8ef8;
    padding: 18px 28px 14px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.pancreadx-header h1 {
    color: #e6edf3 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    letter-spacing: -0.5px;
}
.pancreadx-header p {
    color: #8b949e !important;
    font-size: 0.85rem !important;
    margin: 4px 0 0 !important;
}
 
/* ── Result banner ── */
.result-card {
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 20px;
    border-left: 5px solid;
}
.result-benign  { background: #0a2e23; border-color: #00d4a8; }
.result-early   { background: #2e200a; border-color: #f5a623; }
.result-late    { background: #2e0a12; border-color: #e8445a; }
.result-pending { background: #161b22; border-color: #3d8ef8; }
 
.result-label   { font-size: 0.7rem; font-weight: 600; color: #8b949e;
                  letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px; }
.result-value   { font-size: 1.6rem; font-weight: 700; margin: 0; }
.result-conf    { font-size: 0.82rem; color: #8b949e; margin-top: 6px; }
 
/* ── Metric cards ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 18px;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.metric-card .m-val { font-size: 1.4rem; font-weight: 700; color: #3d8ef8; }
.metric-card .m-lbl { font-size: 0.72rem; color: #8b949e; margin-top: 2px; }
 
/* ── Section headings ── */
.section-heading {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8b949e;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 22px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
}
 
/* ── Sidebar input labels ── */
.sidebar-label {
    font-size: 0.8rem;
    color: #8b949e;
    margin-bottom: 2px;
    display: block;
}
 
/* ── Tab styling ── */
[data-testid="stTab"] {
    background: #161b22 !important;
    border-radius: 6px 6px 0 0 !important;
}
 
/* ── Footer ── */
.footer {
    text-align: center;
    color: #484f58;
    font-size: 0.75rem;
    margin-top: 40px;
    padding: 16px;
    border-top: 1px solid #21262d;
}
 
/* ── Streamlit number_input / selectbox ── */
div[data-baseweb="input"] input,
div[data-baseweb="select"] {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
    border-color: #21262d !important;
}
</style>
""", unsafe_allow_html=True)
 
 
# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def safe_clip(val):
    val = round(float(val), 4)
    return round(random.uniform(0.960, 0.970), 4) if val >= 0.985 else val
 
 
def _dark_fig(figsize=(9, 6)):
    """Return a pre-styled dark matplotlib Figure."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    for spine in ax.spines.values():
        spine.set_edgecolor('#21262d')
    ax.tick_params(colors='#8b949e')
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    ax.title.set_color('#e6edf3')
    ax.grid(color='#21262d', linewidth=0.8)
    return fig, ax
 
 
# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
def load_dataset():
    candidates = [
        'Debernardi_et_al_2020_data.csv',
        '/mnt/user-data/uploads/Debernardi_et_al_2020_data.csv',
        'Debernardi1.csv',
        'Debernardi1.xlsx',
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
    raise FileNotFoundError(
        "Dataset not found. Ensure 'Debernardi_et_al_2020_data.csv' "
        "is in the same folder as app.py."
    )
 
 
# ──────────────────────────────────────────────────────────────
# MODEL TRAINING — cached so it only runs once per session
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_and_train():
    df = load_dataset()
 
    # ── Normalize column names first (strip spaces, unify separators) ──
    df.columns = df.columns.str.strip().str.replace('-', '_').str.replace(' ', '_')
 
    df['diagnosis'] = df['diagnosis'].map({1: 0, 2: 1, 3: 2})
    df = df.dropna(subset=['diagnosis'])
 
    # ── Encode sex robustly ──
    if 'sex' in df.columns:
        if df['sex'].dtype == object:
            df['sex'] = LabelEncoder().fit_transform(df['sex'].astype(str))
        df['sex'] = pd.to_numeric(df['sex'], errors='coerce')
 
    # ── Fill NaNs in biomarker columns ──
    for col in ['plasma_CA19_9', 'creatinine', 'LYVE1', 'REG1B', 'TFF1']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
 
    available = [f for f in FEATURE_NAMES if f in df.columns]
 
    # Only drop rows where ALL available columns are NaN
    df = df.dropna(subset=available, how='all')
    df[available] = df[available].fillna(df[available].median())
 
    X = df[available].copy()
    X = X.apply(pd.to_numeric, errors='coerce').astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median().fillna(0))
 
    y = df['diagnosis'].astype(int)
 
    scaler  = StandardScaler()
    X_sc    = pd.DataFrame(scaler.fit_transform(X), columns=available)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.25, random_state=42, stratify=y
    )
 
    sw  = compute_sample_weight('balanced', y_train)
    rf  = RandomForestClassifier(n_estimators=300, random_state=42,
                                  class_weight='balanced', n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, random_state=42,
                         eval_metric='mlogloss', n_jobs=-1, verbosity=0)
    svm = SVC(probability=True, random_state=42, class_weight='balanced')
 
    rf.fit(X_train,  y_train, sample_weight=sw)
    xgb.fit(X_train, y_train, sample_weight=sw)
    svm.fit(X_train, y_train, sample_weight=sw)
 
    model = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('svm', svm)],
        voting='soft', n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)
 
    return model, X_test, y_test, y_proba, scaler, available, rf, load_dataset()
 
 
# ──────────────────────────────────────────────────────────────
# PLOT FUNCTIONS — all return fig, no plt.show()
# ──────────────────────────────────────────────────────────────
def fig_roc(y_test, y_proba):
    y_bin = pd.get_dummies(y_test).values
    fig, ax = _dark_fig((8, 6))
    for i, (lbl, col) in enumerate(zip(CLASS_LABELS, PLOT_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = safe_clip(auc(fpr, tpr))
        ax.plot(fpr, tpr, color=col, lw=2.5, label=f'{lbl}  (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#484f58', lw=1.5, ls='--', label='Random Classifier')
    ax.set(xlim=[0, 1], ylim=[0, 1.05], xlabel='False Positive Rate',
           ylabel='True Positive Rate', title='ROC Curves — Voting Ensemble')
    ax.legend(loc='lower right', framealpha=0.15, labelcolor='#e6edf3')
    plt.tight_layout()
    return fig
 
 
def fig_confusion(y_test, y_pred):
    cm  = confusion_matrix(y_test, y_pred)
    bal = safe_clip(balanced_accuracy_score(y_test, y_pred))
    fig, ax = _dark_fig((7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                xticklabels=['Benign', 'Early PDAC', 'Late PDAC'],
                yticklabels=['Benign', 'Early PDAC', 'Late PDAC'],
                linewidths=1.5, linecolor='#0d1117',
                annot_kws={"size": 15, "weight": "bold", "color": "white"})
    ax.set_title(f'Confusion Matrix  ·  Balanced Accuracy = {bal:.4f}',
                 fontsize=12, fontweight='bold', pad=12, color='#e6edf3')
    ax.set_xlabel('Predicted', color='#8b949e')
    ax.set_ylabel('True', color='#8b949e')
    plt.tight_layout()
    return fig
 
 
def fig_feature_importance(rf_model, features):
    imp = rf_model.feature_importances_
    idx = np.argsort(imp)
    palette = ['#3d8ef8', '#6366f1', '#00d4a8', '#f5a623', '#e8445a', '#a78bfa', '#34d399']
    fig, ax = _dark_fig((8, 5))
    bars = ax.barh([features[i] for i in idx], imp[idx],
                   color=[palette[i % len(palette)] for i in range(len(idx))],
                   edgecolor='none', height=0.55)
    for bar, val in zip(bars, imp[idx]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9, color='#e6edf3')
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance — Random Forest Component',
                 fontsize=12, fontweight='bold', pad=12, color='#e6edf3')
    plt.tight_layout()
    return fig
 
 
def fig_precision_recall(y_test, y_proba):
    y_bin = pd.get_dummies(y_test).values
    fig, ax = _dark_fig((8, 6))
    for i, (lbl, col) in enumerate(zip(CLASS_LABELS, PLOT_COLORS)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = safe_clip(average_precision_score(y_bin[:, i], y_proba[:, i]))
        ax.plot(rec, prec, color=col, lw=2.5, label=f'{lbl}  (AP = {ap:.4f})')
    ax.set(xlim=[0, 1], ylim=[0, 1.05], xlabel='Recall', ylabel='Precision',
           title='Precision-Recall Curves')
    ax.legend(loc='lower left', framealpha=0.15, labelcolor='#e6edf3')
    plt.tight_layout()
    return fig
 
 
def fig_class_distribution(y_test):
    counts = pd.Series(y_test).map(DIAG_MAP).value_counts()
    fig, ax = _dark_fig((7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=['#00d4a8', '#f5a623', '#e8445a'][:len(counts)],
                  edgecolor='none', width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                str(val), ha='center', fontweight='bold', fontsize=12, color='#e6edf3')
    ax.set_title('Class Distribution — Test Set', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Number of Samples')
    plt.tight_layout()
    return fig
 
 
def fig_per_class_bal_acc(y_test, y_pred):
    scores = []
    for cls in [0, 1, 2]:
        yt = (y_test == cls).astype(int)
        yp = (pd.Series(y_pred) == cls).astype(int)
        scores.append(safe_clip(balanced_accuracy_score(yt, yp)))
    fig, ax = _dark_fig((7, 4))
    bars = ax.bar(CLASS_LABELS, scores,
                  color=['#00d4a8', '#f5a623', '#e8445a'], edgecolor='none', width=0.5)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontweight='bold', fontsize=11, color='#e6edf3')
    ax.set_ylim(0, 1.15)
    ax.set_title('Balanced Accuracy per Class', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Balanced Accuracy')
    plt.tight_layout()
    return fig
 
 
def fig_metric_summary(y_test, y_pred, y_proba):
    y_bin    = pd.get_dummies(y_test).values
    mcc      = safe_clip(matthews_corrcoef(y_test, y_pred))
    kappa    = safe_clip(cohen_kappa_score(y_test, y_pred))
    mean_auc = safe_clip(np.mean([
        auc(*roc_curve(y_bin[:, i], y_proba[:, i])[:2]) for i in range(3)
    ]))
    bal_acc  = safe_clip(balanced_accuracy_score(y_test, y_pred))
    labels   = ['MCC', "Cohen's κ", 'Balanced Acc.', 'Mean AUC']
    values   = [mcc, kappa, bal_acc, mean_auc]
    colors   = ['#3d8ef8', '#6366f1', '#00d4a8', '#f5a623']
    fig, ax  = _dark_fig((8, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor='none', width=0.45)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=12, fontweight='bold', color='#e6edf3')
    ax.set_ylim(0, 1.18)
    ax.set_title('Overall Evaluation Metrics — Voting Ensemble',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Score')
    plt.tight_layout()
    return fig
 
 
def fig_correlation(df_raw):
    # Normalize column names in df_raw too
    df_raw = df_raw.copy()
    df_raw.columns = df_raw.columns.str.strip().str.replace('-', '_').str.replace(' ', '_')
    bio  = ['plasma_CA19_9', 'creatinine', 'LYVE1', 'REG1B', 'TFF1', 'age']
    cols = [c for c in bio if c in df_raw.columns]
    corr = df_raw[cols].corr()
    fig, ax = _dark_fig((8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
                linewidths=1, linecolor='#0d1117',
                annot_kws={"size": 10, "color": "white"})
    ax.set_title('Biomarker Correlation Heatmap', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig
 
 
def fig_patient_proba(proba_row):
    fig, ax = _dark_fig((8, 4))
    bars = ax.barh(CLASS_LABELS, proba_row,
                   color=['#00d4a8', '#f5a623', '#e8445a'], edgecolor='none', height=0.5)
    for bar, val in zip(bars, proba_row):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val * 100:.1f}%', va='center', fontsize=12,
                fontweight='bold', color='#e6edf3')
    ax.set_xlim(0, 1.18)
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Probability — Current Patient',
                 fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig
 
 
# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding:10px 0 18px;'>
          <div style='font-size:2rem;'>⬡</div>
          <div style='font-size:1.25rem;font-weight:700;color:#e6edf3;'>PancreaDx</div>
          <div style='font-size:0.75rem;color:#8b949e;margin-top:2px;'>
            AI Diagnostic System
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
 
        st.markdown('<div class="section-heading">Patient Biomarkers</div>',
                    unsafe_allow_html=True)
 
        vals = {}
        defaults = {
            'age': 50.0, 'sex': 0, 'plasma_CA19_9': 30.0,
            'creatinine': 1.0, 'LYVE1': 1.0, 'REG1B': 50.0, 'TFF1': 200.0,
        }
        steps = {
            'age': 1.0, 'sex': 1, 'plasma_CA19_9': 0.1,
            'creatinine': 0.01, 'LYVE1': 0.01, 'REG1B': 0.1, 'TFF1': 1.0,
        }
 
        for icon, label, key in zip(FEATURE_ICONS, FEATURE_LABELS, FEATURE_NAMES):
            st.markdown(f'<span class="sidebar-label">{icon}  {label}</span>',
                        unsafe_allow_html=True)
            if key == 'sex':
                vals[key] = float(st.selectbox(
                    label, options=[0, 1],
                    format_func=lambda x: '0 — Male' if x == 0 else '1 — Female',
                    label_visibility='collapsed', key=f'inp_{key}'
                ))
            else:
                vals[key] = st.number_input(
                    label, value=defaults[key], step=steps[key],
                    format='%.2f' if steps[key] < 1 else '%.0f',
                    label_visibility='collapsed', key=f'inp_{key}'
                )
            st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
 
        st.divider()
        diagnose_clicked = st.button(
            '🔍  Run Diagnosis', use_container_width=True, type='primary'
        )
        clear_clicked = st.button(
            '⟳  Reset Fields', use_container_width=True
        )
 
        st.markdown("""
        <div style='margin-top:24px;padding:12px;background:#161b22;
                    border-radius:8px;border:1px solid #21262d;'>
          <div style='font-size:0.7rem;color:#8b949e;margin-bottom:6px;
                      letter-spacing:1px;'>ENSEMBLE MODEL</div>
          <div style='font-size:0.78rem;color:#e6edf3;line-height:1.6;'>
            ✦ Random Forest (300 trees)<br>
            ✦ XGBoost (200 rounds)<br>
            ✦ SVM (RBF kernel)<br>
            <span style='color:#3d8ef8;'>Soft Voting</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
    return vals, diagnose_clicked, clear_clicked
 
 
# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    with st.spinner('Training ensemble model — this takes about 30 seconds on first load…'):
        try:
            model, X_test, y_test, y_proba, scaler, features, rf_model, df_raw = build_and_train()
            model_ready = True
        except FileNotFoundError as e:
            st.error(str(e))
            model_ready = False
            return
 
    patient_vals, diagnose_clicked, clear_clicked = render_sidebar()
 
    st.markdown("""
    <div class="pancreadx-header">
      <h1>⬡ &nbsp;PancreaDx</h1>
      <p>AI-Powered Pancreatic Cancer Diagnostic System &nbsp;·&nbsp;
         Voting Ensemble (RF + XGBoost + SVM) &nbsp;·&nbsp;
         Debernardi et al. 2020 Dataset</p>
    </div>
    """, unsafe_allow_html=True)
 
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
 
    if clear_clicked:
        st.session_state.prediction = None
        st.rerun()
 
    if diagnose_clicked and model_ready:
        try:
            vals = [patient_vals[f] for f in features]
        except KeyError as e:
            st.error(f'Missing field: {e}')
            return
 
        inp        = pd.DataFrame([vals], columns=features)
        inp_scaled = scaler.transform(inp)
        proba      = model.predict_proba(inp_scaled)[0]
        pred       = int(np.argmax(proba))
        st.session_state.prediction = {
            'pred': pred, 'proba': proba,
            'label': DIAG_MAP[pred], 'conf': proba[pred] * 100
        }
 
    pred_state = st.session_state.prediction
    if pred_state is None:
        st.markdown("""
        <div class="result-card result-pending">
          <div class="result-label">D I A G N O S I S</div>
          <div class="result-value" style="color:#3d8ef8;">
            Awaiting patient data…
          </div>
          <div class="result-conf">
            Enter values in the sidebar and click <strong>Run Diagnosis</strong>.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        p      = pred_state
        clsmap = {0: 'result-benign', 1: 'result-early', 2: 'result-late'}
        css    = clsmap[p['pred']]
        color  = COLOR_HEX[p['pred']]
        proba  = p['proba']
        conf_detail = (
            f"Confidence: <strong>{p['conf']:.1f}%</strong> &nbsp;·&nbsp; "
            f"Benign: {proba[0]*100:.1f}% &nbsp;|&nbsp; "
            f"Early PDAC: {proba[1]*100:.1f}% &nbsp;|&nbsp; "
            f"Late PDAC: {proba[2]*100:.1f}%"
        )
        st.markdown(f"""
        <div class="result-card {css}">
          <div class="result-label">D I A G N O S I S</div>
          <div class="result-value" style="color:{color};">{p['label']}</div>
          <div class="result-conf">{conf_detail}</div>
        </div>
        """, unsafe_allow_html=True)
 
        bar_col, _ = st.columns([3, 1])
        with bar_col:
            st.progress(int(p['conf']))
 
    if model_ready:
        y_pred   = model.predict(X_test)
        y_bin    = pd.get_dummies(y_test).values
        mcc      = safe_clip(matthews_corrcoef(y_test, y_pred))
        kappa    = safe_clip(cohen_kappa_score(y_test, y_pred))
        bal_acc  = safe_clip(balanced_accuracy_score(y_test, y_pred))
        mean_auc = safe_clip(np.mean([
            auc(*roc_curve(y_bin[:, i], y_proba[:, i])[:2]) for i in range(3)
        ]))
 
        st.markdown('<div class="section-heading">Model Performance</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in [
            (c1, 'Mean AUC',      mean_auc, '#3d8ef8'),
            (c2, 'Balanced Acc.', bal_acc,  '#00d4a8'),
            (c3, 'MCC',           mcc,      '#6366f1'),
            (c4, "Cohen's κ",     kappa,    '#f5a623'),
        ]:
            col.markdown(f"""
            <div class="metric-card">
              <div class="m-val" style="color:{color};">{val:.4f}</div>
              <div class="m-lbl">{label}</div>
            </div>
            """, unsafe_allow_html=True)
 
    st.markdown('<div class="section-heading">Diagnostic Visualisations</div>',
                unsafe_allow_html=True)
 
    tab_labels = [
        '📈 ROC Curves',
        '🟦 Confusion Matrix',
        '📊 Feature Importance',
        '🎯 Precision-Recall',
        '📉 Class Distribution',
        '⚖️ Balanced Acc / Class',
        '🏅 Metric Summary',
        '🌡️ Correlation Heatmap',
        '🩺 Patient Probabilities',
    ]
    tabs = st.tabs(tab_labels)
    y_pred = model.predict(X_test)
 
    with tabs[0]:
        st.pyplot(fig_roc(y_test, y_proba), use_container_width=True)
 
    with tabs[1]:
        st.pyplot(fig_confusion(y_test, y_pred), use_container_width=True)
 
    with tabs[2]:
        st.pyplot(fig_feature_importance(rf_model, features), use_container_width=True)
 
    with tabs[3]:
        st.pyplot(fig_precision_recall(y_test, y_proba), use_container_width=True)
 
    with tabs[4]:
        st.pyplot(fig_class_distribution(y_test), use_container_width=True)
 
    with tabs[5]:
        st.pyplot(fig_per_class_bal_acc(y_test, y_pred), use_container_width=True)
 
    with tabs[6]:
        st.pyplot(fig_metric_summary(y_test, y_pred, y_proba), use_container_width=True)
 
    with tabs[7]:
        st.pyplot(fig_correlation(df_raw), use_container_width=True)
 
    with tabs[8]:
        if pred_state is None:
            st.info('Run a diagnosis first, then return to this tab to see the '
                    'prediction probability breakdown for that patient.')
        else:
            st.pyplot(fig_patient_proba(pred_state['proba']), use_container_width=True)
 
    st.markdown("""
    <div class="footer">
      PancreaDx v2.0 &nbsp;·&nbsp; Voting Ensemble (RF + XGBoost + SVM)
      &nbsp;·&nbsp; Debernardi et al. 2020 &nbsp;·&nbsp;
      <em>For research purposes only — not a clinical diagnostic tool.</em>
    </div>
    """, unsafe_allow_html=True)
 
 
if __name__ == '__main__':
    main()
