"""
Streamlit Dashboard for Credit Card Fraud Detection
=====================================================
Run: streamlit run fraud_detection/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.fraud-badge {
    background: #E74C3C22;
    border: 1px solid #E74C3C;
    color: #E74C3C;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: bold;
}
.safe-badge {
    background: #2ECC7122;
    border: 1px solid #2ECC71;
    color: #2ECC71;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ─── Caching: Load & Train Model ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training fraud detection models...")
def load_detector():
    from fraud_detection.fraud_detector import RealTimeFraudDetector
    detector = RealTimeFraudDetector()
    detector.train(verbose=False)
    return detector


@st.cache_data(show_spinner="Generating sample transactions...")
def generate_demo_data(n=500):
    from fraud_detection.fraud_detector import TransactionDataGenerator
    return TransactionDataGenerator().generate(n_samples=n, fraud_rate=0.05)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/Fraud%20Detection-v1.0-red?style=for-the-badge", width=250)
st.sidebar.title("💳 Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🔍 Single Transaction",
    "📦 Batch Analysis",
    "📊 Model Performance",
    "⚙️ Model Info",
])

st.sidebar.markdown("---")
st.sidebar.caption("Models: Isolation Forest + Random Forest + XGBoost")


# ─── Load Model ──────────────────────────────────────────────────────────────
detector = load_detector()


# ─── Helper: Risk Badge ───────────────────────────────────────────────────────
RISK_COLORS = {
    'low': '#2ECC71', 'medium': '#F39C12',
    'high': '#E67E22', 'critical': '#E74C3C'
}

def risk_badge(risk_level, score):
    color = RISK_COLORS.get(risk_level, '#999')
    return f"""<div style="background:{color}22;border:1px solid {color};
    color:{color};padding:6px 16px;border-radius:20px;font-weight:bold;
    display:inline-block;">
    {risk_level.upper()} ({score:.1f}/100)</div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("Real-time transaction scoring using an ensemble of three ML models.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models", "3", "Isolation Forest + RF + XGBoost")
    col2.metric("Features", "18", "Engineered from raw transaction data")
    col3.metric("Fraud Rate", "~2%", "Typical real-world rate")
    col4.metric("Score Range", "0-100", "Higher = more suspicious")

    st.markdown("---")
    st.subheader("Architecture")

    arch_data = {
        'Component': ['Data Generator', 'Feature Engineering', 'Isolation Forest',
                       'Random Forest', 'XGBoost', 'Transaction Scorer', 'REST API'],
        'Role': ['Synthetic transaction data', 'Feature engineering & encoding',
                  'Anomaly detection (unsupervised)', 'Classification (supervised)',
                  'Gradient boost classification', 'Weighted ensemble scoring',
                  'FastAPI real-time endpoint'],
        'Weight': ['—', '—', '25%', '35%', '40%', '100%', '—'],
    }
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Sample Transactions")
    demo_df = generate_demo_data(200)
    fig = px.histogram(demo_df, x='amount', color='is_fraud',
                        color_discrete_map={0: '#3498DB', 1: '#E74C3C'},
                        labels={'is_fraud': 'Is Fraud', 'amount': 'Amount (USD)'},
                        title='Transaction Amount Distribution',
                        nbins=50, barmode='overlay', opacity=0.7)
    fig.update_layout(template='plotly_dark', paper_bgcolor='#0D1117', plot_bgcolor='#0D1117')
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Single Transaction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Single Transaction":
    st.title("🔍 Real-time Transaction Scoring")
    st.markdown("Enter transaction details to score it instantly.")

    MERCHANT_CATEGORIES = [
        'grocery', 'restaurant', 'gas_station', 'online_shopping',
        'entertainment', 'travel', 'healthcare', 'electronics',
        'clothing', 'atm_withdrawal'
    ]

    with st.form("transaction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Transaction")
            amount = st.number_input("Amount ($)", min_value=0.01, value=250.0, step=0.01)
            merchant_cat = st.selectbox("Merchant Category", MERCHANT_CATEGORIES)
            card_present = st.selectbox("Card Present", [1, 0], format_func=lambda x: "Yes" if x else "No")
            is_international = st.selectbox("International", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with col2:
            st.subheader("Time & Location")
            hour = st.slider("Hour of Day", 0, 23, 14)
            day = st.slider("Day of Week (0=Mon)", 0, 6, 2)
            distance = st.number_input("Distance from Home (km)", min_value=0.0, value=5.0, step=0.5)
            unusual_loc = st.selectbox("Unusual Location", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with col3:
            st.subheader("History")
            velocity = st.number_input("Txns in Last Hour", min_value=0, value=2, step=1)
            avg_amount = st.number_input("Avg Amount 7d ($)", min_value=0.01, value=80.0)
            txn_count = st.number_input("Txns Last 7 Days", min_value=0, value=15, step=1)

        submitted = st.form_submit_button("⚡ Score Transaction", use_container_width=True)

    if submitted:
        txn = {
            "amount": amount, "hour": hour, "day_of_week": day,
            "merchant_category": merchant_cat, "velocity_1h": int(velocity),
            "distance_from_home_km": distance, "is_international": is_international,
            "card_present": card_present, "unusual_location": unusual_loc,
            "avg_amount_7d": avg_amount, "transaction_count_7d": int(txn_count),
        }

        with st.spinner("Scoring transaction..."):
            result = detector.predict(txn)

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fraud Score", f"{result['fraud_score']:.1f}/100")
        col2.metric("Risk Level", result['risk_level'].upper())
        col3.metric("Decision", "FRAUD" if result['is_fraud'] else "LEGITIMATE")
        col4.metric("Latency", f"{result['processing_time_ms']:.1f}ms")

        st.markdown(risk_badge(result['risk_level'], result['fraud_score']), unsafe_allow_html=True)

        st.markdown("#### Model Breakdown")
        gauge_cols = st.columns(3)
        for col, (label, val) in zip(gauge_cols, [
            ("Isolation Forest", result['anomaly_score']),
            ("Random Forest",    result['rf_probability']),
            ("XGBoost",          result['xgb_probability']),
        ]):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': label, 'font': {'size': 13}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#E74C3C' if val > 50 else '#2ECC71'},
                    'steps': [
                        {'range': [0, 30],  'color': '#0D2B0D'},
                        {'range': [30, 60], 'color': '#2B1A0D'},
                        {'range': [60, 100],'color': '#2B0D0D'},
                    ],
                }
            ))
            fig.update_layout(height=200, margin=dict(t=30, b=0, l=0, r=0),
                               paper_bgcolor='#0D1117', font_color='#C9D1D9')
            col.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Risk Factors")
        for reason in result['explanation']:
            st.write(f"{'⚠️' if result['is_fraud'] else 'ℹ️'} {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Batch Analysis
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Batch Analysis":
    st.title("📦 Batch Transaction Analysis")

    n_txns = st.slider("Number of sample transactions", 100, 2000, 500, step=100)

    if st.button("🔄 Generate & Score Batch", use_container_width=True):
        with st.spinner("Generating and scoring transactions..."):
            demo_df = generate_demo_data(n_txns)
            scored_df = detector.predict_batch(demo_df)

        st.success(f"Scored {len(scored_df):,} transactions")

        flagged = scored_df['is_fraud_x'] if 'is_fraud_x' in scored_df.columns else scored_df['is_fraud']
        actual_fraud = demo_df['is_fraud']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{len(scored_df):,}")
        col2.metric("Flagged as Fraud", f"{scored_df['is_fraud'].sum():,}")
        col3.metric("Actual Fraud", f"{actual_fraud.sum():,}")
        col4.metric("Avg Score", f"{scored_df['fraud_score'].mean():.1f}")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(scored_df, x='fraud_score',
                                color='risk_level',
                                title='Score Distribution by Risk Level',
                                nbins=50)
            fig.update_layout(template='plotly_dark', paper_bgcolor='#0D1117')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            risk_counts = scored_df['risk_level'].value_counts()
            fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                          title='Risk Level Distribution',
                          color_discrete_sequence=['#2ECC71','#F39C12','#E67E22','#E74C3C'])
            fig.update_layout(template='plotly_dark', paper_bgcolor='#0D1117')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Flagged Transactions")
        top_flagged = scored_df.nlargest(20, 'fraud_score')[
            ['amount', 'merchant_category', 'fraud_score', 'risk_level',
             'rf_probability', 'xgb_probability', 'anomaly_score']
        ]
        st.dataframe(top_flagged.style.background_gradient(subset=['fraud_score'], cmap='RdYlGn_r'),
                      use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Metrics")

    from fraud_detection.fraud_detector import TransactionDataGenerator, FeatureEngineer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score

    @st.cache_data
    def compute_metrics():
        gen = TransactionDataGenerator()
        df = gen.generate(n_samples=5000, fraud_rate=0.02)
        fe = FeatureEngineer()
        df_feat = fe.fit_transform(df)
        X = df_feat[fe.feature_columns].values
        y = df_feat['is_fraud'].values
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        scores = {
            'Isolation Forest': detector.anomaly_detector.score(X_test),
            'Random Forest':    detector.rf_model.predict_proba(X_test),
            'XGBoost':          detector.xgb_model.predict_proba(X_test),
        }
        return X_test, y_test, scores

    with st.spinner("Computing metrics..."):
        X_test, y_test, scores = compute_metrics()

    rows = []
    for name, proba in scores.items():
        rows.append({
            'Model': name,
            'ROC-AUC': round(roc_auc_score(y_test, proba), 4),
            'Avg Precision': round(average_precision_score(y_test, proba), 4),
        })

    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('ROC-AUC', 'Avg Precision'))
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    for i, row in enumerate(rows):
        fig.add_trace(go.Bar(name=row['Model'], x=[row['Model']], y=[row['ROC-AUC']],
                              marker_color=colors[i], showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(name=row['Model'], x=[row['Model']], y=[row['Avg Precision']],
                              marker_color=colors[i], showlegend=False), row=1, col=2)
    fig.update_layout(template='plotly_dark', paper_bgcolor='#0D1117', height=400)
    st.plotly_chart(fig, use_container_width=True)

    if hasattr(detector.rf_model, 'feature_importances_') and detector.rf_model.feature_importances_ is not None:
        from fraud_detection.fraud_detector import FeatureEngineer
        fe2 = FeatureEngineer()
        feat_names = fe2.feature_columns
        fi = detector.rf_model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': fi})
        fi_df = fi_df.nlargest(15, 'Importance').sort_values('Importance')
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                         title='Top 15 Feature Importances (Random Forest)',
                         color='Importance', color_continuous_scale='Blues')
        fig_fi.update_layout(template='plotly_dark', paper_bgcolor='#0D1117')
        st.plotly_chart(fig_fi, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Model Info
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Model Info":
    st.title("⚙️ Model Configuration")

    st.subheader("Isolation Forest (Anomaly Detection)")
    st.markdown("""
- **Type**: Unsupervised anomaly detection
- **n_estimators**: 200 trees
- **contamination**: 2% (expected fraud rate)
- **Weight in ensemble**: 25%
- **Use case**: Catches novel fraud patterns without labels
""")

    st.subheader("Random Forest (Supervised)")
    st.markdown("""
- **Type**: Supervised classification
- **n_estimators**: 300 trees
- **class_weight**: balanced (handles class imbalance)
- **max_depth**: 12
- **Weight in ensemble**: 35%
- **Use case**: Stable, interpretable predictions
""")

    st.subheader("XGBoost (Gradient Boosting)")
    st.markdown("""
- **Type**: Supervised gradient boosted trees
- **n_estimators**: 500
- **learning_rate**: 0.05
- **scale_pos_weight**: 50 (handles extreme imbalance)
- **eval_metric**: aucpr (area under PR curve)
- **Weight in ensemble**: 40%
- **Use case**: Highest accuracy on structured tabular data
""")

    st.subheader("Ensemble Scoring")
    st.markdown("""
The final fraud score (0-100) is a weighted combination:

    score = 0.25 * IF_score + 0.35 * RF_proba + 0.40 * XGB_proba

| Score Range | Risk Level |
|-------------|-----------|
| 0-30        | Low       |
| 30-60       | Medium    |
| 60-80       | High      |
| 80-100      | Critical  |
""")

    st.subheader("18 Features Used")
    features = [
        ("amount", "Transaction amount in USD"),
        ("hour", "Hour of day (0-23)"),
        ("day_of_week", "Day of week (0=Monday)"),
        ("merchant_category_enc", "Encoded merchant category"),
        ("velocity_1h", "Number of transactions in last hour"),
        ("distance_from_home_km", "Distance from cardholder home"),
        ("is_international", "1 if cross-border transaction"),
        ("card_present", "1 if physical card was used"),
        ("unusual_location", "1 if unusual merchant location"),
        ("avg_amount_7d", "Average transaction amount (7 days)"),
        ("transaction_count_7d", "Transaction count (7 days)"),
        ("amount_vs_avg", "Current amount / 7-day average"),
        ("is_night", "1 if hour < 6 or > 22"),
        ("is_weekend", "1 if Saturday or Sunday"),
        ("high_velocity", "1 if velocity_1h > 5"),
        ("log_amount", "Log-scaled transaction amount"),
        ("log_distance", "Log-scaled distance from home"),
        ("manual_risk", "Hand-crafted composite risk score"),
    ]
    feat_df = pd.DataFrame(features, columns=['Feature', 'Description'])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
