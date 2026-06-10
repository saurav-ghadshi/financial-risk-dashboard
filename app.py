"""
AI/ML Financial Risk Assessment Dashboard
==========================================
Flask backend — trains models, exposes REST API for the real-time dashboard.
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import warnings, os, json, time, random
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

np.random.seed(42)

# ──────────────────────────────────────────
#  DATA GENERATION
# ──────────────────────────────────────────
def generate_data(n=10000):
    df = pd.DataFrame()
    df["customer_id"] = range(1, n + 1)
    df["annual_income"]            = np.clip(np.random.gamma(3, 20, n) + 20, 15, 200)
    df["years_employed"]           = np.clip(np.random.exponential(5, n), 0, 40)
    df["total_debt"]               = df["annual_income"] * np.random.uniform(0, 0.8, n)
    df["debt_to_income_ratio"]     = df["total_debt"] / df["annual_income"]
    df["credit_score"]             = np.clip(np.random.normal(680, 100, n), 300, 850)
    df["payment_history_score"]    = np.random.beta(8, 2, n) * 100
    df["num_credit_lines"]         = np.random.poisson(3, n)
    df["num_defaults_past"]        = np.random.poisson(0.3, n)

    df["monthly_online_transactions"] = np.random.exponential(15, n)
    df["avg_transaction_value"]       = np.random.lognormal(4, 1, n)
    df["utility_payment_consistency"] = np.random.beta(7, 2, n)
    df["utility_payment_delay"]       = np.random.exponential(0.5, n)
    df["monthly_mobile_topups"]       = np.random.poisson(4, n)
    df["ecommerce_per_month"]         = np.random.exponential(2, n)
    df["online_activity_freq"]        = np.random.beta(4, 3, n)

    edu_levels   = ["High School", "Diploma", "Bachelor", "Master", "PhD"]
    edu_weights  = [0.15, 0.20, 0.45, 0.15, 0.05]
    df["education_level"] = np.random.choice(edu_levels, n, p=edu_weights)
    edu_enc = {"High School": 1, "Diploma": 2, "Bachelor": 3, "Master": 4, "PhD": 5}
    df["education_encoded"] = df["education_level"].map(edu_enc)

    prestige = []
    for e in df["education_level"]:
        if e == "High School": prestige.append(np.random.uniform(1, 3))
        elif e == "Diploma":   prestige.append(np.random.uniform(2, 4))
        elif e == "Bachelor":  prestige.append(np.random.uniform(3, 9))
        elif e == "Master":    prestige.append(np.random.uniform(5, 10))
        else:                  prestige.append(np.random.uniform(7, 10))
    df["institution_prestige"] = prestige

    fields        = ["Engineering", "Finance", "Business", "IT", "Liberal Arts", "Science", "Medicine"]
    field_weights = [0.18, 0.12, 0.15, 0.20, 0.15, 0.12, 0.08]
    df["field_of_study"] = np.random.choice(fields, n, p=field_weights)
    le = LabelEncoder()
    df["field_encoded"] = le.fit_transform(df["field_of_study"])

    df["years_since_grad"]    = np.clip(np.random.exponential(5, n), 0, 50)
    df["social_posts_week"]   = np.random.exponential(2, n)
    df["social_network_size"] = np.random.lognormal(4, 1.5, n)
    df["social_engagement"]   = np.random.beta(2, 5, n)
    df["tenure_months"]       = np.random.exponential(24, n)

    risk = (
        (1 - df["credit_score"] / 850) * 0.20 +
        df["debt_to_income_ratio"]      * 0.15 +
        (1 - df["payment_history_score"] / 100) * 0.15 +
        (1 - df["education_encoded"] / 5)       * 0.20 +
        (1 - df["utility_payment_consistency"])  * 0.10 +
        (1 - df["online_activity_freq"])         * 0.08 +
        (df["num_defaults_past"] / (df["num_defaults_past"].max() + 1)) * 0.12
    )
    risk = (risk - risk.min()) / (risk.max() - risk.min())
    prob = 0.15 + risk * 0.85
    df["default"] = (np.random.random(n) < prob).astype(int)

    return df


# ──────────────────────────────────────────
#  PREPROCESSING
# ──────────────────────────────────────────
FEATURES = [
    "annual_income", "years_employed", "debt_to_income_ratio", "credit_score",
    "payment_history_score", "num_credit_lines", "num_defaults_past",
    "monthly_online_transactions", "avg_transaction_value",
    "utility_payment_consistency", "utility_payment_delay",
    "monthly_mobile_topups", "ecommerce_per_month", "online_activity_freq",
    "education_encoded", "institution_prestige", "field_encoded",
    "years_since_grad", "social_posts_week", "social_network_size",
    "social_engagement", "tenure_months",
]

def preprocess(df):
    scaler = StandardScaler()
    X = df[FEATURES].copy().astype(float)
    for col in X.columns:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        IQR = Q3 - Q1
        X[col] = X[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
    scaled = scaler.fit_transform(X)
    X = pd.DataFrame(scaled, columns=FEATURES, index=X.index)
    y = df["default"]
    return X, y, scaler


# ──────────────────────────────────────────
#  MODEL TRAINING
# ──────────────────────────────────────────
def train_models(X, y):
    sm = SMOTE(random_state=42, k_neighbors=5)
    Xb, yb = sm.fit_resample(X, y)
    Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.2, random_state=42, stratify=yb)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest":        RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight="balanced", n_jobs=-1),
        "XGBoost":              xgb.XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42, verbosity=0,
                                                   scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum()),
    }

    results = {}
    trained = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        yp   = m.predict(Xte)
        yprob= m.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, yprob)
        cm   = confusion_matrix(yte, yp).tolist()
        results[name] = {
            "accuracy":  round(accuracy_score(yte, yp),  4),
            "precision": round(precision_score(yte, yp), 4),
            "recall":    round(recall_score(yte, yp),    4),
            "f1":        round(f1_score(yte, yp),        4),
            "auc":       round(roc_auc_score(yte, yprob),4),
            "roc_fpr":   [round(v, 4) for v in fpr.tolist()[::20]],
            "roc_tpr":   [round(v, 4) for v in tpr.tolist()[::20]],
            "cm":        cm,
        }
        trained[name] = m

    # Feature importance from RF
    rf = trained["Random Forest"]
    fi = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])

    return results, trained, fi, Xte, yte


# ──────────────────────────────────────────
#  EDUCATION IMPACT ANALYSIS
# ──────────────────────────────────────────
def education_impact(df, trained_models, Xte, yte):
    edu_stats = df.groupby("education_level").agg(
        count=("default", "count"),
        defaults=("default", "sum"),
        default_rate=("default", "mean"),
        avg_income=("annual_income", "mean"),
        avg_credit=("credit_score", "mean"),
    ).round(3).reset_index().to_dict("records")

    xgb_model = trained_models["XGBoost"]
    Xte_no_edu = Xte.copy()
    edu_idx = [i for i, f in enumerate(FEATURES) if "education" in f or "prestige" in f or "field" in f]
    Xte_no_edu.iloc[:, edu_idx] = 0

    auc_with    = round(roc_auc_score(yte, xgb_model.predict_proba(Xte)[:, 1]), 4)
    auc_without = round(roc_auc_score(yte, xgb_model.predict_proba(Xte_no_edu)[:, 1]), 4)

    return {
        "edu_stats": edu_stats,
        "auc_with_edu":    auc_with,
        "auc_without_edu": auc_without,
        "improvement_pct": round((auc_with - auc_without) / auc_without * 100, 2),
    }


# ──────────────────────────────────────────
#  GLOBAL CACHE — trained once on startup
# ──────────────────────────────────────────
CACHE = {}

def boot():
    print("⚙  Generating data & training models …")
    t0 = time.time()
    df = generate_data(10000)
    X, y, scaler = preprocess(df)
    results, trained, fi, Xte, yte = train_models(X, y)
    edu = education_impact(df, trained, Xte, yte)

    CACHE["df"]       = df
    CACHE["results"]  = results
    CACHE["fi"]       = fi
    CACHE["edu"]      = edu
    CACHE["trained"]  = trained
    CACHE["scaler"]   = scaler
    CACHE["Xte"]      = Xte
    CACHE["yte"]      = yte
    print(f"✓  Ready in {time.time()-t0:.1f}s")

# ──────────────────────────────────────────
#  RUN boot() AT MODULE IMPORT TIME
#  (required for Gunicorn — it imports the
#   module directly, never runs __main__)
# ──────────────────────────────────────────
boot()

# ──────────────────────────────────────────
#  API ROUTES
# ──────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/api/summary")
def api_summary():
    df = CACHE["df"]
    return jsonify({
        "total_customers":  int(len(df)),
        "default_count":    int(df["default"].sum()),
        "non_default_count":int((df["default"] == 0).sum()),
        "default_rate":     round(float(df["default"].mean()) * 100, 2),
        "avg_income":       round(float(df["annual_income"].mean()), 2),
        "avg_credit_score": round(float(df["credit_score"].mean()), 2),
        "best_model":       max(CACHE["results"], key=lambda k: CACHE["results"][k]["auc"]),
        "best_auc":         max(v["auc"] for v in CACHE["results"].values()),
        "education_importance_pct": round(
            sum(imp for feat, imp in CACHE["fi"] if "education" in feat or "prestige" in feat or "field" in feat) * 100, 2
        ),
    })

@app.route("/api/models")
def api_models():
    return jsonify(CACHE["results"])

@app.route("/api/feature_importance")
def api_fi():
    return jsonify([{"feature": f, "importance": round(float(i), 5)} for f, i in CACHE["fi"][:20]])

@app.route("/api/education")
def api_education():
    return jsonify(CACHE["edu"])

@app.route("/api/distribution")
def api_distribution():
    df = CACHE["df"]
    income_bins  = np.histogram(df["annual_income"], bins=20)
    credit_bins  = np.histogram(df["credit_score"],  bins=20)
    return jsonify({
        "income_hist":  {"values": income_bins[0].tolist(), "edges": [round(e,1) for e in income_bins[1].tolist()]},
        "credit_hist":  {"values": credit_bins[0].tolist(), "edges": [round(e,1) for e in credit_bins[1].tolist()]},
        "education_counts": df["education_level"].value_counts().to_dict(),
        "field_counts":     df["field_of_study"].value_counts().to_dict(),
    })

@app.route("/api/compare")
def api_compare():
    """Side-by-side traditional vs AI model comparison."""
    r = CACHE["results"]
    traditional = r["Logistic Regression"]
    best_name   = max(r, key=lambda k: r[k]["auc"])
    best        = r[best_name]
    return jsonify({
        "traditional": {"name": "Logistic Regression (Traditional)", **traditional},
        "ai":          {"name": f"{best_name} (AI Model)",           **best},
        "improvement": {
            "auc":      round((best["auc"]      - traditional["auc"])      / traditional["auc"]      * 100, 2),
            "accuracy": round((best["accuracy"] - traditional["accuracy"]) / traditional["accuracy"] * 100, 2),
            "f1":       round((best["f1"]       - traditional["f1"])       / traditional["f1"]       * 100, 2),
        },
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Live single-customer prediction endpoint."""
    data     = request.json
    model_nm = data.get("model", "Random Forest")
    model    = CACHE["trained"].get(model_nm)
    scaler   = CACHE["scaler"]
    if not model:
        return jsonify({"error": "Unknown model"}), 400

    # Build feature vector with defaults for missing fields
    edu_enc = {"High School": 1, "Diploma": 2, "Bachelor": 3, "Master": 4, "PhD": 5}
    field_map= {"Engineering": 0, "Finance": 1, "Business": 2, "IT": 3,
                "Liberal Arts": 4, "Science": 5, "Medicine": 6}
    row = [
        float(data.get("annual_income",    60)),
        float(data.get("years_employed",   5)),
        float(data.get("debt_to_income",   0.3)),
        float(data.get("credit_score",     650)),
        float(data.get("payment_history",  75)),
        float(data.get("credit_lines",     3)),
        float(data.get("past_defaults",    0)),
        float(data.get("online_tx",        10)),
        float(data.get("avg_tx_value",     100)),
        float(data.get("utility_consist",  0.8)),
        float(data.get("utility_delay",    0.2)),
        float(data.get("mobile_topups",    4)),
        float(data.get("ecommerce",        2)),
        float(data.get("online_activity",  0.6)),
        float(edu_enc.get(data.get("education", "Bachelor"), 3)),
        float(data.get("institution_prestige", 6)),
        float(field_map.get(data.get("field", "Business"), 2)),
        float(data.get("years_since_grad", 5)),
        float(data.get("social_posts",     3)),
        float(data.get("social_network",   200)),
        float(data.get("social_engage",    0.3)),
        float(data.get("tenure_months",    24)),
    ]
    X_input = scaler.transform([row])
    prob    = float(model.predict_proba(X_input)[0][1])
    label   = int(model.predict(X_input)[0])
    risk_label = "High Risk" if prob > 0.65 else ("Medium Risk" if prob > 0.40 else "Low Risk")
    return jsonify({"probability": round(prob, 4), "default": label, "risk_label": risk_label})

@app.route("/api/live_stream")
def api_live_stream():
    """Return a batch of 20 freshly-generated customers for the live feed."""
    df_live = generate_data(20)
    model   = CACHE["trained"]["Random Forest"]
    scaler  = CACHE["scaler"]
    X_live, _, _ = preprocess(df_live)
    probs  = model.predict_proba(X_live)[:, 1]
    result = []
    for i, row in df_live.iterrows():
        p = float(probs[i % len(probs)])
        result.append({
            "id":        int(row["customer_id"]),
            "education": row["education_level"],
            "income":    round(float(row["annual_income"]), 1),
            "credit":    round(float(row["credit_score"]), 0),
            "dti":       round(float(row["debt_to_income_ratio"]), 3),
            "risk_prob": round(p, 4),
            "risk":      "High" if p > 0.65 else ("Medium" if p > 0.40 else "Low"),
            "actual":    int(row["default"]),
        })
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
