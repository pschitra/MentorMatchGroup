
import itertools
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MentorMatch Intelligence Dashboard",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_TEMPLATE = "plotly_dark"
PRIMARY = "#7C3AED"
SECONDARY = "#22D3EE"
ACCENT = "#F97316"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"
BG = "#0B1020"
CARD = "#111827"

CUSTOM_CSS = """
<style>
    :root {
        --mentor-bg: #0B1020;
        --mentor-card: #111827;
        --mentor-card-2: #151D2E;
        --mentor-text: #F8FAFC;
        --mentor-muted: #94A3B8;
        --mentor-purple: #7C3AED;
        --mentor-cyan: #22D3EE;
        --mentor-orange: #F97316;
        --mentor-green: #10B981;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(124, 58, 237, 0.22), transparent 32%),
            radial-gradient(circle at top right, rgba(34, 211, 238, 0.16), transparent 30%),
            linear-gradient(180deg, #070B16 0%, #0B1020 45%, #090D18 100%);
        color: var(--mentor-text);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }
    .main-title {
        font-size: 2.55rem;
        line-height: 1.05;
        font-weight: 900;
        letter-spacing: -0.06em;
        background: linear-gradient(90deg, #FFFFFF, #A78BFA, #22D3EE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #CBD5E1;
        font-size: 1.05rem;
        margin-bottom: 1.15rem;
    }
    .story-card {
        padding: 1.2rem 1.25rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.94), rgba(30, 41, 59, 0.82));
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem 1.05rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.22), rgba(15, 23, 42, 0.92));
        border: 1px solid rgba(167, 139, 250, 0.22);
        min-height: 116px;
    }
    .metric-label {
        color: #A5B4FC;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 1.9rem;
        font-weight: 900;
        margin-top: 0.2rem;
    }
    .metric-note {
        color: #CBD5E1;
        font-size: 0.86rem;
        margin-top: 0.2rem;
    }
    .section-kicker {
        color: #22D3EE;
        text-transform: uppercase;
        font-weight: 800;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        margin-bottom: 0.3rem;
    }
    .insight-box {
        padding: 1rem;
        border-radius: 1rem;
        background: rgba(15, 23, 42, 0.78);
        border-left: 4px solid #22D3EE;
        border-top: 1px solid rgba(148, 163, 184, 0.16);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        margin: 0.6rem 0;
        color: #E5E7EB;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 1rem;
        background: rgba(249, 115, 22, 0.10);
        border-left: 4px solid #F97316;
        margin: 0.6rem 0;
        color: #F8FAFC;
    }
    .small-muted {
        color: #94A3B8;
        font-size: 0.88rem;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 1rem;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.16);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(15, 23, 42, 0.76);
        border-radius: 999px;
        color: #CBD5E1;
        padding: 0.6rem 1rem;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(124, 58, 237, 0.95), rgba(34, 211, 238, 0.72));
        color: #FFFFFF;
        border-color: rgba(255,255,255,0.28);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def one_hot_encoder():
    """Compatibility wrapper for scikit-learn versions using sparse_output or sparse."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv("MentorMatch_UAE_Clean.csv")


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def clean_and_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    original_rows, original_cols = df.shape
    data = df.copy()
    data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]

    if "respondent_id" in data.columns:
        data = data.drop_duplicates(subset=["respondent_id"], keep="first")
    else:
        data = data.drop_duplicates()

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype(str).str.strip()
        data[col] = data[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    known_numeric = [
        "age", "years_experience", "monthly_income_aed", "months_active",
        "sessions_attended", "messages_sent", "num_mentors", "mentor_match_score",
        "avg_session_rating", "response_time_hours", "goals_completed",
        "income_growth_pct", "skills_gained", "confidence_score", "days_to_subscribe"
    ]
    for col in known_numeric:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "churned" in data.columns:
        data["churned"] = data["churned"].fillna("Not Applicable")

    cat_cols = data.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        data[col] = data[col].fillna("Unknown")

    num_cols = data.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    if "age" in data.columns:
        data["age_group"] = pd.cut(
            data["age"],
            bins=[0, 24, 29, 34, 44, 100],
            labels=["18-24", "25-29", "30-34", "35-44", "45+"],
            include_lowest=True,
        ).astype(str)

    if "years_experience" in data.columns:
        data["experience_band"] = pd.cut(
            data["years_experience"],
            bins=[-0.1, 1, 3, 5, 10, 50],
            labels=["0-1 yrs", "1-3 yrs", "3-5 yrs", "5-10 yrs", "10+ yrs"],
        ).astype(str)

    if "monthly_income_aed" in data.columns:
        data["income_band"] = pd.cut(
            data["monthly_income_aed"],
            bins=[-1, 0, 5000, 10000, 15000, 25000, 999999],
            labels=["No income", "<5k", "5k-10k", "10k-15k", "15k-25k", "25k+"],
        ).astype(str)

    if "career_outcome" in data.columns:
        data["positive_outcome"] = np.where(data["career_outcome"].str.lower().eq("positive"), 1, 0)

    if "subscription_tier" in data.columns:
        data["paid_subscriber"] = np.where(data["subscription_tier"].str.lower().ne("free"), 1, 0)
        data["premium_subscriber"] = np.where(data["subscription_tier"].str.lower().eq("premium"), 1, 0)

    if "mentor_industry_match" in data.columns:
        data["mentor_match_flag"] = np.where(data["mentor_industry_match"].str.lower().eq("yes"), 1, 0)

    if "referred_friend" in data.columns:
        data["referral_flag"] = np.where(data["referred_friend"].str.lower().eq("yes"), 1, 0)

    if {"sessions_attended", "months_active"}.issubset(data.columns):
        data["sessions_per_month"] = data["sessions_attended"] / data["months_active"].replace(0, np.nan)
        data["sessions_per_month"] = data["sessions_per_month"].replace([np.inf, -np.inf], np.nan).fillna(0)

    if {"messages_sent", "months_active"}.issubset(data.columns):
        data["messages_per_month"] = data["messages_sent"] / data["months_active"].replace(0, np.nan)
        data["messages_per_month"] = data["messages_per_month"].replace([np.inf, -np.inf], np.nan).fillna(0)

    engagement_parts = []
    for col, weight in [
        ("sessions_attended", 0.28),
        ("messages_sent", 0.18),
        ("months_active", 0.16),
        ("num_mentors", 0.12),
        ("mentor_match_score", 0.12),
        ("avg_session_rating", 0.08),
        ("goals_completed", 0.06),
    ]:
        if col in data.columns:
            engagement_parts.append(normalize_series(data[col]) * weight)
    if engagement_parts:
        data["engagement_score"] = sum(engagement_parts) * 100

    outcome_parts = []
    for col, weight in [
        ("confidence_score", 0.28),
        ("income_growth_pct", 0.24),
        ("goals_completed", 0.18),
        ("skills_gained", 0.16),
        ("mentor_match_score", 0.08),
        ("avg_session_rating", 0.06),
    ]:
        if col in data.columns:
            outcome_parts.append(normalize_series(data[col]) * weight)
    if outcome_parts:
        data["career_roi_score"] = sum(outcome_parts) * 100

    cleaning_report = {
        "Original rows": original_rows,
        "Original columns": original_cols,
        "Rows after duplicate removal": data.shape[0],
        "Columns after feature engineering": data.shape[1],
        "Duplicates removed": original_rows - data.shape[0],
    }
    return data, cleaning_report


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### 🎛️ Story filters")
    filtered = df.copy()

    for col, label in [
        ("emirate", "Emirate"),
        ("age_group", "Age group"),
        ("experience_band", "Experience band"),
        ("goal_type", "Career goal"),
        ("subscription_tier", "Subscription tier"),
        ("career_outcome", "Career outcome"),
    ]:
        if col in filtered.columns:
            values = sorted([v for v in filtered[col].dropna().unique().tolist() if str(v) != "nan"])
            selected = st.sidebar.multiselect(label, values, default=values)
            if selected:
                filtered = filtered[filtered[col].isin(selected)]

    return filtered


def metric_card(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def story_box(title: str, body: str, kind: str = "insight"):
    cls = "warning-box" if kind == "warning" else "insight-box"
    st.markdown(f"""<div class="{cls}"><b>{title}</b><br>{body}</div>""", unsafe_allow_html=True)


def fig_layout(fig, title=None, height=430):
    fig.update_layout(
        template=DARK_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.40)",
        font=dict(color="#E5E7EB", family="Inter, Arial"),
        title=dict(text=title, x=0.02, xanchor="left") if title else None,
        margin=dict(l=25, r=25, t=60 if title else 25, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def positive_rate_table(df: pd.DataFrame, feature: str, target_col: str = "positive_outcome") -> pd.DataFrame:
    if feature not in df.columns or target_col not in df.columns:
        return pd.DataFrame()
    grp = (
        df.groupby(feature)
        .agg(users=(target_col, "size"), positive_rate=(target_col, "mean"))
        .reset_index()
    )
    grp["positive_rate_pct"] = (grp["positive_rate"] * 100).round(1)
    return grp.sort_values(["positive_rate", "users"], ascending=False)


def crosstab_percent(df: pd.DataFrame, row: str, col: str) -> pd.DataFrame:
    tab = pd.crosstab(df[row], df[col], normalize="index") * 100
    return tab.round(1)


def make_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", one_hot_encoder()),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])


def get_feature_names(preprocessor, X: pd.DataFrame) -> List[str]:
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    names = list(numeric_features)
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_features).tolist()
        names.extend(cat_names)
    except Exception:
        names.extend(categorical_features)
    return names


def target_series_for_classification(df: pd.DataFrame, target_choice: str) -> pd.Series:
    if target_choice == "Positive career outcome":
        return df["positive_outcome"].astype(int)
    if target_choice == "Paid subscriber":
        return df["paid_subscriber"].astype(int)
    if target_choice == "Premium subscriber":
        return df["premium_subscriber"].astype(int)
    if target_choice == "Churn risk":
        return np.where(df["churned"].astype(str).str.lower().eq("yes"), 1, 0)
    return df["positive_outcome"].astype(int)


def drop_leakage_columns(df: pd.DataFrame, target_choice: str, strict: bool = True) -> pd.DataFrame:
    drop_cols = [
        "respondent_id",
        "career_outcome",
        "positive_outcome",
        "paid_subscriber",
        "premium_subscriber",
    ]

    if target_choice in ["Paid subscriber", "Premium subscriber"]:
        drop_cols += ["subscription_tier", "days_to_subscribe"]
    if target_choice == "Positive career outcome" and strict:
        drop_cols += ["income_growth_pct", "confidence_score", "goals_completed", "career_roi_score"]
    if target_choice == "Churn risk":
        drop_cols += ["churned"]

    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


@st.cache_data(show_spinner=False)
def run_classification_models(
    df: pd.DataFrame,
    target_choice: str,
    test_size: float,
    knn_k: int,
    max_depth: int,
    n_estimators: int,
    strict_leakage: bool,
    threshold: float,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    y = target_series_for_classification(df, target_choice)
    X = drop_leakage_columns(df, target_choice, strict=strict_leakage)
    X = X.loc[:, X.nunique(dropna=False) > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() == 2 else None
    )

    preprocessor = make_preprocessor(X_train)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=knn_k),
        "Decision Tree": DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=12, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=n_estimators, max_depth=None, min_samples_leaf=6, random_state=42, n_jobs=-1),
        "Gradient Boosted": GradientBoostingClassifier(n_estimators=max(60, n_estimators // 2), learning_rate=0.06, max_depth=3, random_state=42),
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        train_pred = pipe.predict(X_train)
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            test_prob = pipe.predict_proba(X_test)[:, 1]
            test_pred = (test_prob >= threshold).astype(int)
        else:
            test_prob = None
            test_pred = pipe.predict(X_test)

        auc = roc_auc_score(y_test, test_prob) if test_prob is not None and len(np.unique(y_test)) == 2 else np.nan
        rows.append({
            "Model": name,
            "Train accuracy": accuracy_score(y_train, train_pred),
            "Test accuracy": accuracy_score(y_test, test_pred),
            "Precision": precision_score(y_test, test_pred, zero_division=0),
            "Recall": recall_score(y_test, test_pred, zero_division=0),
            "F1-score": f1_score(y_test, test_pred, zero_division=0),
            "ROC-AUC": auc,
        })
        fitted[name] = {
            "pipeline": pipe,
            "X_test": X_test,
            "y_test": y_test,
            "test_prob": test_prob,
            "test_pred": test_pred,
        }

    results = pd.DataFrame(rows)
    return results, fitted, X


def regression_target_options(df: pd.DataFrame) -> List[str]:
    exclude = {"respondent_id", "positive_outcome", "paid_subscriber", "premium_subscriber", "mentor_match_flag", "referral_flag"}
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]
    preferred = ["income_growth_pct", "confidence_score", "career_roi_score", "days_to_subscribe", "monthly_income_aed"]
    ordered = [c for c in preferred if c in numeric_cols] + [c for c in numeric_cols if c not in preferred]
    return ordered


def regression_feature_frame(df: pd.DataFrame, target: str, strict_leakage: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    y = pd.to_numeric(df[target], errors="coerce")
    X = df.drop(columns=[target], errors="ignore")

    drop_cols = ["respondent_id", "career_outcome", "positive_outcome"]
    if strict_leakage:
        if target == "income_growth_pct":
            drop_cols += ["career_roi_score", "confidence_score", "goals_completed"]
        if target == "confidence_score":
            drop_cols += ["career_roi_score", "career_outcome"]
        if target == "career_roi_score":
            drop_cols += ["income_growth_pct", "confidence_score", "goals_completed"]

    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    X = X.loc[:, X.nunique(dropna=False) > 1]
    return X, y


@st.cache_data(show_spinner=False)
def run_regression_models(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    alpha: float,
    l1_ratio: float,
    max_depth: int,
    n_estimators: int,
    strict_leakage: bool,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    X, y = regression_feature_frame(df, target, strict_leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    preprocessor = make_preprocessor(X_train)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=alpha, random_state=42),
        "Lasso Regression": Lasso(alpha=alpha, max_iter=12000, random_state=42),
        "Elastic Net": ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=12000, random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=12, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=n_estimators, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=max(80, n_estimators // 2), learning_rate=0.06, max_depth=3, random_state=42),
        "AdaBoost Regressor": AdaBoostRegressor(n_estimators=max(60, n_estimators // 3), learning_rate=0.05, random_state=42),
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        train_pred = pipe.predict(X_train)
        test_pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "Train R²": r2_score(y_train, train_pred),
            "Test R²": r2_score(y_test, test_pred),
            "MAE": mean_absolute_error(y_test, test_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        })
        fitted[name] = {
            "pipeline": pipe,
            "X_test": X_test,
            "y_test": y_test,
            "test_pred": test_pred,
        }

    return pd.DataFrame(rows), fitted, X


def build_cluster_matrix(df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    X = df[selected_features].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype(str)
    pre = make_preprocessor(X)
    matrix = pre.fit_transform(X)
    return pd.DataFrame(matrix)


@st.cache_data(show_spinner=False)
def run_kmeans_suite(df: pd.DataFrame, selected_features: List[str], max_k: int = 15):
    X_mat = build_cluster_matrix(df, selected_features)
    inertias, silhouettes = [], []
    ks = list(range(2, max_k + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_mat)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_mat, labels))

    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(X_mat)
    return X_mat, ks, inertias, silhouettes, coords


def association_transactions(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    items = pd.DataFrame(index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            series = pd.qcut(df[col].rank(method="first"), q=4, duplicates="drop")
            labels = [f"{col}=Q{i+1}" for i in range(series.cat.categories.size)]
            mapped = pd.qcut(df[col].rank(method="first"), q=series.cat.categories.size, labels=labels, duplicates="drop")
            dummies = pd.get_dummies(mapped)
        else:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col)
        items = pd.concat([items, dummies.astype(bool)], axis=1)
    items = items.loc[:, items.sum() > 0]
    return items


@st.cache_data(show_spinner=False)
def generate_pair_rules(df: pd.DataFrame, cols: List[str], min_support: float, min_confidence: float, min_lift: float) -> pd.DataFrame:
    items = association_transactions(df, cols)
    n = len(items)
    if n == 0 or items.shape[1] < 2:
        return pd.DataFrame()

    supports = items.mean()
    rows = []
    columns = list(items.columns)

    for a, b in itertools.permutations(columns, 2):
        both = (items[a] & items[b]).mean()
        if both < min_support:
            continue
        conf = both / supports[a] if supports[a] > 0 else 0
        lift = conf / supports[b] if supports[b] > 0 else 0
        leverage = both - supports[a] * supports[b]
        conviction = (1 - supports[b]) / (1 - conf) if conf < 1 else np.inf

        if conf >= min_confidence and lift >= min_lift:
            rows.append({
                "Antecedent": a,
                "Consequent": b,
                "Support": both,
                "Confidence": conf,
                "Lift": lift,
                "Leverage": leverage,
                "Conviction": conviction,
            })

    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules
    return rules.sort_values(["Lift", "Confidence", "Support"], ascending=False).head(250)


def render_header():
    st.markdown('<div class="main-title">MentorMatch Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A dark-themed storytelling dashboard for improving mentor matching, subscription conversion, and career outcomes for early-career professionals in the UAE.</div>',
        unsafe_allow_html=True,
    )


def render_kpis(df: pd.DataFrame):
    users = len(df)
    pos_rate = df["positive_outcome"].mean() * 100 if "positive_outcome" in df.columns else np.nan
    paid_rate = df["paid_subscriber"].mean() * 100 if "paid_subscriber" in df.columns else np.nan
    avg_sessions = df["sessions_attended"].mean() if "sessions_attended" in df.columns else np.nan
    avg_match = df["mentor_match_score"].mean() if "mentor_match_score" in df.columns else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Filtered users", f"{users:,}", "Current analysis base")
    with c2:
        metric_card("Positive outcomes", f"{pos_rate:.1f}%", "Career outcome success rate")
    with c3:
        metric_card("Paid conversion", f"{paid_rate:.1f}%", "Basic + Premium users")
    with c4:
        metric_card("Avg sessions", f"{avg_sessions:.1f}", "Mentorship touchpoints")
    with c5:
        metric_card("Avg match score", f"{avg_match:.1f}", "Mentor fit quality")


def tab_executive(df: pd.DataFrame):
    st.markdown('<div class="section-kicker">Executive story</div>', unsafe_allow_html=True)
    st.markdown("### From career confusion to measurable career momentum")
    render_kpis(df)

    st.markdown(
        """
        <div class="story-card">
        <b>Business lens:</b> MentorMatch is not just a mentor-booking app. It is a decision-support platform for young professionals navigating the UAE job market.
        The dashboard connects user profile, career goal, mentor engagement, app behaviour, subscription tier, and outcomes so that leadership can answer three practical questions:
        <br><br>
        <b>1.</b> Who gets better career outcomes?<br>
        <b>2.</b> What behaviours convert free users into subscribers?<br>
        <b>3.</b> Which interventions help mentors and mentees both win?
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        if {"subscription_tier", "career_outcome"}.issubset(df.columns):
            fig = px.histogram(
                df,
                x="subscription_tier",
                color="career_outcome",
                barmode="group",
                text_auto=True,
                title="Career outcomes by subscription tier",
                color_discrete_sequence=[SECONDARY, SUCCESS, ACCENT],
            )
            st.plotly_chart(fig_layout(fig, height=440), use_container_width=True)

    with c2:
        if {"engagement_score", "career_roi_score", "subscription_tier"}.issubset(df.columns):
            fig = px.scatter(
                df,
                x="engagement_score",
                y="career_roi_score",
                color="subscription_tier",
                size="sessions_attended" if "sessions_attended" in df.columns else None,
                hover_data=["goal_type", "career_stage", "career_outcome"],
                title="Engagement score vs career ROI score",
                color_discrete_sequence=[PRIMARY, SECONDARY, SUCCESS, ACCENT],
            )
            st.plotly_chart(fig_layout(fig, height=440), use_container_width=True)

    story_box(
        "Leadership takeaway",
        "The most valuable users are not only the highest-income users. Strong outcomes often come from a mix of mentor fit, repeated sessions, fast responses, goal completion, and career-stage relevance. This is why the dashboard analyses multiple drivers instead of focusing only on income.",
    )


def tab_cleaning(df_raw: pd.DataFrame, df: pd.DataFrame, report: Dict[str, int]):
    st.markdown('<div class="section-kicker">Data foundation</div>', unsafe_allow_html=True)
    st.markdown("### Cleaning, validation and engineered business features")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Raw rows", f"{report['Original rows']:,}", "Before cleaning")
    with c2:
        metric_card("Clean rows", f"{report['Rows after duplicate removal']:,}", "After deduplication")
    with c3:
        metric_card("Raw columns", f"{report['Original columns']:,}", "Original variables")
    with c4:
        metric_card("Final columns", f"{report['Columns after feature engineering']:,}", "With engineered features")

    story_box(
        "Feature engineering applied",
        "Created age groups, experience bands, income bands, paid-subscriber flags, positive-outcome flag, mentor-match flag, referral flag, engagement score, sessions per month, messages per month, and a career ROI score.",
    )

    missing = df_raw.isna().mean().mul(100).sort_values(ascending=False).reset_index()
    missing.columns = ["Column", "Missing %"]
    missing = missing[missing["Missing %"] > 0]
    c1, c2 = st.columns([0.95, 1.05])
    with c1:
        if not missing.empty:
            fig = px.bar(missing.head(15), x="Missing %", y="Column", orientation="h", text="Missing %", title="Missing values before cleaning")
            fig.update_traces(texttemplate="%{text:.1f}%")
            st.plotly_chart(fig_layout(fig, height=430), use_container_width=True)
        else:
            story_box("Data health", "No missing values detected in the raw upload.")
    with c2:
        sample_cols = [
            "age", "age_group", "years_experience", "experience_band",
            "subscription_tier", "career_outcome", "engagement_score", "career_roi_score"
        ]
        st.dataframe(df[[c for c in sample_cols if c in df.columns]].head(12), use_container_width=True)

    with st.expander("View cleaned dataset preview"):
        st.dataframe(df.head(100), use_container_width=True)


def tab_descriptive(df: pd.DataFrame):
    st.markdown('<div class="section-kicker">Descriptive analytics</div>', unsafe_allow_html=True)
    st.markdown("### What does the MentorMatch user base look like?")

    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        row_feature = st.selectbox(
            "Cross-tab feature against career outcome",
            [c for c in ["age_group", "experience_band", "goal_type", "primary_challenge", "subscription_tier", "target_industry", "emirate", "career_stage", "mentor_industry_match"] if c in df.columns],
        )
        normalize = st.radio("Cross-tab view", ["Percentage by row", "Counts"], horizontal=False)

    with c2:
        if "career_outcome" in df.columns:
            if normalize == "Percentage by row":
                tab = crosstab_percent(df, row_feature, "career_outcome")
                fig = px.imshow(
                    tab,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Purples",
                    title=f"{row_feature.replace('_', ' ').title()} vs career outcome (%)",
                )
                st.plotly_chart(fig_layout(fig, height=430), use_container_width=True)
                st.dataframe(tab, use_container_width=True)
            else:
                tab = pd.crosstab(df[row_feature], df["career_outcome"])
                fig = px.imshow(
                    tab,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Teal",
                    title=f"{row_feature.replace('_', ' ').title()} vs career outcome (counts)",
                )
                st.plotly_chart(fig_layout(fig, height=430), use_container_width=True)
                st.dataframe(tab, use_container_width=True)

    st.markdown("### Correlation lens: career outcome is multi-factor, not income-only")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr_cols = st.multiselect(
        "Select numeric variables for correlation",
        numeric_cols,
        default=[c for c in [
            "age", "years_experience", "monthly_income_aed", "months_active",
            "sessions_attended", "messages_sent", "mentor_match_score", "avg_session_rating",
            "response_time_hours", "goals_completed", "income_growth_pct",
            "skills_gained", "confidence_score", "engagement_score", "career_roi_score",
            "positive_outcome", "paid_subscriber"
        ] if c in numeric_cols],
    )
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(numeric_only=True).round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Correlation heatmap")
        st.plotly_chart(fig_layout(fig, height=720), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if "goal_type" in df.columns:
            goal_counts = df["goal_type"].value_counts().reset_index()
            goal_counts.columns = ["Goal", "Users"]
            fig = px.bar(goal_counts, x="Users", y="Goal", orientation="h", title="Most common career goals", text="Users")
            st.plotly_chart(fig_layout(fig, height=430), use_container_width=True)
    with c2:
        if "primary_challenge" in df.columns:
            challenge_counts = df["primary_challenge"].value_counts().reset_index()
            challenge_counts.columns = ["Challenge", "Users"]
            fig = px.bar(challenge_counts, x="Users", y="Challenge", orientation="h", title="Main early-career struggles", text="Users")
            st.plotly_chart(fig_layout(fig, height=430), use_container_width=True)


def tab_diagnostic(df: pd.DataFrame):
    st.markdown('<div class="section-kicker">Diagnostic analytics</div>', unsafe_allow_html=True)
    st.markdown("### Why do some mentees achieve better outcomes and convert to subscribers?")

    driver = st.selectbox(
        "Choose a driver to investigate",
        [c for c in ["age_group", "experience_band", "goal_type", "primary_challenge", "target_industry", "subscription_tier", "career_stage", "mentor_industry_match", "signup_channel"] if c in df.columns],
    )

    if "positive_outcome" in df.columns:
        rate = positive_rate_table(df, driver)
        c1, c2 = st.columns([0.58, 0.42])
        with c1:
            fig = px.bar(
                rate,
                x="positive_rate_pct",
                y=driver,
                orientation="h",
                text="positive_rate_pct",
                hover_data=["users"],
                title=f"Positive career outcome rate by {driver.replace('_', ' ')}",
            )
            fig.update_traces(texttemplate="%{text:.1f}%")
            st.plotly_chart(fig_layout(fig, height=520), use_container_width=True)
        with c2:
            st.dataframe(rate.rename(columns={driver: "Segment", "users": "Users", "positive_rate_pct": "Positive outcome %"})[["Segment", "Users", "Positive outcome %"]], use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if {"engagement_score", "career_roi_score", "career_outcome"}.issubset(df.columns):
            fig = px.box(
                df,
                x="career_outcome",
                y="engagement_score",
                color="career_outcome",
                title="Does engagement differ by career outcome?",
                color_discrete_sequence=[ACCENT, SUCCESS],
            )
            st.plotly_chart(fig_layout(fig, height=430), use_container_width=True)
    with c2:
        if {"response_time_hours", "career_outcome"}.issubset(df.columns):
            fig = px.violin(
                df,
                x="career_outcome",
                y="response_time_hours",
                color="career_outcome",
                box=True,
                points="outliers",
                title="Mentor response time by career outcome",
                color_discrete_sequence=[ACCENT, SECONDARY],
            )
            st.plotly_chart(fig_layout(fig, height=430), use_container_width=True)

    st.markdown("### Diagnostic story for the UAE early-career market")
    story_box(
        "Career reality",
        "Entry-level professionals in the UAE often face uncertainty around visa stability, salary growth, networking, interview readiness, and industry switching. MentorMatch can win by giving structure: the right mentor, practical action plans, quick feedback, and visible progress.",
    )
    story_box(
        "Conversion lens",
        "Users are more likely to pay when the app moves from 'advice marketplace' to 'career progress system'. High-impact behaviours to encourage include first mentor match, session attendance, message follow-ups, goal completion, and referral loops.",
    )
    story_box(
        "Mentor win-win",
        "Mentors need quality matches, clear expectations, ratings, repeat bookings, and recognition. Better mentor experience improves response time and session quality, which can improve outcomes and subscription renewals.",
    )


def tab_predictive(df: pd.DataFrame):
    st.markdown('<div class="section-kicker">Predictive analytics</div>', unsafe_allow_html=True)
    st.markdown("### Classification and regression models for decision support")

    subtab1, subtab2 = st.tabs(["🧠 Classification", "📈 Regression"])

    with subtab1:
        st.markdown("#### Predict users likely to convert or achieve positive outcomes")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            target_choice = st.selectbox("Classification target", ["Positive career outcome", "Paid subscriber", "Premium subscriber", "Churn risk"])
        with c2:
            test_size = st.slider("Test size", 0.15, 0.40, 0.25, 0.05)
        with c3:
            knn_k = st.slider("KNN neighbours", 3, 25, 7, 2)
        with c4:
            max_depth = st.slider("Decision tree max depth", 2, 14, 6)

        c1, c2, c3 = st.columns(3)
        with c1:
            n_estimators = st.slider("Tree ensemble estimators", 50, 400, 180, 25)
        with c2:
            threshold = st.slider("Classification threshold", 0.10, 0.90, 0.50, 0.05)
        with c3:
            strict_leakage = st.toggle("Leakage-safe mode", value=True, help="Excludes variables that may directly reveal the target outcome.")

        with st.spinner("Training classification models..."):
            results, fitted, X_used = run_classification_models(df, target_choice, test_size, knn_k, max_depth, n_estimators, strict_leakage, threshold)

        st.dataframe(results.style.format({
            "Train accuracy": "{:.3f}",
            "Test accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1-score": "{:.3f}",
            "ROC-AUC": "{:.3f}",
        }), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            long = results.melt(id_vars="Model", value_vars=["Test accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"], var_name="Metric", value_name="Score")
            fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group", title="Model performance comparison")
            st.plotly_chart(fig_layout(fig, height=460), use_container_width=True)
        with c2:
            model_choice = st.selectbox("Inspect model", list(fitted.keys()))
            item = fitted[model_choice]
            cm = confusion_matrix(item["y_test"], item["test_pred"])
            fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Purples",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["No", "Yes"],
                y=["No", "Yes"],
                title=f"Confusion matrix: {model_choice}",
            )
            st.plotly_chart(fig_layout(fig, height=460), use_container_width=True)

        roc_fig = go.Figure()
        for name, item in fitted.items():
            if item["test_prob"] is not None:
                fpr, tpr, _ = roc_curve(item["y_test"], item["test_prob"])
                auc = roc_auc_score(item["y_test"], item["test_prob"])
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} AUC={auc:.3f}"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(dash="dash")))
        roc_fig.update_xaxes(title="False positive rate")
        roc_fig.update_yaxes(title="True positive rate")
        st.plotly_chart(fig_layout(roc_fig, "ROC curve stability check", height=520), use_container_width=True)

    with subtab2:
        st.markdown("#### Predict numeric career/business outcomes with linear and decision-tree family regression")
        targets = regression_target_options(df)
        default_ix = targets.index("income_growth_pct") if "income_growth_pct" in targets else 0
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            target = st.selectbox("Regression target", targets, index=default_ix)
        with c2:
            test_size_r = st.slider("Regression test size", 0.15, 0.40, 0.25, 0.05, key="reg_test")
        with c3:
            alpha = st.slider("Lambda / alpha", 0.001, 20.0, 1.0, 0.1, help="Controls regularisation strength for Ridge, Lasso and Elastic Net.")
        with c4:
            l1_ratio = st.slider("Elastic Net L1 ratio", 0.05, 0.95, 0.50, 0.05)

        c1, c2, c3 = st.columns(3)
        with c1:
            reg_depth = st.slider("Regressor max depth", 2, 16, 7)
        with c2:
            reg_estimators = st.slider("Regressor estimators", 50, 450, 220, 25)
        with c3:
            reg_leakage = st.toggle("Regression leakage-safe mode", value=True)

        with st.spinner("Training regression models..."):
            reg_results, reg_fitted, X_reg = run_regression_models(df, target, test_size_r, alpha, l1_ratio, reg_depth, reg_estimators, reg_leakage)

        st.dataframe(reg_results.sort_values("Test R²", ascending=False).style.format({
            "Train R²": "{:.3f}",
            "Test R²": "{:.3f}",
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
        }), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(reg_results.sort_values("Test R²"), x="Test R²", y="Model", orientation="h", title=f"Regression model ranking for {target}")
            st.plotly_chart(fig_layout(fig, height=520), use_container_width=True)
        with c2:
            err = reg_results.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Error metric", value_name="Value")
            fig = px.bar(err, x="Model", y="Value", color="Error metric", barmode="group", title="Error comparison: lower is better")
            st.plotly_chart(fig_layout(fig, height=520), use_container_width=True)

        inspect_reg = st.selectbox("Actual vs predicted model", list(reg_fitted.keys()))
        item = reg_fitted[inspect_reg]
        pred_df = pd.DataFrame({"Actual": item["y_test"], "Predicted": item["test_pred"]})
        fig = px.scatter(pred_df, x="Actual", y="Predicted", trendline="ols", title=f"Actual vs predicted: {inspect_reg}", hover_data=pred_df.columns)
        st.plotly_chart(fig_layout(fig, height=500), use_container_width=True)


def tab_clustering(df: pd.DataFrame):
    st.markdown('<div class="section-kicker">Segmentation</div>', unsafe_allow_html=True)
    st.markdown("### K-Means clustering to discover MentorMatch user personas")

    feature_candidates = [c for c in [
        "age", "years_experience", "monthly_income_aed", "months_active", "sessions_attended",
        "messages_sent", "num_mentors", "mentor_match_score", "avg_session_rating", "response_time_hours",
        "goals_completed", "skills_gained", "confidence_score", "engagement_score",
        "career_stage", "goal_type", "primary_challenge", "subscription_tier",
        "target_industry", "mentor_industry_match", "emirate"
    ] if c in df.columns]

    selected_features = st.multiselect(
        "Select clustering features",
        feature_candidates,
        default=[c for c in ["age", "years_experience", "sessions_attended", "mentor_match_score", "confidence_score", "engagement_score", "goal_type", "subscription_tier", "primary_challenge"] if c in feature_candidates],
    )
    if len(selected_features) < 2:
        st.warning("Select at least two features for clustering.")
        return

    max_k = st.slider("Maximum K for elbow method", 8, 15, 15)
    X_mat, ks, inertias, silhouettes, coords = run_kmeans_suite(df, selected_features, max_k=max_k)

    c1, c2 = st.columns(2)
    with c1:
        elbow_df = pd.DataFrame({"K": ks, "Inertia": inertias})
        fig = px.line(elbow_df, x="K", y="Inertia", markers=True, title="Elbow method: cluster distance / inertia")
        st.plotly_chart(fig_layout(fig, height=420), use_container_width=True)
    with c2:
        sil_df = pd.DataFrame({"K": ks, "Silhouette score": silhouettes})
        fig = px.line(sil_df, x="K", y="Silhouette score", markers=True, title="Silhouette analysis: higher is better")
        st.plotly_chart(fig_layout(fig, height=420), use_container_width=True)

    best_k = int(sil_df.sort_values("Silhouette score", ascending=False).iloc[0]["K"])
    selected_k = st.slider("Choose K for final segmentation", 2, max_k, best_k)

    km = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    clusters = km.fit_predict(X_mat)
    final = df.copy()
    final["Cluster"] = clusters.astype(str)
    final["Distance_to_cluster_center"] = np.min(km.transform(X_mat), axis=1)

    plot_df = pd.DataFrame(coords, columns=["PC1", "PC2", "PC3"])
    plot_df["Cluster"] = final["Cluster"].values
    for col in ["respondent_id", "career_outcome", "subscription_tier", "goal_type", "primary_challenge", "engagement_score"]:
        if col in final.columns:
            plot_df[col] = final[col].values

    fig = px.scatter_3d(
        plot_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        hover_data=[c for c in ["respondent_id", "career_outcome", "subscription_tier", "goal_type", "primary_challenge", "engagement_score"] if c in plot_df.columns],
        title=f"Interactive 3D user segmentation with K={selected_k}",
    )
    st.plotly_chart(fig_layout(fig, height=720), use_container_width=True)

    st.markdown("#### Cluster profile summary")
    summary = final.groupby("Cluster").agg(
        users=("Cluster", "size"),
        avg_age=("age", "mean") if "age" in final.columns else ("Cluster", "size"),
        avg_experience=("years_experience", "mean") if "years_experience" in final.columns else ("Cluster", "size"),
        avg_sessions=("sessions_attended", "mean") if "sessions_attended" in final.columns else ("Cluster", "size"),
        avg_match_score=("mentor_match_score", "mean") if "mentor_match_score" in final.columns else ("Cluster", "size"),
        positive_rate=("positive_outcome", "mean") if "positive_outcome" in final.columns else ("Cluster", "size"),
        paid_rate=("paid_subscriber", "mean") if "paid_subscriber" in final.columns else ("Cluster", "size"),
        avg_distance=("Distance_to_cluster_center", "mean"),
    ).reset_index()
    if "positive_rate" in summary.columns:
        summary["positive_rate"] *= 100
    if "paid_rate" in summary.columns:
        summary["paid_rate"] *= 100
    st.dataframe(summary.style.format({
        "avg_age": "{:.1f}",
        "avg_experience": "{:.1f}",
        "avg_sessions": "{:.1f}",
        "avg_match_score": "{:.1f}",
        "positive_rate": "{:.1f}%",
        "paid_rate": "{:.1f}%",
        "avg_distance": "{:.2f}",
    }), use_container_width=True)

    fig = px.box(final, x="Cluster", y="Distance_to_cluster_center", color="Cluster", title="Distance from users to assigned cluster centers")
    st.plotly_chart(fig_layout(fig, height=430), use_container_width=True)


def tab_association(df: pd.DataFrame):
    st.markdown('<div class="section-kicker">Association rule mining</div>', unsafe_allow_html=True)
    st.markdown("### Discover which MentorMatch behaviours and needs appear together")

    cols_available = [c for c in [
        "age_group", "experience_band", "income_band", "goal_type", "primary_challenge",
        "subscription_tier", "career_outcome", "target_industry", "career_stage",
        "signup_channel", "mentor_industry_match", "referred_friend", "emirate"
    ] if c in df.columns]

    selected_cols = st.multiselect("Transaction variables", cols_available, default=cols_available[:9])
    c1, c2, c3 = st.columns(3)
    with c1:
        min_support = st.slider("Minimum support", 0.01, 0.30, 0.05, 0.01)
    with c2:
        min_conf = st.slider("Minimum confidence", 0.10, 0.95, 0.35, 0.05)
    with c3:
        min_lift = st.slider("Minimum lift", 0.80, 3.00, 1.05, 0.05)

    rules = generate_pair_rules(df, selected_cols, min_support, min_conf, min_lift)

    if rules.empty:
        st.warning("No rules found. Try lowering support, confidence, or lift.")
        return

    display_rules = rules.copy()
    for col in ["Support", "Confidence", "Lift", "Leverage", "Conviction"]:
        display_rules[col] = display_rules[col].replace([np.inf, -np.inf], np.nan)
    st.dataframe(display_rules.head(80).style.format({
        "Support": "{:.3f}",
        "Confidence": "{:.3f}",
        "Lift": "{:.3f}",
        "Leverage": "{:.3f}",
        "Conviction": "{:.3f}",
    }), use_container_width=True)

    fig = px.scatter(
        display_rules.head(120),
        x="Support",
        y="Confidence",
        size="Lift",
        color="Lift",
        hover_data=["Antecedent", "Consequent", "Leverage"],
        title="Association rules: support vs confidence, sized by lift",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_layout(fig, height=560), use_container_width=True)

    top_rule = display_rules.iloc[0]
    story_box(
        "How to read this",
        f"The strongest displayed rule is: <b>{top_rule['Antecedent']}</b> → <b>{top_rule['Consequent']}</b>. A lift above 1 means this combination appears together more often than random chance. Use strong rules to design bundles, mentor recommendations, onboarding journeys, and subscription nudges.",
    )


def tab_prescriptive(df: pd.DataFrame):
    st.markdown('<div class="section-kicker">Prescriptive analytics</div>', unsafe_allow_html=True)
    st.markdown("### What should MentorMatch do next?")

    recommendations = []

    if {"subscription_tier", "positive_outcome"}.issubset(df.columns):
        tier_rate = positive_rate_table(df, "subscription_tier")
        if not tier_rate.empty:
            best = tier_rate.iloc[0]
            recommendations.append(
                f"Prioritise the behaviours seen in the highest-outcome subscription segment: **{best['subscription_tier']}** users show a positive-outcome rate of **{best['positive_rate_pct']:.1f}%**."
            )

    if {"goal_type", "positive_outcome"}.issubset(df.columns):
        goal_rate = positive_rate_table(df, "goal_type")
        if not goal_rate.empty:
            recommendations.append(
                f"Build specialised journeys for high-performing goals such as **{goal_rate.iloc[0]['goal_type']}**, but also create rescue journeys for lower-outcome goals with structured mentor roadmaps."
            )

    if {"mentor_industry_match", "positive_outcome"}.issubset(df.columns):
        match_rate = positive_rate_table(df, "mentor_industry_match")
        if len(match_rate) >= 2:
            gap = match_rate["positive_rate_pct"].max() - match_rate["positive_rate_pct"].min()
            recommendations.append(
                f"Improve mentor matching quality: the outcome gap across mentor-industry-match groups is around **{gap:.1f} percentage points** in the filtered data."
            )

    if {"response_time_hours", "positive_outcome"}.issubset(df.columns):
        median_resp = df["response_time_hours"].median()
        fast = df[df["response_time_hours"] <= median_resp]["positive_outcome"].mean() * 100
        slow = df[df["response_time_hours"] > median_resp]["positive_outcome"].mean() * 100
        recommendations.append(
            f"Use response-time SLAs for mentors. Faster-than-median responses show **{fast:.1f}%** positive outcomes versus **{slow:.1f}%** for slower responses."
        )

    if {"paid_subscriber", "sessions_attended"}.issubset(df.columns):
        paid_sessions = df[df["paid_subscriber"] == 1]["sessions_attended"].mean()
        free_sessions = df[df["paid_subscriber"] == 0]["sessions_attended"].mean()
        recommendations.append(
            f"Convert users after early engagement: paid users attend **{paid_sessions:.1f}** sessions on average versus **{free_sessions:.1f}** for free users. Trigger upgrade nudges after meaningful session milestones."
        )

    c1, c2 = st.columns([0.65, 0.35])
    with c1:
        for i, rec in enumerate(recommendations[:6], 1):
            story_box(f"Recommendation {i}", rec)

    with c2:
        st.markdown(
            """
            <div class="story-card">
            <b>MentorMatch playbook</b><br><br>
            1. Make onboarding diagnostic-first.<br>
            2. Match by industry + goal + challenge, not only availability.<br>
            3. Push users to their first session quickly.<br>
            4. Offer career-stage subscription bundles.<br>
            5. Track confidence, skills, income growth, and goal completion as outcome KPIs.<br>
            6. Reward mentors for fast response, repeat bookings, and outcome quality.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Interactive summary")
    summary_text = f"""
    MentorMatch currently has **{len(df):,} filtered users** in this analysis view.
    The dashboard indicates that career outcomes should be managed as a system: user background, career goal, primary challenge, mentor fit, engagement, response speed, subscription tier, and follow-through all matter.
    For business growth, the strongest strategy is to convert MentorMatch from a simple booking platform into a measurable career journey platform.
    """
    story_box("Final insight", summary_text)


def main():
    render_header()

    st.sidebar.markdown("## 🧭 MentorMatch")
    st.sidebar.markdown("Dark analytics cockpit for career outcomes and subscription conversion.")

    uploaded_file = st.sidebar.file_uploader("Upload a newer MentorMatch CSV", type=["csv"])
    raw_df = load_csv(uploaded_file)
    df, cleaning_report = clean_and_engineer(raw_df)
    filtered_df = apply_sidebar_filters(df)

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: Use the filters above to turn the dashboard into a storytelling drill-down tool.")

    if filtered_df.empty:
        st.warning("No rows after filtering. Please adjust filters.")
        return

    tabs = st.tabs([
        "🏠 Executive Story",
        "🧹 Data Cleaning",
        "📊 Descriptive",
        "🔎 Diagnostic",
        "🤖 Predictive",
        "🧩 Clustering",
        "🛒 Association Rules",
        "✅ Findings",
    ])

    with tabs[0]:
        tab_executive(filtered_df)
    with tabs[1]:
        tab_cleaning(raw_df, df, cleaning_report)
    with tabs[2]:
        tab_descriptive(filtered_df)
    with tabs[3]:
        tab_diagnostic(filtered_df)
    with tabs[4]:
        tab_predictive(filtered_df)
    with tabs[5]:
        tab_clustering(filtered_df)
    with tabs[6]:
        tab_association(filtered_df)
    with tabs[7]:
        tab_prescriptive(filtered_df)


if __name__ == "__main__":
    main()
