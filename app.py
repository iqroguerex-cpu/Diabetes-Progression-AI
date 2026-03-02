import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Diabetes Progression AI", page_icon="🩺", layout="wide")

# =========================
# CLEAN DARK UI
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #161B22;
}
div[data-testid="stMetricValue"] {
    color: #00FFAA;
}
</style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")
sns.set_style("darkgrid")

# =========================
# LOAD DATA
# =========================
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Progression"] = data.target

# =========================
# SPLIT
# =========================
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# =========================
# SCALING
# =========================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =========================
# MODEL
# =========================
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# METRICS
# =========================
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# =========================
# HEADER
# =========================
st.title("🩺 Diabetes Progression Prediction Dashboard")
st.caption("Multiple Linear Regression • Interactive ML App")

st.divider()

# =========================
# METRICS
# =========================
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")

st.divider()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Predict", "📊 Analytics", "📈 Model Insights", "📥 Download"]
)

# =========================
# TAB 1 — PREDICTION
# =========================
with tab1:
    st.subheader("Adjust Patient Features")

    feature_info = {
        "age": "Age of the patient (standardized value)",
        "sex": "Gender of the patient (encoded numerically)",
        "bmi": "Body Mass Index — indicator of body fat",
        "bp": "Average Blood Pressure",
        "s1": "Total Serum Cholesterol",
        "s2": "Low-Density Lipoproteins (LDL - bad cholesterol)",
        "s3": "High-Density Lipoproteins (HDL - good cholesterol)",
        "s4": "Total Cholesterol / HDL Ratio",
        "s5": "Log of Serum Triglycerides",
        "s6": "Blood Sugar Level"
    }

    user_inputs = []

    cols = st.columns(2)

    for i, feature in enumerate(data.feature_names):
        col = cols[i % 2]

        with col:
            value = st.slider(
                label=feature,
                min_value=float(df[feature].min()),
                max_value=float(df[feature].max()),
                value=float(df[feature].mean()),
                help=feature_info[feature]
            )

            st.caption(feature_info[feature])
            user_inputs.append(value)

    user_inputs = np.array(user_inputs).reshape(1, -1)
    user_inputs = sc.transform(user_inputs)

    prediction = model.predict(user_inputs)

    st.success(f"Predicted Disease Progression: {prediction[0]:.2f}")

# =========================
# TAB 2 — ANALYTICS
# =========================
with tab2:
    st.subheader("Correlation Heatmap")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Actual vs Predicted")

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred, alpha=0.4)
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")

    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax2.plot([min_val, max_val], [min_val, max_val], color='red')

    st.pyplot(fig2)

# =========================
# TAB 3 — MODEL INSIGHTS
# =========================
with tab3:
    st.subheader("Feature Importance")

    coef_df = pd.DataFrame({
        "Feature": data.feature_names,
        "Coefficient": model.coef_,
        "Absolute Impact": np.abs(model.coef_)
    }).sort_values(by="Absolute Impact", ascending=False)

    fig3, ax3 = plt.subplots()
    sns.barplot(x="Absolute Impact", y="Feature", data=coef_df, ax=ax3)
    st.pyplot(fig3)

    st.subheader("Residual Analysis")

    residuals = y_test - y_pred

    fig4, ax4 = plt.subplots()
    ax4.scatter(y_pred, residuals, alpha=0.4)
    ax4.axhline(0, color='red')
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Residuals")

    st.pyplot(fig4)

# =========================
# TAB 4 — DOWNLOAD
# =========================
with tab4:
    results_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
        "Residual": y_test - y_pred
    })

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="diabetes_predictions.csv",
        mime="text/csv"
    )
