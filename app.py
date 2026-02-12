# ==========================================================
# 🚦 Delhi Traffic Travel Time Prediction Dashboard
# FINAL CLEAN USER-FRIENDLY VERSION
# ==========================================================

import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import plotly.express as px

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Delhi Traffic Prediction",
    page_icon="🚦",
    layout="wide"
)

# ==========================================================
# LOAD MODEL & ENCODERS
# ==========================================================

@st.cache_resource
def load_model():
    model_file = [f for f in os.listdir() if "xgboost_model" in f and f.endswith(".pkl")][0]
    return joblib.load(model_file)

@st.cache_resource
def load_encoders():
    encoder_file = [f for f in os.listdir() if "label_encoders" in f and f.endswith(".pkl")][0]
    return joblib.load(encoder_file)

model = load_model()
label_encoders = load_encoders()

# ==========================================================
# SAFE LOG FILE HANDLING
# ==========================================================

log_file = "prediction_logs.csv"
required_columns = list(model.get_booster().feature_names) + ["predicted_time", "timestamp"]

def ensure_log_file():
    if not os.path.exists(log_file):
        pd.DataFrame(columns=required_columns).to_csv(log_file, index=False)
    else:
        try:
            df = pd.read_csv(log_file)
            if not all(col in df.columns for col in required_columns):
                pd.DataFrame(columns=required_columns).to_csv(log_file, index=False)
        except:
            pd.DataFrame(columns=required_columns).to_csv(log_file, index=False)

ensure_log_file()

# ==========================================================
# TITLE
# ==========================================================

st.title("🚦 Delhi Traffic Travel Time Prediction")
st.markdown("### Estimate your travel time based on traffic conditions")
st.markdown("---")

# ==========================================================
# SIDEBAR INPUTS
# ==========================================================

st.sidebar.header("⚙️ Enter Trip Details")

distance_km = st.sidebar.number_input("Distance (km)", 0.0, 100.0, 5.0)

time_of_day = st.sidebar.selectbox(
    "Time of Day",
    label_encoders["time_of_day"].classes_
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    label_encoders["day_of_week"].classes_
)

weather_condition = st.sidebar.selectbox(
    "Weather Condition",
    label_encoders["weather_condition"].classes_
)

traffic_density_level = st.sidebar.selectbox(
    "Traffic Density Level",
    label_encoders["traffic_density_level"].classes_
)

road_type = st.sidebar.selectbox(
    "Road Type",
    label_encoders["road_type"].classes_
)

# ==========================================================
# PREPROCESS FUNCTION
# ==========================================================

def preprocess_input(data):
    df = pd.DataFrame([data])
    for col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])
    df = df[model.get_booster().feature_names]
    return df

# ==========================================================
# PREDICTION
# ==========================================================

if st.sidebar.button("🚀 Predict Travel Time"):

    input_data = {
        "distance_km": distance_km,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "weather_condition": weather_condition,
        "traffic_density_level": traffic_density_level,
        "road_type": road_type
    }

    processed_data = preprocess_input(input_data)
    prediction = float(model.predict(processed_data)[0])

    st.markdown("## 📊 Estimated Travel Time")

    st.metric(
        label="Travel Time (minutes)",
        value=f"{round(prediction, 2)} min"
    )

    if prediction < 20:
        st.success("🟢 Light Traffic Expected")
    elif prediction < 40:
        st.warning("🟡 Moderate Traffic Conditions")
    else:
        st.error("🔴 Heavy Traffic Expected")

    # Save prediction
    log_row = processed_data.copy()
    log_row["predicted_time"] = prediction
    log_row["timestamp"] = datetime.now()

    log_row.to_csv(log_file, mode="a", header=False, index=False)

# ==========================================================
# VISUALIZATION SECTION
# ==========================================================

st.markdown("---")
st.subheader("📈 Travel Time Trend Over Predictions")

logs = pd.read_csv(log_file)

if len(logs) > 0:

    logs = logs.reset_index()
    logs["Prediction Number"] = logs.index + 1

    chart_data = logs[["Prediction Number", "predicted_time"]]
    chart_data = chart_data.rename(
        columns={"predicted_time": "Travel Time (minutes)"}
    )

    fig = px.line(
        chart_data,
        x="Prediction Number",
        y="Travel Time (minutes)",
        markers=True,
        title="Travel Time Trend Over Predictions"
    )

    fig.update_layout(
        xaxis_title="Prediction Number (Each time you clicked Predict)",
        yaxis_title="Estimated Travel Time (minutes)"
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    col1.metric("Total Predictions Made", len(logs))
    col2.metric(
        "Average Travel Time",
        f"{round(logs['predicted_time'].mean(), 2)} min"
    )

else:
    st.info("No predictions yet. Enter details and click Predict.")
