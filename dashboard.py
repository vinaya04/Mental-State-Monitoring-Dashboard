import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import joblib 
from collections import deque

# ====================================================================
# CONFIGURATION AND FEATURE DEFINITIONS
# ====================================================================

st.set_page_config(layout="wide", page_title="Real-Time Cognitive State Dashboard")

# Constants
CSV_FILE = "CompleteFeatureEngineering.csv"
WORKLOAD_MODEL_FILE = "workload_model .pkl"
ACCURACY_MODEL_FILE = "accuracy_model_balanced.pkl"
EMOTION_MODEL_FILE = "emotion_model.pkl"

UPDATE_INTERVAL_SECONDS = 1.0 
HISTORY_SIZE = 30 

# Full list of 47 features (Used for data loading)
FULL_FEATURE_COLUMNS = [
    'eeg_mean_delta', 'eeg_mean_theta', 'eeg_mean_alpha', 'eeg_mean_beta', 'eeg_mean_gamma',
    'eeg_std_theta', 'eeg_std_alpha', 'eeg_std_beta', 'eeg_beta_alpha_ratio', 'eeg_theta_beta_ratio',
    'eye_mean_pupil', 'eye_fixation_count', 'eye_gaze_dispersion', 'ivt_mean_fix_dur', 'ivt_mean_sac_amp',
    'ivt_fixation_count', 'ivt_saccade_count', 'gsr_mean', 'gsr_max', 'gsr_std', 'gsr_peak_count',
    'tiva_mean_anger', 'tiva_max_anger', 'tiva_mean_contempt', 'tiva_max_contempt', 'tiva_mean_disgust',
    'tiva_max_disgust', 'tiva_mean_fear', 'tiva_max_fear', 'tiva_mean_joy', 'tiva_max_joy',
    'tiva_mean_sadness', 'tiva_max_sadness', 'tiva_mean_surprise', 'tiva_max_surprise', 'tiva_mean_engagement',
    'tiva_max_engagement', 'tiva_mean_valence', 'tiva_max_valence', 'tiva_mean_confusion', 'tiva_max_confusion',
    'tiva_mean_neutral', 'tiva_max_neutral', 'tiva_mean_attention', 'tiva_max_attention', 'tiva_mean_blinkrate',
    'tiva_max_blinkrate'
]


# --- Feature Subsets for Specific Models (Based on previous fixes) ---
WORKLOAD_FEATURES = [
    'eeg_mean_delta', 'eeg_mean_theta', 'eeg_mean_alpha', 'eeg_mean_beta', 'eeg_mean_gamma',
    'eeg_std_theta', 'eeg_std_alpha', 'eeg_std_beta', 'eeg_beta_alpha_ratio', 'eeg_theta_beta_ratio',
    'eye_mean_pupil', 'eye_fixation_count', 'eye_gaze_dispersion',
    'gsr_mean', 'gsr_max', 'gsr_std', 'gsr_peak_count'
]

ACCURACY_FEATURES = [
    'eeg_mean_delta', 'eeg_mean_theta', 'eeg_mean_alpha', 'eeg_mean_beta', 'eeg_mean_gamma',
    'eeg_std_theta', 'eeg_std_alpha', 'eeg_std_beta', 'eeg_beta_alpha_ratio', 'eeg_theta_beta_ratio',
    'eye_mean_pupil', 'eye_fixation_count', 'eye_gaze_dispersion',
    'ivt_saccade_count', 
    'gsr_mean', 'gsr_max', 'gsr_std', 'gsr_peak_count'
]

EMOTION_FEATURES = [
    'eeg_mean_delta', 'eeg_mean_theta', 'eeg_mean_alpha', 'eeg_mean_beta', 'eeg_mean_gamma',
    'eeg_std_theta', 'eeg_std_alpha', 'eeg_std_beta', 'eeg_beta_alpha_ratio', 'eeg_theta_beta_ratio',
    'eye_mean_pupil', 'eye_fixation_count', 'eye_gaze_dispersion',
    'gsr_mean', 'gsr_max', 'gsr_std', 'gsr_peak_count'
] 

WORKLOAD_LABELS = ['Low', 'Medium', 'High']
EMOTION_LABELS = ['Neutral', 'Joy', 'Sadness', 'Anger', 'Confusion']
ACCURACY_LABELS = [0, 1] 

# ====================================================================
# HELPER FUNCTIONS (No Changes)
# ====================================================================

@st.cache_resource
def load_models():
    """Load pretrained models using joblib."""
    try:
        workload_model = joblib.load(WORKLOAD_MODEL_FILE)
        accuracy_model = joblib.load(ACCURACY_MODEL_FILE)
        emotion_model = joblib.load(EMOTION_MODEL_FILE)
        st.success("Models loaded successfully!")
        return workload_model, accuracy_model, emotion_model
    except FileNotFoundError as e:
        st.error(f"Error loading model file: {e.filename}. Please ensure all .pkl files are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}. The files may be corrupted or in the wrong format.")
        st.stop()


@st.cache_data
def load_data():
    """Load the feature-engineered data for simulation."""
    try:
        data = pd.read_csv(CSV_FILE)
        data = data[FULL_FEATURE_COLUMNS].dropna().reset_index(drop=True) 
        if data.empty:
            st.error("The CSV data is empty or all rows were dropped after selecting feature columns.")
            st.stop()
        return data
    except FileNotFoundError:
        st.error(f"Error: {CSV_FILE} not found. Ensure it is in the same directory.")
        st.stop()
    except KeyError as e:
        st.error(f"Error: One or more feature columns are missing from {CSV_FILE}. Missing column: {e}")
        st.stop()


def get_predictions(data_row, w_model, a_model, e_model):
    """Make predictions for a single data point, using the correct feature subset for each model."""
    
    # 1. Workload Prediction (17 features)
    w_features = data_row[WORKLOAD_FEATURES].values.reshape(1, -1)
    workload_proba = w_model.predict_proba(w_features)[0]
    workload_pred_idx = np.argmax(workload_proba)
    workload_pred = WORKLOAD_LABELS[workload_pred_idx]
    workload_confidence = workload_proba[workload_pred_idx]

    # 2. Accuracy Prediction (18 features)
    a_features = data_row[ACCURACY_FEATURES].values.reshape(1, -1)
    accuracy_proba = a_model.predict_proba(a_features)[0]
    try:
        correct_idx = a_model.classes_.tolist().index(1) 
    except:
        correct_idx = 1
    accuracy_prob_correct = accuracy_proba[correct_idx]
    accuracy_pred = "Correct" if accuracy_prob_correct >= 0.5 else "Incorrect"

    # 3. Emotion Prediction (17 features)
    e_features = data_row[EMOTION_FEATURES].values.reshape(1, -1)
    emotion_proba = e_model.predict_proba(e_features)[0]
    emotion_pred_idx = np.argmax(emotion_proba)
    emotion_label = EMOTION_LABELS[emotion_pred_idx]
    emotion_confidence = emotion_proba[emotion_pred_idx]

    return {
        'workload': workload_pred,
        'workload_confidence': workload_confidence,
        'accuracy_pred': accuracy_pred,
        'accuracy_prob_correct': accuracy_prob_correct,
        'emotion': emotion_label,
        'emotion_confidence': emotion_confidence,
        'timestamp': time.time()
    }

# --- Visualization Functions (No Changes) ---

def workload_gauge_chart(current_workload, confidence):
    level_map = {'Low': 0.165, 'Medium': 0.5, 'High': 0.835}
    value = level_map.get(current_workload, 0)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': f"Current Workload (Confidence: **{confidence:.1%}**)"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'shape': "angular",
            'axis': {'range': [0, 1], 'tickvals': [0, 0.33, 0.67, 1], 'ticktext': ['', 'Low', 'Medium', 'High'], 'visible': False},
            'bar': {'color': "darkgray", 'thickness': 0.1},
            'steps': [
                {'range': [0, 0.33], 'color': "green"},
                {'range': [0.33, 0.67], 'color': "yellow"},
                {'range': [0.67, 1], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value 
            }
        }))
    
    fig.update_layout(height=300, margin=dict(t=50, b=0, l=0, r=0))
    return fig

def accuracy_line_chart(history_df):
    fig = px.line(history_df, x='Timestamp_Relative', y='Accuracy_Prob', 
                  title='Accuracy Probability Over Time',
                  labels={'Timestamp_Relative': 'Time (seconds ago)', 'Accuracy_Prob': 'P(Correct)'})
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold (0.5)", annotation_position="top right")
    
    fig.update_layout(
        yaxis_range=[0, 1],
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_xaxes(autorange="reversed") 
    return fig

def emotion_timeline_chart(history_df):
    history_df['Timeline_Y'] = 1
    
    fig = px.bar(history_df, x='Timestamp_Relative', y='Timeline_Y', color='Emotion',
                 title='Emotion Timeline Snapshot (Latest States)',
                 labels={'Timeline_Y': '', 'Timestamp_Relative': 'Time (seconds ago)'},
                 orientation='v',
                 color_discrete_map={
                     'Neutral': 'gray', 'Joy': 'green', 'Sadness': 'blue', 
                     'Anger': 'red', 'Confusion': 'purple'
                 })
    
    fig.update_layout(
        yaxis={'visible': False, 'showticklabels': False}, 
        barmode='stack',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_xaxes(autorange="reversed") 
    return fig

# ====================================================================
# MAIN STREAMLIT APP (LAYOUT ADJUSTMENT APPLIED HERE)
# ====================================================================

def run_dashboard():
    st.title("Mental State Monitoring Dashboard 🧠")

    
    w_model, a_model, e_model = load_models()
    data = load_data()
    
    if 'history_data' not in st.session_state:
        st.session_state.history_data = deque(maxlen=HISTORY_SIZE)
        
    # 1. Define containers for the main charts (Top Row)
    workload_col, accuracy_col, emotion_col = st.columns([1, 1, 1])
    
    with workload_col:
        st.subheader("3.1 Mental Workload Panel")
        workload_chart_placeholder = st.empty()

    with accuracy_col:
        st.subheader("3.2 Accuracy Probability Panel")
        accuracy_chart_placeholder = st.empty()

    with emotion_col:
        st.subheader("3.3 Emotion Transitions Panel (Timeline View)")
        emotion_chart_placeholder = st.empty()

    st.markdown("---")
    
    # 2. Define container for the Summary Panel (Below Charts)
    summary_container = st.container()
    
    status_text = st.empty() 
    
    current_time_start = time.time()
    
    for i in range(len(data)):
        current_data_row = data.iloc[i]
        
        results = get_predictions(current_data_row, w_model, a_model, e_model)
        
        current_timestamp = results['timestamp']
        new_row = {
            'Timestamp_Absolute': current_timestamp,
            'Timestamp_Relative': round(current_timestamp - current_time_start, 2),
            'Workload': results['workload'],
            'Workload_Confidence': results['workload_confidence'],
            'Accuracy_Prob': results['accuracy_prob_correct'],
            'Accuracy_Pred': results['accuracy_pred'],
            'Emotion': results['emotion'],
            'Emotion_Confidence': results['emotion_confidence']
        }
        
        st.session_state.history_data.append(new_row)
        
        history_df = pd.DataFrame(st.session_state.history_data)
        history_df['Timestamp_Relative'] = history_df['Timestamp_Absolute'].apply(lambda t: round(current_timestamp - t, 2))
        
        latest = history_df.iloc[-1]
        
        # 3. Update Charts (Now above the Summary Panel)
        fig_workload = workload_gauge_chart(latest['Workload'], latest['Workload_Confidence'])
        workload_chart_placeholder.plotly_chart(fig_workload, use_container_width=True, key=f"workload_gauge_{i}")

        fig_accuracy = accuracy_line_chart(history_df)
        accuracy_chart_placeholder.plotly_chart(fig_accuracy, use_container_width=True, key=f"accuracy_line_{i}")

        fig_emotion = emotion_timeline_chart(history_df)
        emotion_chart_placeholder.plotly_chart(fig_emotion, use_container_width=True, key=f"emotion_timeline_{i}")
        
        # 4. Update Participant Summary Panel (Now below charts)
        with summary_container:
            # Clear the container before redrawing the metrics
            summary_container.empty() 
            st.markdown("### 3.4 Participant Summary Panel (Last Second Snapshot)")
            col1, col2, col3, col4 = st.columns(4)
            
            # Using f-string for metrics for clarity in this example
            col1.metric("Current Workload", latest['Workload'], delta=f"{latest['Workload_Confidence']:.1%} Confidence")
            col2.metric("Predicted Accuracy", latest['Accuracy_Pred'], delta=f"{latest['Accuracy_Prob']:.1%} Probability")
            col3.metric("Current Emotion", latest['Emotion'], delta=f"{latest['Emotion_Confidence']:.1%} Confidence")
            col4.metric("Data Index (N)", i + 1)
        
        status_text.info(f"Streaming data point **{i + 1} / {len(data)}**. Next update in {UPDATE_INTERVAL_SECONDS}s.")

        time.sleep(UPDATE_INTERVAL_SECONDS)

    status_text.success(f"Simulation Complete. Total **{len(data)}** data points processed.")
    
    st.sidebar.markdown(f"**STEP 5: Evaluation & Testing**")
    st.sidebar.markdown(f"The simulation ran at an update rate of **{UPDATE_INTERVAL_SECONDS} second(s)**. To perform a **stress test**, simply modify the `UPDATE_INTERVAL_SECONDS` constant in the script (e.g., to `0.5` or `0.1`).")


if __name__ == '__main__':
    run_dashboard()