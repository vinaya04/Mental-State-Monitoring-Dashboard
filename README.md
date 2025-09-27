# Mental-State-Monitoring-Dashboard
Real-Time Cognitive State Monitoring Dashboard 🧠

This project implements a real-time dashboard using Streamlit and Plotly to monitor a participant's cognitive state, including Mental Workload, Accuracy Probability, and Emotion Transitions, using predictions from pre-trained machine learning models.

🚀 Key Features
Real-Time Data Simulation: Replays a session log (CompleteFeatureEngineering.csv) to simulate live sensor data streaming.
Three Concurrent ML Predictions: Uses three separate RandomForestClassifier models for continuous state prediction.

Dynamic Visualizations:
Workload: Real-time Gauge Chart (Low/Med/High).
Accuracy: Line Chart of predicted probability over time with a 0.5 confidence threshold.
Emotion: Timeline Bar Chart showing emotion state transitions.
Participant Snapshot: Displays current workload, accuracy, and emotion metrics with confidence percentages.

⚙️ Setup and Installation
Prerequisites
Python 3.8+
Git

For rapid development and iteration on the dashboard layout and real-time logic, the core Python files (dashboard.py, utils.py, and components.py) were initially consolidated into a single file, dashboard.py.
