import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configure page
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detector")


@st.cache_resource
def load_model():
    """Load the pre-trained model and preprocessor"""
    try:
        # Try loading a full pipeline first
        return joblib.load("src/models/fraud_detection_pipeline.pkl")
    except:
        try:
            # Fallback to separate model and preprocessor
            model = joblib.load("models/random_forest_model.pkl")
            preprocessor = joblib.load("models/logistic_model.pkl")
            return Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()


# Load model/pipeline
pipeline = load_model()


def validate_input_data(df):
    """Check if uploaded data has required columns"""
    required_columns = {
        'amt': 'numeric',
        'category': 'categorical',
        'gender': 'categorical',
        'city_pop': 'numeric',
        'lat': 'numeric',
        'long': 'numeric',
        'merch_lat': 'numeric',
        'merch_long': 'numeric',
        'state': 'categorical'
    }

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()
    return True


def preprocess_input(df):
    """Preprocess the input data to match training format"""
    # Make copy to avoid modifying original
    df = df.copy()

    # Convert date fields if they exist
    date_cols = ['trans_date_trans_time', 'dob']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Feature engineering (must match training)
    if 'trans_date_trans_time' in df.columns:
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek

    if 'dob' in df.columns:
        df['age'] = (pd.to_datetime('today') - df['dob']).dt.days // 365

    # Drop columns not used in training
    drop_cols = ['trans_date_trans_time', 'merchant', 'first', 'last',
                 'street', 'city', 'job', 'dob', 'trans_num']
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')


# File uploader
uploaded_file = st.file_uploader("Upload transaction data (CSV)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

    # Validate input data
    validate_input_data(df)

    # Show raw data
    with st.expander("🔍 Raw Data Preview"):
        st.dataframe(df.head())

    # Preprocess data
    try:
        processed_df = preprocess_input(df)

        # Make predictions
        with st.spinner("Making predictions..."):
            predictions = pipeline.predict(processed_df)
            probas = pipeline.predict_proba(processed_df)[:, 1]  # Fraud probability

            # Add results to dataframe
            results = df.copy()
            results['Fraud Prediction'] = predictions
            results['Fraud Probability'] = probas

            # Display results
            st.success(f"✅ Analysis complete! Processed {len(df)} transactions")

            # Show fraud cases
            fraud_cases = results[results['Fraud Prediction'] == 1]
            if not fraud_cases.empty:
                st.subheader(f"🚨 Detected {len(fraud_cases)} potentially fraudulent transactions")
                st.dataframe(fraud_cases.sort_values('Fraud Probability', ascending=False))

                # Download results
                csv = fraud_cases.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Fraud Cases",
                    data=csv,
                    file_name="fraud_cases.csv",
                    mime="text/csv"
                )
            else:
                st.success("No fraudulent transactions detected")

            # Show summary stats
            with st.expander("📊 Prediction Statistics"):
                col1, col2 = st.columns(2)
                col1.metric("Total Transactions", len(df))
                col2.metric("Fraud Rate", f"{len(fraud_cases) / len(df):.2%}")

                st.bar_chart(results['Fraud Probability'].value_counts(bins=10))

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.exception(e)