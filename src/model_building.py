import pandas as pd
import os
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # Changed from sklearn.pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
import numpy as np
from pathlib import Path

# Logger setup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and validation"""

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load and validate raw data"""
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found at {file_path}")

            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise


class DataPreprocessor:
    """Handles all data preprocessing"""

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for modeling"""
        try:
            # Drop unnecessary columns
            cols_to_drop = [
                'trans_date_trans_time', 'merchant', 'first', 'last',
                'street', 'city', 'job', 'dob', 'trans_num'
            ]
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

            # Standardize categorical values
            if 'category' in df.columns:
                df['category'] = df['category'].str.lower().str.strip()

            return df

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise


class FraudDetectionModel:
    """Main model training and evaluation class"""

    def __init__(self):
        self.categorical_features = ['category', 'gender', 'state']
        self.numerical_features = ['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long']
        self.target = 'is_fraud'

    def create_pipeline(self) -> Pipeline:
        """Create the complete ML pipeline"""
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ],
            remainder='drop'
        )

        return Pipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            ))
        ])

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """Train and evaluate the model"""
        try:
            # Validate input
            required_cols = self.categorical_features + self.numerical_features + [self.target]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Split data
            X = df.drop(columns=[self.target])
            y = df[self.target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Create and train pipeline
            pipeline = self.create_pipeline()

            logger.info("Starting model training...")
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Evaluate
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'training_time': training_time
            }

            return pipeline, metrics

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def save_model(self, pipeline: Pipeline, path: str) -> None:
        """Save the trained pipeline"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(pipeline, path)
            logger.info(f"Model saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise


def main():
    """Main execution function"""
    try:
        # Configuration
        DATA_PATHS = [
            "data/fraudTest/fraudTest.csv",
            "../data/fraudTest/fraudTest.csv",
            "D:/credit-card-fraud-detection/data/fraudTest/fraudTest.csv"
        ]
        MODEL_SAVE_PATH = "models/fraud_detection_pipeline.pkl"

        # Load data
        for path in DATA_PATHS:
            try:
                raw_df = DataLoader.load_data(path)
                break
            except:
                continue
        else:
            raise FileNotFoundError("Could not find data file in any of the specified paths")

        # Preprocess data
        df = DataPreprocessor.preprocess(raw_df)

        # Train model
        model = FraudDetectionModel()
        pipeline, metrics = model.train(df)

        # Display results
        print("\nModel Evaluation Results:")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])
        print(f"\nROC AUC Score: {metrics['roc_auc']:.4f}")
        print(f"Training Time: {metrics['training_time']:.2f} seconds")

        # Save model
        model.save_model(pipeline, MODEL_SAVE_PATH)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    print("=" * 50)
    print("Credit Card Fraud Detection - Training Pipeline")
    print("=" * 50)
    main()