import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Set
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(filepath: str, required_columns: Set[str] = None, drop_na: bool = True) -> Optional[pd.DataFrame]:
    """
    Load and preprocess credit card fraud data with comprehensive error handling.
    Updated to work with the detailed transactional dataset format.

    Args:
        filepath: Path to the dataset CSV file
        required_columns: Set of columns that must be present
                         (default: {'trans_date_trans_time', 'amt', 'is_fraud'})
        drop_na: Flag to drop rows with NaN values (default: True)

    Returns:
        Preprocessed DataFrame or None if loading fails
    """
    # Set default required columns for this dataset format
    if required_columns is None:
        required_columns = {'trans_date_trans_time', 'amt', 'is_fraud'}

    try:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Attempting to load data from: {filepath}")

        # 1. File validation
        if not os.path.exists(filepath):
            abs_path = os.path.abspath(filepath)
            raise FileNotFoundError(f"File not found at path: {abs_path}")

        logger.info("✓ File exists")
        logger.info(f"File size: {os.path.getsize(filepath) / 1024:.2f} KB")

        # 2. Load data
        logger.info("\nLoading CSV file...")
        df = pd.read_csv(filepath)
        logger.info(f"✓ Successfully loaded. Initial shape: {df.shape}")

        # 3. Validate columns
        logger.info("\nChecking required columns...")
        missing_columns = required_columns - set(df.columns)
        available_columns = set(df.columns)

        if missing_columns:
            logger.error("✗ Missing columns:")
            for col in missing_columns:
                logger.error(f"- {col}")

            logger.info("\nAvailable columns:")
            logger.info(list(available_columns))
            return None

        logger.info("✓ All required columns present")

        # 4. Data cleaning
        logger.info("\nCleaning data...")
        df = df.copy()  # Avoid SettingWithCopyWarning

        # Handle null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.info("Null values detected:")
            logger.info(null_counts[null_counts > 0])
            if drop_na:
                initial_count = len(df)
                df = df.dropna()
                logger.info(f"Dropped {initial_count - len(df)} rows with nulls")
            else:
                logger.info("Rows with null values were not dropped, missing data will remain")
        else:
            logger.info("✓ No null values found")

        # 5. Feature engineering
        logger.info("\nEngineering features...")

        # Convert transaction time to datetime and extract features
        if 'trans_date_trans_time' in df.columns:
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
            df['hour_of_day'] = df['trans_date_trans_time'].dt.hour
            df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
            df['month'] = df['trans_date_trans_time'].dt.month
            logger.info("✓ Extracted temporal features from transaction timestamp")

        # Create age feature from dob if available
        if 'dob' in df.columns:
            df['dob'] = pd.to_datetime(df['dob'])
            df['age'] = (pd.to_datetime('today') - df['dob']).dt.days // 365
            logger.info("✓ Calculated customer age from date of birth")

        # Encode categorical variables
        categorical_cols = ['category', 'gender', 'state']  # Add others as needed
        label_encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
                logger.info(f"✓ Label encoded {col}")

        # 6. Feature scaling
        logger.info("\nScaling features...")
        scaler = StandardScaler()
        numeric_cols = ['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long']
        numeric_cols = [col for col in numeric_cols if col in df.columns]

        if numeric_cols:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            logger.info(f"✓ Scaled numeric columns: {numeric_cols}")
        else:
            logger.info("No numeric columns found for scaling")

        # 7. Target analysis
        if 'is_fraud' in df.columns:
            logger.info("\nTarget variable analysis:")
            class_dist = df['is_fraud'].value_counts(normalize=True)
            logger.info(f"Non-fraud (0): {class_dist.get(0, 0):.4%}")
            logger.info(f"Fraud (1): {class_dist.get(1, 0):.4%}")
            if class_dist.get(1, 0) > 0:
                logger.info(f"Imbalance ratio: {class_dist.get(0, 0) / class_dist.get(1, 0):.1f}:1")

        logger.info("\n✓ Data preprocessing completed successfully")
        logger.info(f"Final shape: {df.shape}")
        return df

    except pd.errors.EmptyDataError:
        logger.error("\n✗ Error: The file is empty or corrupt")
    except UnicodeDecodeError:
        logger.error("\n✗ Error: File encoding issue. Try specifying encoding='latin1'")
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {str(e)}")

    return None


if __name__ == "__main__":
    # Try multiple possible file locations
    possible_paths = [
        "data/fraudTest/fraudTest.csv",  # Relative path
        "../data/fraudTest/fraudTest.csv",  # Alternative relative path
        "D:/credit-card-fraud-detection/data/fraudTest/fraudTest.csv",  # Absolute path
        os.path.join(os.path.dirname(__file__), "..", "data", "fraudTest", "fraudTest.csv")  # Robust relative path
    ]

    logger.info("=" * 50)
    logger.info("Credit Card Fraud Detection - Data Preprocessing")
    logger.info("=" * 50)

    for i, path in enumerate(possible_paths, 1):
        logger.info(f"\nAttempt {i}: Trying path '{path}'")
        processed_data = load_and_preprocess_data(
            filepath=path,
            required_columns={'trans_date_trans_time', 'amt', 'is_fraud'}
        )

        if processed_data is not None:
            logger.info("\nSuccess! Sample of processed data:")
            logger.info(processed_data.head(3))
            logger.info("\nData types:")
            logger.info(processed_data.dtypes)
            break
    else:
        logger.error("\nFailed to load data after all attempts. Please:")
        logger.error("- Check the file exists at one of these locations:")
        for path in possible_paths:
            logger.error(f"  - {os.path.abspath(path) if not os.path.isabs(path) else path}")
        logger.error("- Verify the filename is exactly 'fraudTest.csv'")
        logger.error("- Check file permissions")

