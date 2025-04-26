from src.data_preprocessing import load_and_preprocess_data
from src.model_building import train_and_evaluate_model

def main():
    print("=" * 60)
    print("           Credit Card Fraud Detection - Pipeline Start")
    print("=" * 60)

    print("\n🔄 Attempting to load and preprocess data...")
    possible_paths = [
        "data/fraudTest/fraudTest.csv",
        "../data/fraudTest/fraudTest.csv",
        "D:/credit-card-fraud-detection/data/fraudTest/fraudTest.csv"
    ]

    df = None
    for path in possible_paths:
        df = load_and_preprocess_data(path)
        if df is not None:
            print(f"\n✅ Data loaded from: {path}")
            break

    if df is not None:
        print("\n🚀 Starting model training and evaluation...")
        train_and_evaluate_model(df)
    else:
        print("\n❌ Failed to load and preprocess data from all provided paths.")

if __name__ == "__main__":
    main()
