from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    '''Evaluates the model with confusion matrix and classification report'''
    print("\nEvaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # 1. Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 2. Classification Report with zero_division handling
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))  # prevents division-by-zero warnings

# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Create dummy data for testing
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Train a simple model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
