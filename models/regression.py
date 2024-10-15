from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def logistic_regression(X, y, llm_config):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    model = LogisticRegression(max_iter=1000)
    print (f"Using {len(X_train)} samples for training")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    return accuracy