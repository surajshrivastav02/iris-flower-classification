from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("IRIS FLOWER CLASSIFICATION")
    print("=" * 30)
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

    # Example measurements: sepal length, sepal width, petal length, petal width (cm)
    sample = [[5.1, 3.5, 1.4, 0.2]]
    predicted_index = model.predict(sample)[0]
    print("Sample measurements:", sample[0])
    print("Predicted species:", iris.target_names[predicted_index])

if __name__ == "__main__":
    main()
