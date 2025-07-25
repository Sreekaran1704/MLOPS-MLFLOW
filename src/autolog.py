import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

mlflow.set_experiment("YT_Experiment-1")

mlflow.autolog()
mlflow.set_tracking_uri("http://localhost:5000")

# load wine dataset
wine = load_wine()
x = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

max_depth = 10
n_estimators = 5


with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")

    #log artifacts using mlflow
    mlflow.log_artifact(__file__)

    #Log tags
    mlflow.set_tags({"Author": "Sreekaran", "Practice": "YT-MLFLOW"})

    print(f"Accuracy: {accuracy}")