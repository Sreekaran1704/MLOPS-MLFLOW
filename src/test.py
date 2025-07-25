import mlflow

print("The tracking URI scheme is :")
print(mlflow.get_tracking_uri())
print('\n')

mlflow.set_tracking_uri("http://localhost:5000")
print("The tracking URI scheme is :")
print(mlflow.get_tracking_uri())
print('\n')