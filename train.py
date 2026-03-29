import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SkLR

from model import LinearRegression

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Train your model
model = LinearRegression(lr=0.01, epoches=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    return 1 - (np.sum((y_true - y_pred)**2) /
                np.sum((y_true - y_mean)**2))

print("Your Model R2:", r2_score(y_test, y_pred))

# Sklearn comparison
sk_model = SkLR()
sk_model.fit(X_train, y_train)
print("Sklearn R2:", sk_model.score(X_test, y_test))

# Plot loss
plt.plot(model.losses)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("loss.png")
plt.show()

# Plot predictions
plt.scatter(y_test, y_pred)

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.savefig("prediction.png")
plt.show()
