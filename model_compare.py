import sys
import subprocess

print("\n—— FULL MODEL PERFORMANCE SUMMARY ——")

# Regression
print("\nRunning Linear Regression Model...")
subprocess.run([sys.executable, "src/train_regression.py"])

# Classification
print("\nRunning Logistic Classification Model...")
subprocess.run([sys.executable, "src/train_classification.py"])

# Regularization models
print("\nRunning Regularization Models (L1 / L2)...")
subprocess.run([sys.executable, "src/regularized_models.py"])

# Custom gradient descent implementation
print("\nRunning Custom Gradient Descent Implementation...")
subprocess.run([sys.executable, "src/gradient_descent_regression.py"])

print("\n—— All Evaluation Complete ——")
