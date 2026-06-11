# Titanic Survival Prediction — ML Model
# =========================================
# A beginner machine learning project that predicts whether a
# Titanic passenger survived, using a Decision Tree Classifier.
#
# Accuracy: 80%
#
#
# PROJECT STRUCTURE
# -----------------
# titanic_model.py   → Main script
# submission.csv     → Kaggle submission file
# train.csv          → Training data (from Kaggle)
# test.csv           → Test data (from Kaggle)
#
#
# WORKFLOW
# --------
# Step 1: Load & Explore
#   - Loaded train.csv (891 rows, 12 columns)
#   - Inspected shape, data types, and missing values
#
# Step 2: Clean the Data
#   - Age      → filled missing values with median
#   - Cabin    → dropped (too many nulls)
#   - Embarked → filled missing values with mode
#   - Fare     → filled missing values with median (in test set)
#
# Step 3: Prepare Features
#   - Encoded Sex      → male=1, female=0
#   - Encoded Embarked → S=0, C=1, Q=2
#   - Features used: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
#   - Split: 80% train / 20% test (random_state=42)
#
# Step 4: Train the Model
#   - Algorithm: DecisionTreeClassifier (scikit-learn)
#   - max_depth=5 to prevent overfitting
#
# Step 5: Evaluate
#   - Accuracy: 80%
#   - Confusion Matrix:
#       [[95  10]
#        [26  48]]
#     95 → correctly predicted deaths
#     48 → correctly predicted survivors
#     10 → missed survivors (false negatives)
#     26 → predicted survival, actually died (false positives)
#
# Step 6: Generate Submission
#   - Ran model on test.csv
#   - Saved predictions to submission.csv for Kaggle
#
#
# HOW TO RUN
# ----------
# pip install pandas scikit-learn
# python titanic_model.py
#
#
# TECH STACK
# ----------
# Python, Pandas, scikit-learn
