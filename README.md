
#  Titanic Survival Prediction — ML Model

A beginner machine learning project that predicts whether a Titanic passenger survived, using a **Decision Tree Classifier**.

Built as part of the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic).

---

##  Results

| Metric | Score |
|--------|-------|
| Model | Decision Tree Classifier |
| Accuracy | **80%** |

---

##  Project Structure

```
titanic-ml/
├── titanic_model.py      # Model training and prediction script
├── submission.csv        # Kaggle submission file
└── README.md
```

---

##  How It Works

1. **Data Cleaning** — handled missing values in `Age`, `Embarked`, and other columns
2. **Feature Selection** — selected relevant features for prediction
3. **Model Training** — trained a Decision Tree Classifier using scikit-learn
4. **Prediction** — generated survival predictions on the test set
5. **Submission** — exported results to `submission.csv` for Kaggle

---

##  Tech Stack

- Python
- Pandas
- Scikit-learn
- NumPy

---

##  Run It Yourself

```bash
git clone https://github.com/Mansha777/titanic-ml.git
cd titanic-ml
pip install pandas scikit-learn numpy
python titanic_model.py
```

---

##  Notes

This is a learning project built while studying the Kaggle Intro to ML course.
Next steps: try Random Forest or XGBoost to improve accuracy.

---

*By [Mansha777](https://github.com/Mansha777)*
