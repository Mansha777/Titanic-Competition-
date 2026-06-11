
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────
# Step 1: Load & Explore
# ─────────────────────────────────────────
df = pd.read_csv(r"D:\Data_analysis\titanic (1)\train.csv")
print(df.shape)
print(df.head())
print(df.info())

# ─────────────────────────────────────────
# Step 2: Clean the Data
# ─────────────────────────────────────────
df['Age'] = df['Age'].fillna(df['Age'].median())
df.drop(columns=['Cabin'], inplace=True)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
print(df.isnull().sum())

# ─────────────────────────────────────────
# Step 3: Prepare Features
# ─────────────────────────────────────────
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

# ─────────────────────────────────────────
# Step 4: Train the Model
# ─────────────────────────────────────────
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)
print("Model Trained!")

# ─────────────────────────────────────────
# Step 5: Evaluate the Model
# ─────────────────────────────────────────
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, model.predict(X_test))
print(cm)

# ─────────────────────────────────────────
# Step 6: Predict on Test Data & Save
# ─────────────────────────────────────────
test_df = pd.read_csv(r"D:\Data_analysis\titanic (1)\test.csv")

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df.drop(columns=['Cabin'], inplace=True)
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0})
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X_test_final = test_df[features]
predictions = model.predict(X_test_final)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})
submission.to_csv(r"D:\Data_analysis\submission.csv", index=False)
print("Submission saved!")