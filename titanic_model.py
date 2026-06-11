#Step:1 Load and Explore the Data
import pandas as pd 
df = pd.read_csv(r"D:\Data_analysis\titanic (1)\train.csv")
print(df.shape)
print(df.head())
print(df.info())

#Step:2 Clean the Data
#fill missing Age with median 
df['Age'] =df['Age'].fillna(df['Age'].median())

#drop cabin 
df.drop(columns = ['Cabin'], inplace = True)

#Fill missing Embarked with mode 
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

#confirm no more nulls 
print(df.isnull().sum())

# Step:3 Prepare features and target variable
#convert Sex to numbers (male =1, female=0)
df['Sex'] = df['Sex'].map({'male': 1, 'female':0})

#convert Embarked to numbers (s=0, c=1, q =2)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

#split into train and test (ik we have different files but just for practice we will split this)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42 )
print(X_train.shape, X_test.shape)

#712 rows for training and 179 rows for testing 

#Step:4 Train the Model 

from sklearn.tree import DecisionTreeClassifier 
model = DecisionTreeClassifier (random_state = 42, max_depth = 5)
model.fit(X_train, y_train)

print("Model Trained!")

#Step: 5 Evaluate the Model 
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

#confusion matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, model.predict(X_test))
print(cm)

#Step:6 Make Predictions on New Data

#load test drive 
test_df = pd.read_csv(r"D:\Data_analysis\titanic (1)\test.csv")

#clean test data same way as train data 

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df.drop(columns = ['Cabin'], inplace = True)
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

print(test_df.isnull().sum())

#prepare features 
test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female':0})

#convert Embarked to numbers (s=0, c=1, q =2)
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']
X_test_final = test_df[features]

predictions = model.predict(X_test_final)
print(predictions)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})

submission.to_csv(r"D:\Data_analysis\submission.csv", index=False)
print("Submission saved!")