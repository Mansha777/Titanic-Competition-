# STUDENT PLACEMENT DATA ANALYSIS PROJECT

# 1. IMPORTING LIBRARIES
# Pandas is used for reading and analyzing tabular data
import pandas as pd

# NumPy is used for mathematical/statistical operations
import numpy as np

# Matplotlib is used for creating graphs and visualizations
import matplotlib.pyplot as plt

# Seaborn is used for advanced visualizations like heatmaps
import seaborn as sns

# 2. LOADING THE DATASET

# read_csv() reads the CSV dataset file
# df stands for DataFrame (table-like structure)

df = pd.read_csv(r"C:\Users\abc\Desktop\Data_analysis\test.csv")

# 3. EXPLORING THE DATASET

# head() shows the first 5 rows of the dataset
print(df.head())

# info() gives information about:
# - number of rows
# - columns
# - data types
# - missing values
print(df.info())

# describe() gives statistical summary like:
# mean, min, max, standard deviation etc.
print(df.describe())

# 4. CHECKING FOR MISSING VALUES


# isnull() checks empty values
# sum() counts total empty values in each column

print(df.isnull().sum())



# 5. CREATING CATEGORIES

# Instead of using raw values like:
# 5.8, 7.2, 8.9 etc.
# We categorize them into:
# Low, Medium, High


# CGPA Categories
df['CGPA_Category'] = pd.cut( df['CGPA'], bins=[0, 6, 7.5, 10], labels=['Low', 'Medium', 'High'])   # category names

# Coding Skill Categories
df['Coding_Level'] = pd.cut(df['Coding_Skills'],bins=[0, 4, 7, 10],labels=['Low', 'Medium', 'High'])

# Aptitude Categories
df['Aptitude_Level'] = pd.cut(df['Aptitude_Test_Score'],bins=[0, 50, 75, 100],labels=['Low', 'Medium', 'High'])

# Communication Skill Categories
df['Comm_Level'] = pd.cut(df['Communication_Skills'],bins=[0, 4, 7, 10],labels=['Low', 'Medium', 'High'])


# 6. ANALYSIS USING CROSSTAB


# Crosstab compares two columns together
# normalize='index'
# converts raw counts into percentages
# *100 converts decimal into percentage


# Internships vs Placement
print("\nInternships vs Placement")
print(
    pd.crosstab(
        df['Internships'],
        df['Placement_Status'],
        normalize='index'
    ) * 100
)

# CGPA vs Placement
print("\nCGPA Category vs Placement")

print(
    pd.crosstab(
        df['CGPA_Category'],
        df['Placement_Status'],
        normalize='index'
    ) * 100
)

# Aptitude vs Placement
print("\nAptitude Level vs Placement")

print(
    pd.crosstab(
        df['Aptitude_Level'],
        df['Placement_Status'],
        normalize='index'
    ) * 100
)

# Coding Skills vs Placement
print("\nCoding Skills vs Placement")

print(
    pd.crosstab(
        df['Coding_Level'],
        df['Placement_Status'],
        normalize='index'
    ) * 100
)

# Communication Skills vs Placement
print("\nCommunication Skills vs Placement")

print(
    pd.crosstab(
        df['Comm_Level'],
        df['Placement_Status'],
        normalize='index'
    ) * 100
)

# Degree vs Placement
print("\nDegree vs Placement")

print(
    pd.crosstab(
        df['Degree'],
        df['Placement_Status'],
        normalize='index'
    ) * 100
)



# 7. NUMPY USAGE

# mean() calculates average value
print("\nMean CGPA:", np.mean(df['CGPA']))

# median() calculates middle value
print("Median Aptitude Score:",
      np.median(df['Aptitude_Test_Score']))

# 8. DATA VISUALIZATION

# GRAPH 1 : BAR GRAPH
# Internships vs Placement

pd.crosstab(
    df['Internships'],df['Placement_Status'], normalize='index').plot(kind='bar')

plt.title('Internships vs Placement')
plt.xlabel('Number of Internships')
plt.ylabel('Percentage')
plt.show()


# GRAPH 2 : SCATTER PLOT
# CGPA vs Aptitude Score


# Colors for placement status
colors = {
    'Placed': 'blue',
    'Not Placed': 'red'
}

# scatter() creates scatter plot
# each dot represents one student

plt.scatter(df['CGPA'],df['Aptitude_Test_Score'],c=df['Placement_Status'].map(colors),alpha=0.6)

plt.title('CGPA vs Aptitude Score')
plt.xlabel('CGPA')
plt.ylabel('Aptitude Test Score')


# Creating legend manually
plt.scatter([], [], color='blue', label='Placed')
plt.scatter([], [], color='red', label='Not Placed')
plt.legend()
plt.show()


# GRAPH 3 : BOX PLOT
# Coding Skills by Placement Status


# boxplot() compares data distribution

df.boxplot(column='Coding_Skills', by='Placement_Status')
plt.title('Coding Skills by Placement Status')

# removes unnecessary automatic title
plt.suptitle('')
plt.ylabel('Coding Skills')
plt.show()

# GRAPH 4 : PIE CHART
# Placement Distribution

# value_counts() counts total occurrences

placement_counts = df['Placement_Status'].value_counts()

plt.pie(placement_counts,labels=placement_counts.index,autopct='%1.1f%%')
plt.title('Placement Distribution')
plt.show()



# GRAPH 5 : LINE GRAPH
# Placement Rate vs Internships


placement_rate = pd.crosstab(df['Internships'],df['Placement_Status'],normalize='index') * 100

plt.plot( placement_rate.index,placement_rate['Placed'],marker='o')
plt.title('Placement Rate vs Internships')
plt.xlabel('Number of Internships')
plt.ylabel('Placement Percentage')
plt.show()



# GRAPH 6 : HISTOGRAM
# Distribution of CGPA


# histogram shows distribution of values

plt.hist( df['CGPA'],bins=10)
plt.title('Distribution of CGPA')
plt.xlabel('CGPA')
plt.ylabel('Number of Students')
plt.show()



# GRAPH 7 : HEATMAP
# Correlation Heatmap


# corr() calculates relationships between columns

correlation = df[['CGPA','Coding_Skills','Aptitude_Test_Score','Communication_Skills','Internships']].corr()


# heatmap visualizes correlation values

plt.figure(figsize=(7, 5))
sns.heatmap(correlation,annot=True, cmap='coolwarm', linewidths=0.5)

plt.title('Correlation Heatmap')
plt.show()