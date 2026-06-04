# importing the libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset 
df = pd.read_csv(r"D:\Data_analysis\startup_founder_burnout_2026.csv")

#exploring the dataset 
print(df.head())
print(df.describe())
print(df.info())

#dropping the unnecessary columns
df = df.drop(columns=['Work_Mode', 'Work_Life_Balance_Score', 'Seeks_Mental_Health_Support', 'Founder_Burnout_Flag', 'Shutdown_Probability', 'Shutdown_Risk'])
print(df.columns.tolist())
print(df.shape)

#checking for missing values
print(df.isnull().sum())

#creating categories for some columns 

#sleep hours categories
df['Sleep_Hours_Category'] = pd.cut(df['Sleep_Hours'], bins =[0,5,7,10], labels = ['Poor','Moderate','Good'])

#weekly work hours categories
df['Weekly_Work_Hours_Category'] = pd.cut(df['Weekly_Work_Hours'], bins =[0,40,60,100], labels = ['Normal','Overworked','Extreme'])

#runway months remaining categories 
df['Runway_Months_Remaining_Category'] = pd.cut(df['Runway_Months_Remaining'], bins =[0,6,12,24], labels = ['Critical','Tight','Safe'])

#Founder Age categories]
df['Founder_Age_Category'] = pd.cut(df['Founder_Age'], bins =[20,30,45,100], labels = ['Young','Mid-Age','Senior'])

#founder experience categories
df['Founder_Experience_Category'] = pd.cut(df['Founder_Experience_Years'], bins =[0,2,7,100], labels = ['Beginner','Intermediate','Experienced'])

#team size categories
df['Team_Size_Category'] = pd.cut(df['Team_Size'], bins =[0,5,10,100], labels = ['Small','Medium','Large'])

#startup age months categories
df['Startup_Age_Months_Category'] = pd.cut(df['Startup_Age_Months'], bins =[0,12,36,120], labels = ['New','Growing','Established'])

#Investor pressure score categories
df['Investor_Pressure_Score_Category'] = pd.cut(df['Investor_Pressure_Score'], bins =[0,3,7,10], labels = ['Low','Moderate','High'])

#co founder conflict score categories
df['Cofounder_Conflict_Score_Category'] = pd.cut(df['Cofounder_Conflict_Score'], bins =[0,3,7,10], labels = ['Low','Moderate','High'])



#comparing using crosstab 

#Section:1 What leads to founder burnout?

#sleep hours category vs burnout level 
print("Sleep Schedules vs Burnout Levels")
print( 
    pd.crosstab(df['Sleep_Hours_Category'], df['Burnout_Level'], normalize='index') * 100) 

#weekly work hours category vs burnout level
print("Weekly Work Hours vs Burnout Levels")
print( 
    pd.crosstab(df['Weekly_Work_Hours_Category'], df['Burnout_Level'], normalize='index') * 100)    

#founder type category vs burnout level
print("Founder Type vs Burnout Levels")
print( 
    pd.crosstab(df['Founder_Type'], df['Burnout_Level'], normalize='index') * 100)  

#founder age category and experience category vs burnout level
print("Founder Age and Experience vs Burnout Levels")
print( 
    pd.crosstab([df['Founder_Age_Category'], df['Founder_Experience_Category']], df['Burnout_Level'], normalize='index') * 100)

#investor pressure category and co founder conflict category vs burnout level
print("Investor Pressure and Co-Founder Conflict vs Burnout Levels")
print( 
    pd.crosstab([df['Investor_Pressure_Score_Category'], df['Cofounder_Conflict_Score_Category']], df['Burnout_Level'], normalize='index') * 100)    


#section 2: What leads to startup failure?

#Do burned out founders fail more often?
print("Burnout Levels vs Startup Failure")
print( 
    pd.crosstab(df['Burnout_Level'], df['Startup_Failure_Flag'], normalize='index') * 100) 

#Does critical runway lead to higher failure rates? 
print("Runway Months Remaining vs Startup Failure")
print( 
    pd.crosstab(df['Runway_Months_Remaining_Category'], df['Startup_Failure_Flag'], normalize='index') * 100)  

#Does funding stage affecyt failure rates?
print("Funding Stage vs Startup Failure")
print( 
    pd.crosstab(df['Funding_Stage'], df['Startup_Failure_Flag'], normalize='index') * 100)

#Does inductry typr affect failure rates?
print("Industry Type vs Startup Failure")
print( 
    pd.crosstab(df['Industry'], df['Startup_Failure_Flag'], normalize='index') * 100) 

#Does economic climate affect failure rates?
print("Economic Climate vs Startup Failure")
print( 
    pd.crosstab(df['Economic_Climate'], df['Startup_Failure_Flag'], normalize='index') * 100)  

#Does team size affect failure rates?
print("Team Size vs Startup Failure")
print( 
    pd.crosstab(df['Team_Size_Category'], df['Startup_Failure_Flag'], normalize='index') * 100)    


#Numpy usage  


burnout_groups = df.groupby('Burnout_Level')['Burnout_Score'].mean()
print(burnout_groups)

stress_groups = df.groupby('Startup_Failure_Flag')['Stress_Score'].mean()
print(stress_groups)

decision_fatigue_groups = df.groupby('Startup_Failure_Flag')['Decision_Fatigue_Score'].mean()
print(decision_fatigue_groups)

product_market_fit_groups = df.groupby('Startup_Failure_Flag')['Product_Market_Fit_Score'].mean()
print(product_market_fit_groups)

monthly_revenue_growth_groups = df.groupby('Startup_Failure_Flag')['Monthly_Revenue_Growth_Percent'].mean()
print(monthly_revenue_growth_groups)

investor_pressure_groups = df.groupby('Startup_Failure_Flag')['Investor_Pressure_Score'].mean()
print(investor_pressure_groups)

cofounder_conflict_groups = df.groupby('Startup_Failure_Flag')['Cofounder_Conflict_Score'].mean()
print(cofounder_conflict_groups)

runway_months_remaining_groups = df.groupby('Startup_Failure_Flag')['Runway_Months_Remaining'].mean()
print(runway_months_remaining_groups)