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


# Burnout Score across Burnout Levels
# extracting the mean burnout score for Low and Severe groups from the groupby result
low = burnout_groups['Low']
severe = burnout_groups['Severe']

# subtracting Low mean from Severe mean to get the exact point difference
print(f"Severe burnout founders score {np.round(severe - low, 2)} points higher than Low burnout founders")

# Failed vs Non-Failed
# in Startup_Failure_Flag: 0 = survived, 1 = failed
# groupby result is indexed by 0 and 1, so we access them directly

# stress_groups[1] = mean stress of failed founders
# stress_groups[0] = mean stress of survived founders
# difference tells us how much more stressed failed founders were
print(f"Failed founders have {np.round(stress_groups[1] - stress_groups[0], 2)} higher Stress Score")

# same pattern — failed founders had higher decision fatigue
print(f"Failed founders have {np.round(decision_fatigue_groups[1] - decision_fatigue_groups[0], 2)} higher Decision Fatigue")

# PMF is reversed — higher is better, so survived (0) > failed (1)
# we subtract failed from survived to show how much better survivors scored
print(f"Failed startups have {np.round(product_market_fit_groups[0] - product_market_fit_groups[1], 2)} higher PMF Score when they survive")

# same logic — revenue growth is better for survivors, so 0 - 1
print(f"Failed startups grow {np.round(monthly_revenue_growth_groups[0] - monthly_revenue_growth_groups[1], 2)}% less in revenue")

# failed founders faced more investor pressure
print(f"Failed founders have {np.round(investor_pressure_groups[1] - investor_pressure_groups[0], 2)} higher Investor Pressure")

# failed founders had more cofounder conflict
print(f"Failed founders have {np.round(cofounder_conflict_groups[1] - cofounder_conflict_groups[0], 2)} higher Cofounder Conflict")

# runway is better for survivors — they had more months remaining
# so survived (0) - failed (1) shows the advantage survivors had
print(f"Surviving startups had {np.round(runway_months_remaining_groups[0] - runway_months_remaining_groups[1], 2)} more months of runway than failed startups")

#visualization

#GRAPH 1 : HEATMAP
# Correlation Heatmap - all numeric columns together
numeric_cols = ['Burnout_Score', 'Stress_Score', 'Decision_Fatigue_Score','Investor_Pressure_Score','Cofounder_Conflict_Score','Runway_Months_Remaining','Monthly_Revenue_Growth_Percent','Product_Market_Fit_Score','Startup_Failure_Flag']
corr_matrix = df[numeric_cols].corr()
print(corr_matrix)

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Key Metrics')
plt.tight_layout()
plt.show()

#Graph 2: HEATMAP
#Investor pressure and cofounder conflict vs burnout levels

conflict_pressure_burnout = pd.crosstab(
    [df['Investor_Pressure_Score_Category'], df['Cofounder_Conflict_Score_Category']], 
    df['Burnout_Level'], 
    normalize='index'
)*100 

sns.heatmap(conflict_pressure_burnout, annot =True, cmap = 'YlGnBu', linewidths =0.5)
plt.title('Investor Pressure and Co-Founder Conflict vs Burnout Levels')
plt.tight_layout()
plt.ylabel('Investor Pressure - Cofounder Conflict')

plt.show()

#GRAPH 3 : BOX PLOT 
# distribution of Burnout score across low, moderate, severre burnout levels

plt.figure(figsize=(8,6))
sns.boxplot (x = 'Burnout_Level', y = 'Burnout_Score', data =df, hue ='Burnout_Level' , palette = 'coolwarm',order = ['Low', 'Moderate', 'Severe'], legend = False)
plt.title('Burnout score distribution across Burnout Levels')
plt.xlabel('Burnout Level')
plt.ylabel('Burnout Score')
plt.tight_layout()
plt.show()

# GRAPH 4: Box plot - Stress Score + Decision Fatigue vs Startup Failure
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.boxplot(x='Startup_Failure_Flag', y='Stress_Score', data=df, hue='Startup_Failure_Flag', palette='coolwarm', ax=axes[0], legend=False)
axes[0].set_title('Stress Score: Failed vs Survived')
axes[0].set_xlabel('Startup Failure (0=Survived, 1=Failed)')
axes[0].set_ylabel('Stress Score')

sns.boxplot(x='Startup_Failure_Flag', y='Decision_Fatigue_Score', data=df, hue='Startup_Failure_Flag', palette='coolwarm', ax=axes[1], legend=False)
axes[1].set_title('Decision Fatigue: Failed vs Survived')
axes[1].set_xlabel('Startup Failure (0=Survived, 1=Failed)')
axes[1].set_ylabel('Decision Fatigue Score')

plt.tight_layout()
plt.show()

# GRAPH 5: Scatter plot - PMF Score vs Monthly Revenue Growth colored by failure
plt.figure(figsize=(10, 6))
colors = {0: 'steelblue', 1: 'tomato'}
for flag, group in df.groupby('Startup_Failure_Flag'):
    plt.scatter(group['Product_Market_Fit_Score'],group['Monthly_Revenue_Growth_Percent'], c=colors[flag], label='Survived' if flag == 0 else 'Failed', alpha=0.3, s=10)
plt.title('Product Market Fit vs Revenue Growth by Failure Status')
plt.xlabel('Product Market Fit Score')
plt.ylabel('Monthly Revenue Growth %')
plt.legend()
plt.tight_layout()
plt.show()

# GRAPH 6: Stacked bar - Economic Climate vs Failure Rate
econ_failure = pd.crosstab(df['Economic_Climate'], df['Startup_Failure_Flag'], normalize='index') * 100
econ_failure.columns = ['Survived', 'Failed']
econ_failure.plot(kind='bar', stacked=True, figsize=(10, 6), color=['steelblue', 'tomato'])
plt.title('Startup Failure Rate by Economic Climate')
plt.xlabel('Economic Climate')
plt.ylabel('Percentage %')
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# GRAPH 7: Bar chart - Mean differences summary
metrics = ['Stress Score', 'Decision Fatigue', 'PMF Score', 'Revenue Growth %','Investor Pressure', 'Cofounder Conflict', 'Runway Months']
failed_means = [stress_groups[1], decision_fatigue_groups[1], product_market_fit_groups[1],monthly_revenue_growth_groups[1], investor_pressure_groups[1], cofounder_conflict_groups[1], runway_months_remaining_groups[1]]
survived_means = [stress_groups[0], decision_fatigue_groups[0], product_market_fit_groups[0], monthly_revenue_growth_groups[0], investor_pressure_groups[0], cofounder_conflict_groups[0], runway_months_remaining_groups[0]]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(x - width/2, survived_means, width, label='Survived', color='steelblue')
ax.bar(x + width/2, failed_means, width, label='Failed', color='tomato')
ax.set_title('Mean Metric Comparison: Failed vs Survived Startups')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=30, ha='right')
ax.set_ylabel('Mean Value')
ax.legend()
plt.tight_layout()
plt.show()