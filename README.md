# Startup Founder Burnout & Failure — EDA

An exploratory data analysis project on a dataset of 50,000 startup founders, investigating what leads to founder burnout and ultimately startup failure.

## Dataset
- Source: Startup Founder Burnout 2026 (Kaggle)
- Size: 50,000 rows, 24 columns (after cleaning)
- Target Variables: `Burnout_Level`, `Startup_Failure_Flag`

## Project Structure
Startup-Burnout-EDA/
│
├── startup_analysis.py       # Main analysis script
├── startup_founder_burnout_2026.csv  # Dataset
└── README.md


## Key Questions Answered

### Section 1 — What leads to founder burnout?
1. Does the founder's sleep schedule affect their burnout level?
2. Does the founder's weekly work hours affect their burnout level?
3. Does founder type affect their burnout level?
4. Does founder age and experience level affect their burnout level?
5. Do investor pressure and cofounder conflict drive founder burnout?

### Section 2 — What leads to startup failure?
1. Do burned out founders fail more?
2. Does critical runway lead to higher failure rate?
3. Does funding stage affect failure rate?
4. Does industry type affect failure rate?
5. Does economic climate affect failure rate?
6. Do high stress and decision fatigue scores correlate with failure?
7. Does team size affect failure rate?

## Key Findings

### Burnout
- Severe burnout founders score **5.98 points higher** on burnout score than low burnout founders
- Founders with poor sleep have a **26.4% severe burnout rate** vs only **0.6%** for good sleepers
- Extreme work hours (60-100hrs/week) lead to **16% severe burnout** vs **0.3%** for normal hours
- When both investor pressure and cofounder conflict are high, **44.7% of founders reach severe burnout**

### Startup Failure
- Severely burned out founders have a **69% failure rate** vs only **7.9%** for low burnout founders
- Failed founders had **2.5 points higher stress** and **2.67 points higher decision fatigue**
- Surviving startups had **2.19 higher PMF score** — they built what the market wanted
- Surviving startups grew at **3.41% higher monthly revenue**
- Surviving startups had **5.7 more months of runway** on average
- Startups in a **Recession had a 45.7% failure rate** vs only 13.9% in a Bull Market
- Industry type showed **no significant impact** on failure rate


## Technologies Used
- Python 3
- pandas
- numpy
- matplotlib
- seaborn


## How to Run
1. Clone the repo
2. Install dependencies: `pip install pandas numpy matplotlib seaborn`
3. Run: `python startup_analysis.py`