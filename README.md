
# MentorMatch Intelligence Dashboard

A dark-themed, interactive Streamlit dashboard for analysing MentorMatch UAE user data, career outcomes, subscription conversion, mentor matching, clustering, predictive models, and association rules.

## Project Title

**MentorMatch: An Analytics-Driven Dashboard for Career Outcomes and Subscription Conversion in the UAE**

## Business Context

MentorMatch supports early-career professionals in the UAE by connecting them with mentors who can guide them through career confusion, skill gaps, interview preparation, career switching, salary growth, workplace confidence, and professional networking.

This dashboard is designed for the Head Data Analyst at MentorMatch to understand:

- Which user groups achieve better career outcomes
- Which behaviours drive subscription conversion
- How mentor matching and app engagement influence success
- Which segments need targeted interventions
- Which services and user behaviours commonly appear together

## Files Included

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit dashboard application |
| `requirements.txt` | Python dependencies for Streamlit Cloud |
| `README.md` | Project instructions and summary |
| `MentorMatch_UAE_Clean.csv` | Cleaned dataset used by the dashboard |

## Dashboard Sections

1. **Executive Story**
   - KPI cards
   - Subscription tier vs career outcome
   - Engagement vs career ROI story

2. **Data Cleaning**
   - Duplicate removal
   - Missing value handling
   - Feature engineering
   - Data health summary

3. **Descriptive Analytics**
   - Cross-tabulations against career outcomes
   - Outcome distribution
   - Goal and challenge analysis
   - Correlation heatmap across multiple features

4. **Diagnostic Analytics**
   - Deep-dive into age, experience, goals, mentor match, subscription tier, and user challenges
   - UAE early-career professional context
   - Conversion and mentor-mentee experience insights

5. **Predictive Analytics**
   - Classification models:
     - KNN
     - Decision Tree
     - Random Forest
     - Gradient Boosted Classifier
   - Regression models:
     - Linear Regression
     - Ridge Regression
     - Lasso Regression
     - Elastic Net
     - Decision Tree Regressor
     - Random Forest Regressor
     - Extra Trees Regressor
     - Gradient Boosting Regressor
     - AdaBoost Regressor
   - Metrics:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - ROC-AUC
     - Confusion matrix
     - R²
     - MAE
     - RMSE
   - Interactive lambda/alpha slider for Ridge, Lasso, and Elastic Net

6. **Clustering**
   - K-Means clustering
   - Elbow method from K=2 to K=15
   - Silhouette analysis
   - Cluster distance analysis
   - Interactive 3D PCA visualization

7. **Association Rule Mining**
   - Support
   - Confidence
   - Lift
   - Leverage
   - Conviction
   - Interactive rule scatter plot

8. **Findings**
   - Prescriptive recommendations
   - MentorMatch growth playbook
   - Final interactive summary

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to Deploy on Streamlit Cloud

1. Create a GitHub repository.
2. Upload these four files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `MentorMatch_UAE_Clean.csv`
3. Go to Streamlit Cloud.
4. Select your GitHub repository.
5. Set the main file path as:

```text
app.py
```

6. Click **Deploy**.

## Recommended Report Interpretation

This dashboard should be used to explain how MentorMatch can become more than a mentor-booking app. The strongest business positioning is:

> MentorMatch is a career progress platform that uses data to improve mentor matching, subscription conversion, and measurable career outcomes for early-career professionals in the UAE.

## Notes

- The dashboard includes leakage-safe mode for predictive analysis.
- You can upload a new CSV from the dashboard sidebar if the dataset is updated.
- The default dataset must remain in the same folder as `app.py` when deploying.
