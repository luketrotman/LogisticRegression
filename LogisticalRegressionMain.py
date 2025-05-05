#1. Import the neccessary librarires 
import pandas as pd # for data manipulaton
import numpy as np # for numerical operations
import matplotlib.pyplot as plt # for stats plotting 
import seaborn as sns # for heatmapping and nicer data visualisation
from sklearn.model_selection import train_test_split # Split the data
from sklearn.linear_model import LogisticRegression # The classification model I am using
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay # for evaluatiion 
# Description of above metrics 
# Classication_report: shows precision, recall, F1, and accuracy - overall model performance 
# confusion_matric: shows correct/incorrect predicitons - helps identify errors
# Roc_auc_score: shows how well the model speerates the classes (the closer to 1 the better)
# Roc_curve: plots true v false positive rate at all thresholds - visual model performance

#2. Load the dataset 
df = pd.read_csv("/Users/luketrotman/Downloads/hmda_sample.csv")  # HMDA mortgage dataset

#3. quick look at the data to understand the structure
df.head()

#4 drop the missing values
print(df.columns)  # Shows all column names so we can match them exactly

# Clean 'debt_to_income_ratio' by removing '%' and converting to float
df['debt_to_income_ratio'] = df['debt_to_income_ratio'].str.replace('%', '', regex=False)
df['debt_to_income_ratio'] = df['debt_to_income_ratio'].str.extract('(\d+)', expand=False).astype(float)

df = df.dropna() #clean the data and remove any rows with NaNs

#5 select features and target columsn and justify why
X = df[['loan_amount', 'income', 'debt_to_income_ratio']]  

# Justifications:
# loan_amount – higher loan values typically face stricter approval, increasing risk.
# applicant_income – higher income suggests better ability to repay the loan.
# debt_to_income_ratio – measures how much of income is already used; high ratios = higher risk.

df['preapproval'] = df['preapproval'].map({1: 1, 2: 0})  # 1 = preapproved, 0 = not

y = df['preapproval']  # Target variable: 1 if approved, 0 if rejected

#6. seperate data into training data (80%) and test data (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)  # 80% train, 20% test; seed for reproducibility

#7. create and train the logistic regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced') # chosen initial model, # handles class imbalance
model.fit(X_train, y_train) # train the model on data

# 8. Make predictions on test data
y_pred = model.predict(X_test)            # Binary predictions (0 loan denial  or 1 loan approval)

#9 evaluate the model 
print(confusion_matrix(y_test, y_pred))          # Shows TP, TN, FP, FN
print(classification_report(y_test, y_pred))     # Shows precision, recall, F1, accuracy

# 10. Plot ROC Curve
RocCurveDisplay.from_estimator(model, X_test, y_test)  # Plots true vs false positive rate
plt.title("ROC Curve")
plt.show()

# 11. Get and print AUC score
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])  # Probability scores
print("ROC AUC Score:", auc)                                   # Higher = better