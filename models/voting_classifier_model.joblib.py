import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Data Collection - Load the dataset
sepsis_data = pd.read_csv('sepsisdata.csv')

# Step 2: Data Preprocessing - Impute missing values
imputer = SimpleImputer(strategy='mean')
sepsis_data_imputed = pd.DataFrame(imputer.fit_transform(sepsis_data), columns=sepsis_data.columns)

# Step 3: Feature Selection - Select relevant features
selected_features = ['HR_mean', 'O2Sat_mean', 'Temp_mean', 'MAP_mean', 'Resp_mean',
                     'FiO2_mean', 'SaO2_mean', 'AST_mean', 'BUN_mean', 'Creatinine_mean',
                     'Glucose_mean', 'Hgb_mean', 'WBC_mean', 'Platelets_mean', 'Age_mean', 'Gender_first',
                     'HospAdmTime_first', 'ICULOS_max']

X = sepsis_data_imputed[selected_features]
y = sepsis_data_imputed['SepsisLabel_max']

# Step 4: Data Splitting - Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Data Standardization - Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Model Training - Train base models
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
logistic_regression_model = LogisticRegression()
naive_bayes_model = GaussianNB()

# SVM model with RBF kernel
svm_model = SVC(kernel='rbf', probability=True, random_state=42)

# Step 7: Ensemble - Create a Voting Classifier
voting_classifier = VotingClassifier(estimators=[
    ('rf', random_forest_model),
    ('lr', logistic_regression_model),
    ('nb', naive_bayes_model),
    ('svm', svm_model)
], voting='soft')

# Train the Voting Classifier
voting_classifier.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(voting_classifier, 'voting_classifier_model.joblib')

# Step 8: Model Evaluation - Evaluate the Voting Classifier
y_pred = voting_classifier.predict(X_test_scaled)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

