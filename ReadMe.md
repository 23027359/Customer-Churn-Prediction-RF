Task 1: Customer Churn Prediction using Random Forest
📋 Project Overview
This repository contains the implementation for Level 3 Task 1 of my Machine Learning internship. The goal of this project is to develop a predictive model that identifies customers likely to cancel their service (churn) based on their historical usage patterns and account details.

🗂️ Dataset Description
The model utilizes a "Customer Churn" dataset split into two files for training and testing:

Training Set: churn-bigml-80.csv (80% of the data).

Test Set: churn-bigml-20.csv (20% of the data).

The dataset includes 17 key features such as:

Account Information: Account length, International plan, Voice mail plan.

Usage Metrics: Total day/evening/night/international minutes, calls, and charges.

Service Interaction: Number of customer service calls.

🛠️ Implementation Details
1. Data Preprocessing
Categorical Encoding: Features like 'International plan' and 'Voice mail plan' were converted into binary integers (1 for 'Yes', 0 for 'No').

Feature Scaling: Used StandardScaler to normalize numerical data, ensuring that features with large ranges (like minutes) do not disproportionately influence the model.

2. Model Architecture
I implemented a Random Forest Classifier with the following optimized parameters:

n_estimators: 200 trees for robust ensemble learning.

max_depth: 12 to capture complex interactions without overfitting.

class_weight: 'balanced' to account for the minority churn class.

3. Evaluation Metrics
The model was evaluated on the unseen test set using several performance indicators:

Accuracy & Precision

Recall & F1-Score

ROC-AUC Score (to measure class separation capability)

🚀 Key Insights
Based on the Feature Importance analysis, the following factors were identified as the strongest predictors of customer churn:

Total day charge/minutes

Customer service calls

International plan status

💻 How to Run
Ensure you have the following libraries installed:

Bash

pip install pandas numpy matplotlib seaborn scikit-learn
Note: Update the folder variable in the script to point to your local dataset directory (e.g., C:\Users\...\Churn Prediction Data).

💡 Fixing the Code Error
Based on your uploaded screenshot (image_c8dfed.png), you are receiving an IncompleteInputError because the model definition in your Task 3 (Neural Networks) script is missing its closing brackets.

Correction: Ensure the model section ends exactly like this:

Python

model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')
]) # <--- Ensure these closing brackets are present
