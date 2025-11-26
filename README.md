Task 1 — Exploring & Visualizing the Iris Dataset
1. Objective

The main objective of this task is to practice data loading, inspection, and visualization.
The Iris dataset is used to understand variable relationships, detect outliers, and examine feature distributions.

2. Dataset Description

The dataset contains 150 samples and 5 columns:

Column	Description
sepal_length	Length of the sepal (cm)
sepal_width	Width of the sepal (cm)
petal_length	Length of the petal (cm)
petal_width	Width of the petal (cm)
species	Flower species (setosa, versicolor, virginica)

The dataset is clean with no missing values.

3. Exploratory Data Analysis

Key steps performed:

Loaded dataset using sns.load_dataset("iris")

Checked shape, columns, and datatypes

Displayed first few rows

Verified absence of missing values


4. Visualizations
✔ Scatter Plot – Sepal Length vs Petal Length

Shows a clear positive correlation

Species are well separated using color encoding

✔ Histogram – Sepal Length Distribution

Distribution appears close to normal

KDE curve helps identify density peaks

✔ Box Plot – Numeric Features

Useful for detecting outliers in sepal and petal measurements

Shows range and spread of each variable

5. Insights

Iris dataset is small, simple, and ideal for visualization practice

Petal measurements strongly differentiate species

No data cleaning required

Visualization reveals clear patterns between flower types


Task 2 — Credit Risk Prediction (Loan Default Classification)
1. Objective

The goal of this task is to predict whether a loan applicant will be approved or not.
This is a binary classification problem, using the Loan Prediction dataset from Kaggle.

2. Dataset Overview

The dataset includes 614 rows and 13 columns, containing:

Applicant demographic information

Income details

Loan amount and loan term

Credit history

Loan status (target variable)

3. Data Preprocessing

Steps performed:

✔ Handling Missing Values

Numeric columns → filled with median

Categorical columns → filled with mode

Forward fill applied where required

✔ Encoding Categorical Columns

Used LabelEncoder for:

Gender, Married, Dependents

Education, Self_Employed

Property_Area, Loan_Status

✔ Imputation Before Modeling

Used SimpleImputer(strategy="mean") to handle missing numeric values in features.

✔ Train/Test Split

train_test_split(test_size=0.2, random_state=42)

4. Visualization

Created meaningful charts:

Histogram for Loan Amount distribution

Box Plot: Education vs Loan Amount

Scatter Plot: Applicant Income vs Loan Amount

These highlight relationships between education, income, and loan demands.

5. Model Training

Model used: Logistic Regression

Accuracy: ~78.86%

6. Key Insights

Credit history plays a major role in loan approval

Applicant income influences loan amount

Missing values had to be carefully filled before training

Logistic Regression performed reasonably well without tuning

Task 3 – Customer Churn Prediction (ANN Model)
 Objective

The goal of this task is to predict whether a bank customer will churn (leave the bank) or stay, using demographic and financial information.
By identifying customers likely to exit, the bank can take proactive steps to improve retention and customer satisfaction.

 Approach
1. Data Loading & Exploration

Loaded the Churn_Modelling.csv dataset

Inspected shape, datatypes, duplicates

Verified no missing values

Identified the target column: Exited

2. Feature Preparation

Removed the target column from feature set

Performed an 80/20 train-test split

Applied StandardScaler to normalize the input features

Saved the scaler using pickle

3. ANN Model Architecture

A neural network was built using Keras Sequential API:

Input Layer: 8 features

Hidden Layer 1: 8 neurons (ReLU)

Hidden Layer 2: 6 neurons (ReLU)

Hidden Layer 3: 4 neurons (ReLU)

Output Layer: 1 neuron (Sigmoid)

Compiled the model with:

Optimizer: Adam

Loss: Binary Crossentropy

Metrics: Accuracy

4. Model Training

Trained for 50 epochs

Batch size: 100

Observed stable learning curve and convergence

5. Model Evaluation

Predictions made on scaled test data

Probabilities converted to class labels

Accuracy measured using accuracy_score

6. Saving the Model

Exported the trained ANN model as Churn_model.pkl

Saved scaler as scaler.pkl

Results & Insights
Model Performance
Dataset	Accuracy
Training Data	~85.9%
Testing Data	~85%

The ANN model demonstrates solid performance in identifying churn patterns.

 Key Insights About Customer Churn

Inactive members have a significantly higher chance of leaving the bank.

Older customers show slightly higher churn rates.

Customers with high account balance but low engagement are more likely to exit.

 Conclusion

This project successfully demonstrates the use of an Artificial Neural Network for churn prediction.
With ~85% accuracy, the model can help financial institutions identify at-risk customers and implement effective retention strategies.

Task 4 – Predicting Insurance Claim Amounts (Regression Model)
 Objective

The goal of this project is to predict medical insurance charges for individuals using their personal and lifestyle attributes.
This is a regression problem, and the model used is Linear Regression.

The predictions help insurance companies estimate risk and price premiums more accurately.

 Approach
1. Dataset Understanding

Dataset: Medical Cost Personal Dataset

2. Exploratory Data Analysis (EDA)

 Scatter Plot: BMI vs Charges

Higher BMI individuals tend to have higher medical charges.

 Scatter Plot: Age vs Charges

Charges increase with age.

 Box Plot: Smoker vs Charges

Smokers have drastically higher insurance costs.

Visualizations provided clear understanding of trends and feature impact.

3. Data Preprocessing

Separated features (X) and target (y)

Identified categorical columns: sex, smoker, region

Applied OneHotEncoder(drop='first') using ColumnTransformer

Performed train-test split (80/20)

4. Model Training

Model used: Linear Regression


Model Performance Metrics
Metric	Value

MAE (Mean Absolute Error)	~4181.19
RMSE (Root Mean Squared Error)	~5796.28
Interpretation:

On average, predictions differ by about ±4000 USD from actual charges.

RMSE indicates presence of some high-error cases due to extreme values.


 Key Insights

Smoking has the strongest impact on insurance charges — smokers cost significantly more.

Higher BMI increases medical costs due to obesity-related risks.

Age also contributes positively — older individuals tend to have higher expenses.

 Conclusion

The Linear Regression model provides a reasonable baseline for predicting insurance charges with an MAE of ~4000 and RMSE of ~5800.