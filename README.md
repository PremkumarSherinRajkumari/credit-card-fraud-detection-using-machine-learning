## credit-card-fraud-detection-using-machine-learning

This Credit Card Fraud Detection project is a robust and efficient framework designed to identify fraudulent transactions in credit card datasets. It includes preprocessing tools, visualization utilities, and advanced machine learning models to improve fraud detection capabilities. By analyzing historical data, it calculates fraud likelihood, enabling proactive measures against fraudulent activities.

## Project Overview:
Credit card fraud detection is a binary classification problem where the goal is to distinguish between fraudulent (is_fraud=1) and legitimate (is_fraud=0) transactions. This project includes:

- Data preprocessing and feature engineering
- Handling missing values
- Exploratory data analysis (EDA)
- Machine learning model training and evaluation
- Model performance improvement using resampling techniques

## Dataset:

The dataset consists of two parts, which are merged into a single dataset:

Data columns include:
1. Transaction details: trans_date_trans_time, amt, merchant
2. User details: cc_num, first, last, dob, gender, city, state
3. Fraud indicator: is_fraud
4. Geographical and transactional features: lat, long, merch_lat, merch_long

Preprocessing steps include handling missing values, transforming categorical variables, and applying log and power transformations.

## Libraries Used:

Python, Pandas, NumPy, Matplotlib & Seaborn, Scikit-learn

## Key Techniques:

Data Preprocessing: Cleaning, feature scaling, and handling outliers
Visualization: Heatmaps, bar plots, box plots, and histograms for insights
Resampling Techniques: SMOTE - Oversampling for balanced class distribution and ADASYN: Adaptive synthetic sampling

## Models:

Logistic Regression

Random Forest

K-Means clustering with Random Forest

## Results:

Logistic Regression: Accuracy = 89.10%, F1 Score = 0.22

Random Forest: Accuracy = 97.69%, F1 Score = 0.22

K-Means Clustering with Random Forest: Accuracy = 97.57%, F1 Score = 0.22

## Key Findings:

Imbalanced Dataset: Fraud cases are a small fraction of the dataset, requiring resampling.

SMOTE vs. ADASYN:

Both improved recall for fraud detection.

ADASYN provided slightly better results in some scenarios.

## Visualization:

Certain transaction categories and time periods had higher fraud likelihood.

High fraud rates in low-amount transactions.

## Model Limitations:

Precision for fraud detection remains low due to high false positives.

## Getting Started:
1. Clone the repository: git clone https://github.com/PremkumarSherinRajkumari/credit-card-fraud-detection-using-machine-learning.git
2. Navigate to the project directory: cd Credit Card Fraud Detection.ipynb
3. Run the script to preprocess data and train models. 

## Future Enhancements:
- Integration of Real-Time Monitoring:
Enhancing the model to process streaming data and detect fraud in real-time, enabling proactive interventions.
- Incorporation of Explainable AI (XAI):
Developing interpretable models to provide clear insights into why a transaction is classified as fraudulent, which can boost trust and regulatory compliance.
- Advanced Ensemble Techniques:
Combining multiple models, including neural networks, Random Forests, and others, to create a robust ensemble system for higher accuracy and reliability in fraud detection.

## Additional Information:
Feel free to reach out to me at sherinrajkumariis@gmail.com for any inquiries or collaborations.
