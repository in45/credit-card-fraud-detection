# Credit Card Fraud Detection Using Deep Learning Models
## Main Objective of the Analysis
The main objective of this analysis is to develop a deep learning model to detect fraudulent credit card transactions. Given the highly imbalanced nature of the dataset, where fraudulent transactions represent only 0.172% of the total, the challenge lies in accurately identifying fraudulent activities without excessively misclassifying legitimate transactions. This project will utilize various deep learning models, specifically Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM) networks, and AutoEncoders, to determine the most effective approach. The ultimate goal is to provide a robust and accurate model that can assist financial institutions in mitigating the risk of fraud.

## Description of the Dataset

The dataset used for this analysis is sourced from Kaggle and contains credit card transactions made by European cardholders in September 2013. The dataset includes transactions over two days, comprising 492 fraudulent transactions out of 284,807 total transactions. The dataset is highly imbalanced, with fraudulent transactions accounting for only 0.172% of the total.

### Dataset Attributes:
* Time: The seconds elapsed between each transaction and the first transaction in the dataset.
* V1, V2, ..., V28: Principal components obtained via PCA transformation.
* Amount: The transaction amount.
* Class: The target variable, where 1 indicates fraud and 0 indicates a legitimate transaction.

Due to confidentiality, the original features are not provided, and only transformed PCA components are available for analysis.

- Kaggle dataset : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
## Data Exploration and Preprocessing
### Data Exploration
* The dataset comprises 284,807 transactions with 492 labeled as fraudulent.
* The distribution of the target variable (Class) shows a significant imbalance.
* Features V1 to V28 are already scaled due to PCA transformation.
### Data Cleaning and Feature Engineering
* Checked for missing values and found none.
* Standardized the 'Amount' feature to have zero mean and unit variance.
* Split the dataset into training and testing sets using a stratified sampling approach to maintain the class distribution.
* 
## Model Training
### Model Variations
#### -> Convolutional Neural Network (CNN)

* Utilized 1D convolutional layers to capture temporal patterns in the transaction sequences.
* Achieved reasonable accuracy but struggled with the imbalanced nature of the data.
#### -> Recurrent Neural Network (RNN)

* Implemented a simple RNN to model the sequence of transactions.
* Showed moderate performance but was prone to overfitting.
#### -> Long Short-Term Memory (LSTM)

* Leveraged LSTM networks to capture long-term dependencies in transaction sequences.
* Performed better than RNNs in handling the temporal aspect of the data.
#### -> AutoEncoders

* Trained an AutoEncoder to learn the normal patterns of transactions and used reconstruction error to detect anomalies.
* Provided the best performance in terms of detecting fraudulent transactions.

### Model Evaluation
* Evaluated models using Precision, Recall, F1-Score, and Area Under the ROC Curve (AUC).
* AutoEncoders demonstrated the highest AUC and balanced Precision and Recall, making them the preferred model for this analysis.
## Recommended Model
Based on the evaluation metrics, the AutoEncoder model is recommended as the final model. It effectively balances the trade-off between detecting fraudulent transactions and minimizing false positives, which is crucial for a highly imbalanced dataset like this.

## Key Findings and Insights
* The dataset's high imbalance poses significant challenges in fraud detection.
* AutoEncoders, which focus on learning normal transaction patterns, are well-suited for identifying anomalies indicative of fraud.
* While CNNs and LSTMs showed promise, they were less effective in handling the imbalance without extensive tuning and additional data preprocessing.
