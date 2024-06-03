Bank Marketing Decision Tree Classifier
This repository contains a project that builds a decision tree classifier to predict whether a customer will subscribe to a term deposit based on their demographic and behavioral data. The dataset used is the Bank Marketing dataset from the UCI Machine Learning Repository.

Dataset
The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit.

Files
bank.csv: A small sample of the data with 4521 examples.

bank-full.csv: The full dataset with 45211 examples.

bank-names.txt: Contains metadata about the dataset and its attributes.

Installation
Clone the repository:

Create a virtual environment and activate it:

Install the required packages:

pip install -r requirements.txt


Usage
Place the dataset files (bank.csv, bank-full.csv, bank-names.txt) in the data directory.

Run the decision tree classifier script:

Model Training and Evaluation

The script decision_tree_classifier.py performs the following steps:

Loads the dataset.

Preprocesses the data, including converting categorical variables to dummy variables.

Splits the data into training and testing sets.

Trains a decision tree classifier.

Evaluates the model using metrics like accuracy, precision, recall, and F1-score.
