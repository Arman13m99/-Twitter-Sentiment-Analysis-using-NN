# Twitter Sentiment Analysis using Neural Networks

## Overview
This project performs sentiment analysis on Twitter data using neural networks. The goal is to classify tweets into positive and negative sentiments based on their content. The project includes data preprocessing, exploratory data analysis (EDA), and the implementation of a neural network model to predict sentiment.

## Table of Contents
Overview
Dataset
Requirements
Project Structure
Exploratory Data Analysis
Modeling
Results
Usage
Contributing
License
## Dataset
The dataset used in this project is the Sentiment140 dataset, which contains 1,600,000 tweets labeled as either positive or negative. The columns in the dataset are:

label: Sentiment of the tweet (0 = negative, 4 = positive)
time: Time of the tweet
date: Date of the tweet
query: Query (unused)
username: Username of the person who tweeted
text: Text content of the tweet
## Requirements
The following Python libraries are required to run the notebook:

numpy
pandas
seaborn
matplotlib
scikit-learn
nltk
tensorflow
mlxtend
### You can install these libraries using pip:

pip install numpy pandas seaborn matplotlib scikit-learn nltk tensorflow mlxtend
## Project Structure
EDA-Twitter-Sentiment-Analysis-using-NN.ipynb: Jupyter notebook containing the entire analysis and modeling process.
training.1600000.processed.noemoticon.csv: CSV file containing the dataset (not included in the repository; download from the Sentiment140 website).
## Exploratory Data Analysis
The EDA involves:

Loading and visualizing the data
Cleaning and preprocessing the text data
Tokenizing and stemming the words
Visualizing the distribution of sentiments
## Modeling
The modeling process involves:

Splitting the data into training and testing sets
Tokenizing and padding the sequences
Building and training a neural network model using Keras
Evaluating the model using classification metrics
## Results
The model's performance is evaluated using:

Confusion matrix
Classification report (precision, recall, F1-score)
ROC curve and AUC
## Usage
To use this project, follow these steps:

### Clone the repository:
git clone https://github.com/your-username/twitter-sentiment-analysis-nn.git

### Navigate to the project directory:
cd twitter-sentiment-analysis-nn
### Install the required libraries:
pip install -r requirements.txt
### Run the Jupyter notebook:
jupyter notebook EDA-Twitter-Sentiment-Analysis-using-NN.ipynb
## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.
