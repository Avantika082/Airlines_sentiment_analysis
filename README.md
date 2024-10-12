This project is created using two different models one is machine learning model using logistic regression and another one deep learning model to predict the sentiment of the tweet of Airlines sentiment Dataset whether it is positive or negative , used deep learning model uses WordToVec word embeddings and LSTM deep nueral network to train the model and predict the sentiment of new tweet.


### **Table of Contents**

- Project Overview
- Technologies Used
- Features
- Model Explanation
- Project Structure
- Contributing




### **Introduction**

Airlines frequently receive feedback from their customers via online platforms. Analyzing this feedback can help airlines understand customer sentiments and improve their services. In this project, we use two different approaches to classify airline customer reviews:

- Logistic Regression: A basic machine learning model.
- LSTM (Long Short-Term Memory): A deep learning model particularly effective for text data.


### **Dataset**


The dataset used for this project is a collection of airline customer reviews. Each review is labeled as one of the following sentiments:

- Positive
- Neutral
- Negative



### **Dataset Summary**


The Airlines sentiment analysis model uses only below columns to predict the sentiment for both the models , logistic regression model and deep learning model using LSTM.

- Text
- Target



Both the models are in the models folder which consists of jupyter notebooks of their respective models and their saved models . 




### **Technologies Used**


- Python: Programming language for model development.
- Pandas & Numpy: For data manipulation and processing.
- scikit-learn: For building the machine learning model.
- Tensorflow : Python library used for creating deep learning models.
- Keras: A wrapper used for tensorflow to create deep learning models easily and efficiently.
- nltk: For text preprocessing.




### **Features**


- Machine Learning Model: Logistic regression trained on the Airlines dataset.
- Deep learning Model : Deep learning model using LSTM.
- Prediction: Provides a binary prediction (positive or negative for diabetes).




### **Setup Instructions**

Install the required libraries using the following command:

``` pip install numpy pandas scikit-learn tensorflow nltk matplotlib ```


Installation


1. Clone this repository:

``` git clone https://github.com/Avantika082/Airlines_sentiment_analysis.git```
``` cd airlines-sentiment-analysis ```


2. Install the required libraries:

```pip install -r requirements.txt```



### **Models** 


1. Logistic Regression
A standard machine learning model that works well for linearly separable data. We preprocess the text data using TF-IDF (Term Frequency-Inverse Document Frequency) and train the Logistic Regression model on the transformed data.

2. LSTM (Long Short-Term Memory)
LSTM is a type of Recurrent Neural Network (RNN) designed to handle sequence prediction problems. This model is trained on the tokenized reviews using an embedding layer to capture the meaning of words in context.





### **Project Structure**


diabetes-prediction-model/

- ├── Airline-Sentiment_data&nbsp;&nbsp;&nbsp;&nbsp;       # Arlines Sentiment data csv file 
- ├── requirements.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;       # requirments file
- ├── models/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          # Models used in the project
- │   └── logistic regression&nbsp;&nbsp;&nbsp;&nbsp;      # Logistic regression trained model folder
- |       └──logistic_model_notebook.ipynb&nbsp;&nbsp;&nbsp; # Logistic regression jupyter notebook
- |       └──logistic_model.pkl&nbsp;&nbsp;&nbsp;&nbsp;     # Pickle file for logistic regression model
- |   └── LSTM&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;         # LSTM trained model folder
- |       └──lstm_model_notebook.ipynb&nbsp;&nbsp;&nbsp;    # LSTM jupyter notebook
- |       └──lstm_model.h5&nbsp;&nbsp;&nbsp;&nbsp;          # Saved LSTM Model
- |       └──word2vec.lstm_model&nbsp;&nbsp;&nbsp;          # Saved wordtovec model   
- └── README.md&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;              # Project documentation (this file)




### **Contributing**


Feel free to open issues or pull requests if you have suggestions or improvements. Please ensure your contributions adhere to the project's coding guidelines.