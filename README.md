# Introduction-to-random-forest

## Part 1: Using Random Forest for Regression

In this section we will study how random forests can be used to solve regression problems using Scikit-Learn. In the next section we will solve classification problem via random forests.

Problem Definition
The problem here is to predict the gas consumption (in millions of gallons) in 48 of the US states based on petrol tax (in cents), per capita income (dollars), paved highways (in miles) and the proportion of population with the driving license.

Solution
To solve this regression problem we will use the random forest algorithm via the Scikit-Learn Python library. We will follow the traditional machine learning pipeline to solve this problem. Follow these steps:

### 1. Import Libraries
Execute the following code to import the necessary libraries:

import pandas as pd
import numpy as np

### 2. Importing Dataset
The dataset for this problem is available at:

https://drive.google.com/file/d/1mVmGNx6cbfvRHC_DvF12ZL3wGLSHD9f_/view

For the sake of this tutorial, the dataset has been downloaded into the "Datasets" folder of the "D" Drive. You'll need to change the file path according to your own setup.

Execute the following command to import the dataset:

dataset = pd.read_csv('D:\Datasets\petrol_consumption.csv')

To get a high-level view of what the dataset looks like, execute the following command:

dataset.head()

We can see that the values in our dataset are not very well scaled. We will scale them down before training the algorithm.

### 3. Preparing Data For Training
Two tasks will be performed in this section. The first task is to divide data into 'attributes' and 'label' sets. The resultant data is then divided into training and test sets.

The following script divides data into attributes and labels:

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
Finally, let's divide the data into training and testing sets:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### 4. Feature Scaling
We know our dataset is not yet a scaled value, for instance the Average_Income field has values in the range of thousands while Petrol_tax has values in range of tens. Therefore, it would be beneficial to scale our data (although, as mentioned earlier, this step isn't as important for the random forests algorithm). To do so, we will use Scikit-Learn's StandardScaler class. Execute the following code to do so:


### Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### 5. Training the Algorithm
Now that we have scaled our dataset, it is time to train our random forest algorithm to solve this regression problem. Execute the following code:

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
The RandomForestRegressor class of the sklearn.ensemble library is used to solve regression problems via random forest. The most important parameter of the RandomForestRegressor class is the n_estimators parameter. This parameter defines the number of trees in the random forest. We will start with n_estimator=20 to see how our algorithm performs. You can find details for all of the parameters of RandomForestRegressor here.

### 6. Evaluating the Algorithm
The last and final step of solving a machine learning problem is to evaluate the performance of the algorithm. For regression problems the metrics used to evaluate an algorithm are mean absolute error, mean squared error, and root mean squared error. Execute the following code to find these values:

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
The output will look something like this:

Mean Absolute Error: 51.765
Mean Squared Error: 4216.16675
Root Mean Squared Error: 64.932016371
With 20 trees, the root mean squared error is 64.93 which is greater than 10 percent of the average petrol consumption i.e. 576.77. This may indicate, among other things, that we have not used enough estimators (trees).

If the number of estimators is changed to 200, the results are as follows:

Mean Absolute Error: 47.9825
Mean Squared Error: 3469.7007375
Root Mean Squared Error: 58.9041657058
The following chart shows the decrease in the value of the root mean squared error (RMSE) with respect to number of estimators. Here the X-axis contains the number of estimators while the Y-axis contains the value for root mean squared error.

https://s3.amazonaws.com/stackabuse/media/random-forest-algorithm-python-scikit-learn-1.png

You can see that the error values decreases with the increase in number of estimator. After 200 the rate of decrease in error diminishes, so therefore 200 is a good number for n_estimators. You can play around with the number of trees and other parameters to see if you can get better results on your own.

## Part 2: Using Random Forest for Classification
Problem Definition
The task here is to predict whether a bank currency note is authentic or not based on four attributes i.e. variance of the image wavelet transformed image, skewness, entropy, and curtosis of the image.

Solution
This is a binary classification problem and we will use a random forest classifier to solve this problem. Steps followed to solve this problem will be similar to the steps performed for regression.

### 1. Import Libraries
import pandas as pd
import numpy as np

### 2. Importing Dataset
The dataset can be downloaded from the following link:

https://drive.google.com/file/d/13nw-uRXPY8XIZQxKRNZ3yYlho-CYm_Qt/view

The detailed information about the data is available at the following link:

https://archive.ics.uci.edu/ml/datasets/banknote+authentication

The following code imports the dataset:

dataset = pd.read_csv("D:/Datasets/bill_authentication.csv")

To get a high level view of the dataset, execute the following command:

dataset.head()

As was the case with regression dataset, values in this dataset are not very well scaled. The dataset will be scaled before training the algorithm.

### 3. Preparing Data For Training
The following code divides data into attributes and labels:

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
The following code divides data into training and testing sets:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
4. Feature Scaling
As with before, feature scaling works the same way:

### Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### 5. Training the Algorithm
And again, now that we have scaled our dataset, we can train our random forests to solve this classification problem. To do so, execute the following code:

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
In case of regression we used the RandomForestRegressor class of the sklearn.ensemble library. For classification, we will RandomForestClassifier class of the sklearn.ensemble library. RandomForestClassifier class also takes n_estimators as a parameter. Like before, this parameter defines the number of trees in our random forest. We will start with 20 trees again. You can find details for all of the parameters of RandomForestClassifier here.

### 6. Evaluating the Algorithm
For classification problems the metrics used to evaluate an algorithm are accuracy, confusion matrix, precision recall, and F1 values. Execute the following script to find these values:

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

The accuracy achieved for by our random forest classifier with 20 trees is 98.90%. Unlike before, changing the number of estimators for this problem didn't significantly improve the results, as shown in the following chart. Here the X-axis contains the number of estimators while the Y-axis shows the accuracy.

Accuracy vs number of estimators
98.90% is a pretty good accuracy, so there isn't much point in increasing our number of estimators anyway. We can see that increasing the number of estimators did not further improve the accuracy.

To improve the accuracy, I would suggest you to play around with other parameters of the RandomForestClassifier class and see if you can improve on our results.

Made with stackabuse.com

Online courses:
http://stackabu.se/data-science-python-pandas-sklearn-numpy
http://stackabu.se/python-data-science-machine-learning-bootcamp
http://stackabu.se/machine-learning-hands-on-python-data-science
