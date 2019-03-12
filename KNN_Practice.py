# Import the required libraries

import numpy as np
import pandas as pd
import sklearn 
import os
import matplotlib.pyplot as plt


# Import the diabetes data

df = pd.read_csv("C:/Users/Abdul_Yunus/Desktop/Yunus_Personal/Learning/KNN/diabetes.csv")

# print first 5 rows of the data frame

df.head()

# Lets observe the shape of the data frame
df.shape

# We can see that we have 768 observations and 9 variables
# Lets create numpy array for features and target variable

x = df.drop('Outcome', axis = 1).values
y = df['Outcome'].values

# Let's split the data randomly into training and test set.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 123, stratify = y)

# stratify : array-like or None (default is None)If not None, data is split in a stratified fashion, using this as the class labels.

# Let's create a classifier using k-Nearest Neighbors algorithm.

from sklearn.neighbors import KNeighborsClassifier

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,9)

train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    # set up a KNN classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors= k)
    
    # Fit the model
    knn.fit(x_train, y_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(x_train, y_train)
    
    # Compute accuracy on test set
    test_accuracy[i] = knn.score(x_test, y_test)
    
# lets Generate the plots

plt.title('K-NN Varying number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training_Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# We can observe from the plot the we get maximum accuracy at k = 8. so lets create a calssifier at k= 8

knn = KNeighborsClassifier(n_neighbors= 8)

# Fit the model

knn.fit(x_train, y_train)

# Check the accuracy at the test data set

knn.score(x_test,y_test)

# Lets build a confusion matrix

from sklearn.metrics import confusion_matrix

# Let us get the prediction from the classifier we had above

y_pred = knn.predict(x_test)

confusion_matrix(y_test, y_pred)

# Confusion matrix can also be obtained using crosstab method of pandas.

pd.crosstab(y_test, y_pred, rownames = ['True'], colnames = ['Predicted'], margins = True)


# Classification Report
# Another important report is the Classification report. It is a text summary of the precision, recall, F1 score for each class. 
# Scikit-learn provides facility to calculate Classification report using the classification_report method.

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

'''
 Cross Validation

Now before getting into the details of Hyperparamter tuning, let us understand the concept of Cross validation.

The trained model's performance is dependent on way the data is split. It might not representative of the model’s ability to generalize.

The solution is cross validation. Cross-validation is a technique to evaluate predictive models by partitioning the original sample into a training set to train the model, and a test set to evaluate it.

In k-fold cross-validation, the original sample is randomly partitioned into k equal size subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k-1 subsamples are used as training data. 
The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results from the folds can then be averaged (or otherwise combined) to produce a single estimation. The advantage of this method is that all observations are used for both training and validation, and each observation is used for validation exactly once.

Hyperparameter tuning

The value of k (i.e 8) we selected above was selected by observing the curve of accuracy vs number of neighbors. This is a primitive way of hyperparameter tuning.

There is a better way of doing it which involves:

1) Trying a bunch of different hyperparameter values

2) Fitting all of them separately

3) Checking how well each performs

4) Choosing the best performing one

5) Using cross-validation every time

Scikit-learn provides a simple way of achieving this using GridSearchCV i.e Grid Search cross-validation.
'''
