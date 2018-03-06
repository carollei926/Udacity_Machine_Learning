### Decision Trees in sklearn
# Note: This quiz requires you to find an accuracy of 100% on the training set.
# Of course, this screams overfitting! If you pick very large values for your parameters,
# you will fit the training set very well, but it may not be the best model.
# Try to find the smallest possible parameters that do the job, which has less chance
# of overfitting, although this part won't be graded.

# Import statements
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:,0:2]
y = data[:,2]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.

### model = DecisionTreeClassifier() # This returns 100%, next play with parameters
model = DecisionTreeClassifier(max_depth=7, min_samples_split=2, min_samples_leaf=1)
# max_depth=1, acc=0.666
# max_depth=2, acc=0.688
# max_depth=3, acc=0.823
# 4, 0.83; 5, 0.969; 6, 0.98958; 7,1
## set max_depth to 7, next adjust min_samples_split (original value=1)
# min_samples_split=2, acc=1 --> good!
# min_samples_split=3, acc=0.989583 --> need to set min_samples_split=2
## next adjust min_samples_leaf (original value=1)
# min_samples_leaf=2, acc=0.989583
### Smallest possible parameters:


# TODO: Fit the model.
model.fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)

### result: acc=1
