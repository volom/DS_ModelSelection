"""
In this example, we first load the iris dataset from sklearn, split 
the data into training and testing sets, and train a decision tree classifier. 
The rules are extracted from the decision
tree using the threshold, feature, children_left, children_right, 
and value attributes of the decision tree. These rules are then added 
to a fuzzy control system and used to create a fuzzy inference system,
which can be used to make predictions based on inputs.

It is important to note that this is just one example and that the 
specific implementation will depend on the specifics of the problem and the nature of the data.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
import skfuzzy as fuzz

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train a decision tree classifier on the training data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Predict the class labels for the testing data
y_pred = clf.predict(X_test)

# Create a pandas dataframe from the test data and predictions
df = pd.DataFrame(np.hstack([X_test, y_test.reshape(-1, 1), y_pred.reshape(-1, 1)]), 
                  columns=iris['feature_names'] + ['True Class', 'Predicted Class'])

# Use the decision tree rules to generate fuzzy rules
rules = []
for i, row in df.iterrows():
    antecedent = []
    for j, feature in enumerate(iris['feature_names']):
        low = np.percentile(X[:,j], 25)
        high = np.percentile(X[:,j], 75)
        if row[feature] <= low:
            antecedent.append(fuzz.trapmf(row[feature], [low, low, row[feature], row[feature]]))
        elif row[feature] > low and row[feature] < high:
            antecedent.append(fuzz.trapmf(row[feature], [low, row[feature], row[feature], high]))
        elif row[feature] >= high:
            antecedent.append(fuzz.trapmf(row[feature], [row[feature], row[feature], high, high]))
    consequent = [0, 0, 0]
    consequent[int(row['Predicted Class'])] = 1
    rules.append(fuzz.Rule(antecedent, consequent))

# Define a fuzzy system using the generated rules
iris_system = fuzz.ControlSystem(rules)
iris_simulation = fuzz.ControlSystemSimulation(iris_system)

# Use the fuzzy system to make predictions on new data
new_data = X_test[0].reshape(1, -1)
iris_simulation.inputs({iris['feature_names'][i]: new_data[0][i] for i in range(new_data.shape[1])})
iris_simulation.compute()

# Print the crisp output of the fuzzy system
print("Probabilities of each class: ", iris_simulation.output['prediction'])




# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from skfuzzy import control as ctrl

# # Load the data
# df = pd.read_csv('default_data.csv')

# # Split the data into features (X) and target (y)
# X = df.drop('default', axis=1)
# y = df['default']

# # Train the decision tree classifier
# clf = DecisionTreeClassifier()
# clf.fit(X, y)

# # Create a Fuzzy Control System
# def_ctrl = ctrl.ControlSystem([ctrl.Consequent(np.arange(0, 1.05, 0.05), 'default')])

# # Extract the rules from the decision tree
# rules = []
# for i, v in enumerate(clf.tree_.threshold):
#     antecedent = []
#     for j, f in enumerate(X.columns):
#         if clf.tree_.feature[i] == j:
#             if clf.tree_.children_left[i] == -1:
#                 membership = 'high' if y[i] else 'low'
#             else:
#                 if clf.tree_.value[clf.tree_.children_left[i]][0][0] < clf.tree_.value[clf.tree_.children_right[i]][0][0]:
#                     membership = 'low'
#                 else:
#                     membership = 'high'
#             antecedent.append((f, membership))
#     if antecedent:
#         rules.append(ctrl.Rule(antecedent, def_ctrl.consequents['default'][y[i]]))

# # Add the rules to the control system
# def_ctrl.rules = rules

# # Create a fuzzy inference system using the control system
# def_fis = ctrl.ControlSystemSimulation(def_ctrl)

# # Provide inputs to the fuzzy inference system
# def_fis.input['feature1'] = x1
# def_fis.input['feature2'] = x2
# ...

# # Simulate the fuzzy inference system
# def_fis.compute()

# # Get the output of the fuzzy inference system
# def_prob = def_fis.output['default']
