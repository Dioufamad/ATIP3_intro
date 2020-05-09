# SOURCES :
# * https://dzone.com/articles/selecting-optimal-parameters-for-xgboost-model-tra
# * http://andrejusb.blogspot.com/2019/03/selecting-optimal-parameters-for.html
# * https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/
#
# **What is XGBoost?**
# * Xtreme gradient Boosting (XGBoost) is a gradient boosted trees algorithm.
# * Lets create a pattern to choose parameters in order to build models quicker
#
# Datasets used :
# *  [Pima Indians Diabetes Database](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names)
# * we included it from local (NB : it already exist in the kaggle database and the copies have been included. watch out for some discrepancies.
#
# import os
# print(os.listdir("../input"))
#
# **Training XGBoost models : **
#
# This is the Python code that runs XGBoost training step and builds a model. Training is executed by passing pairs of train/test data, which helps to evaluate training quality ad-hoc during model construction:
#
# **Key parameters : **
#
# are the ones that would affect model quality greatly.
#
# Lets assure these are already selected :
# * max_depth (more complex classification task, deeper the tree)
# * subsample (equal to evaluation data percentage)
# * objective (classification algorithm)
#
# the key parameters are :
# * **n_estimators** — the number of runs XGBoost will try to learn
# * **learning_rate** — learning speed
# * **early_stopping_rounds** — overfitting prevention, monitor the performance of the model that is being trained and stop early if no improvement in learning
#
# **Early stopping : **
#
# * avoids overfitting by attempting to automatically select the inflection point where performance on the test dataset starts to decrease while performance on the training dataset continues to improve as the model starts to overfit.
# * The performance measure may be the loss function that is being optimized to train the model (such as logarithmic loss), or an external metric of interest to the problem in general (such as classification accuracy).
# * this capability is used by specifying both a test dataset and an evaluation metric on the call to model.fit() that is used to train the model. Also specifying verbose output shows those errors
#
# **Observe XGBoost models through epochs and get final accuracy on model trained **
#
# # monitor training performance
# from numpy import loadtxt
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# # load data
# # dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# dataset = loadtxt('../input/pima-indians-diabetes.data.csv', delimiter=",")
# # split data into X and y
# X = dataset[:,0:8]
# Y = dataset[:,8]
# # split data into train and test set (trains the model on 67% of the data and evaluates the model
# # every training epoch on a 33% test dataset))
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
# # fit model no training data
# model = XGBClassifier()
# # report the binary classification error rate (“error“)
# # on a standalone test set (eval_set) while training an XGBoost model
# # classification error is reported each training iteration (verbose = true)
# # (in each training iteration, a boosted tree is added to the model and classfication is evaluated)
# eval_set = [(X_test, y_test)]
# model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)
# # make predictions for test data
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# # finally the classification accuracy is reported at the end
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# # show a description of the model parameters loadout to see what to change to better the model
# model
#
# **Evaluate XGBoost Models With Learning Curves**
#
# the idea here is to retrieve the performance of the model on the evaluation dataset and plot it to get insight into how learning unfolded while training.
#
# # plot learning curve
# from numpy import loadtxt
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from matplotlib import pyplot
# # load data
# dataset = loadtxt('../input/pima-indians-diabetes.data.csv', delimiter=",")
# # split data into X and y
# X = dataset[:,0:8]
# Y = dataset[:,8]
# # split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
# # fit model no training data
# model = XGBClassifier()
# # step s1
# eval_set = [(X_train, y_train), (X_test, y_test)]
# model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
# # make predictions for test data
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# # retrieve performance metrics
# results = model.evals_result()
# epochs = len(results['validation_0']['error'])
# x_axis = range(0, epochs)
# # plot log loss
# fig, ax = pyplot.subplots()
# ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
# ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
# ax.legend()
# pyplot.ylabel('Log Loss')
# pyplot.xlabel('Number of training epochs')
# pyplot.title('XGBoost Log Loss')
# pyplot.show()
# # plot classification error
# fig, ax = pyplot.subplots()
# ax.plot(x_axis, results['validation_0']['error'], label='Train')
# ax.plot(x_axis, results['validation_1']['error'], label='Test')
# ax.legend()
# pyplot.ylabel('Classification Error')
# pyplot.title('XGBoost Classification Error')
# pyplot.show()
#
# ======================================================
# # =============================
# # fitting a new model8888 -----------------------
# # objective : 8888
# # split data into X8888 and y8888
# X8888 = dataset[:,0:8]
# Y8888 = dataset[:,8]
# # split data into train and test sets (another similar way to set it up)
# seed8888 = 7
# test_size8888 = 0.33
# X_train8888, X_test8888, y_train8888, y_test8888 = train_test_split(X8888, Y8888, test_size=test_size8888, random_state=seed8888)
# # fit model no training data (verbose in false to hide the metrics values)
# model8888 = XGBClassifier()
# model8888.n_estimators = 1000
# # model8888.n_estimators = model8888previous.best_ntree_limit
# model8888.learning_rate = 0.01
# # model8888.early_stopping_rounds = 0.1*model8888.n_estimators
# eval_set8888 = [(X_train8888, y_train8888), (X_test8888, y_test8888)]
# model8888.fit(X_train8888, y_train8888, early_stopping_rounds=50, eval_metric=["error","logloss"], eval_set=eval_set8888, verbose=False)
# # make predictions for test data
# y_pred8888 = model8888.predict(X_test8888)
# predictions8888 = [round(value) for value in y_pred8888]
# # evaluate predictions
# accuracy8888 = accuracy_score(y_test8888, predictions8888)
# print("Accuracy: %.2f%%" % (accuracy8888 * 100.0))
# model8888
# # retrieve performance metrics
# results8888 = model8888.evals_result()
# epochs8888 = len(results8888['validation_0']['logloss'])
# x_axis = range(0, epochs8888)
# # plot log loss
# fig, ax = pyplot.subplots()
# ax.plot(x_axis, results8888['validation_0']['logloss'], label='Train')
# ax.plot(x_axis, results8888['validation_1']['logloss'], label='Test')
# ax.legend()
# pyplot.ylabel('Log Loss')
# pyplot.xlabel('Number of training epochs')
# pyplot.title('XGBoost Log Loss')
# pyplot.show()
# # plot classification error
# fig, ax = pyplot.subplots()
# ax.plot(x_axis, results8888['validation_0']['error'], label='Train')
# ax.plot(x_axis, results8888['validation_1']['error'], label='Test')
# ax.legend()
# pyplot.ylabel('Classification Error')
# pyplot.xlabel('Number of training epochs')
# pyplot.title('XGBoost Classification Error')
# pyplot.show()
# print ("best iteration : ",model.best_iteration," with score : ",model.best_score," so next, limit estimators to : ",model.best_ntree_limit)
# print ("Observation8888 : 8888")
# print ("Proposition8888 : 8888")
# # -----------------------------------------------------------------------------------------------
# # ==========================================
