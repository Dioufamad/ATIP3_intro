# fitting a new model0 -----------------------
print ("objective : to get the early rounds") 
# split data into X0 and y0
X0 = dataset[:,0:8]
Y0 = dataset[:,8]
# split data into train and test sets (another similar way to set it up)
seed0 = 7
test_size0 = 0.33
X_train0, X_test0, y_train0, y_test0 = train_test_split(X0, Y0, test_size=test_size0, random_state=seed0)
# fit model no training data (verbose in false to hide the metrics values)
model0 = XGBClassifier()
model0.n_estimators = 300
# model0.n_estimators = model0previous.best_ntree_limit
model0.learning_rate = 0.01
# model0.early_stopping_rounds = 0.1*model0.n_estimators
eval_set0 = [(X_train0, y_train0), (X_test0, y_test0)]
model0.fit(X_train0, y_train0, eval_metric=["error","logloss"], eval_set=eval_set0, verbose=False)
# make predictions for test data
y_pred0 = model0.predict(X_test0)
predictions0 = [round(value) for value in y_pred0]
# evaluate predictions
accuracy0 = accuracy_score(y_test0, predictions0)
print("Accuracy0: %.2f%%" % (accuracy0 * 100.0))
model0
# retrieve performance metrics
results0 = model0.evals_result()
epochs0 = len(results0['validation_0']['logloss'])
x_axis = range(0, epochs0)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results0['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results0['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.xlabel('Number of training epochs')
pyplot.title('XGBoost Log Loss')
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results0['validation_0']['error'], label='Train')
ax.plot(x_axis, results0['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.xlabel('Number of training epochs')
pyplot.title('XGBoost Classification Error')
pyplot.show()
print ("best iteration : ",model.best_iteration," with score : ",model.best_score," so next, limit estimators to : ",model.best_ntree_limit)
print ("Observation0 : Model best iteration can be displayed even if not entered as a fixated parameter")
print ("Proposition0 : Lets try to fixate it")
# -----------------------------------------------------------------------------------------------
# fitting a new model -----------------------
print ("objective : see what change when limiting learning to early rounds with figures until infinite")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets (another similar way to set it up)
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data (verbose in false to hide the metrics values)
model = XGBClassifier()
model.n_estimators = 1000
# model.n_estimators = modelprevious.best_ntree_limit
model.learning_rate = 0.01
# model.early_stopping_rounds = 0.1*model.n_estimators
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric=["error","logloss"], eval_set=eval_set, verbose=False)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
model
# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.xlabel('Number of training epochs')
pyplot.title('XGBoost Log Loss')
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.xlabel('Number of training epochs')
pyplot.title('XGBoost Classification Error')
pyplot.show()
print ("best iteration : ",model.best_iteration," with score : ",model.best_score," so next, limit estimators to : ",model.best_ntree_limit)
print ("Observation : early stopping after 10 rounds without improvement works")
print ("Proposition : New model will have this early stopping. lets experiment with learning rate  : lower learning rate while upping the rounds")
# -----------------------------------------------------------------------------------------------
# fitting a new model1 -----------------------
print ("objective : early stopping with lower learning rate")
# split data into X1 and y1
X1 = dataset[:,0:8]
Y1 = dataset[:,8]
# split data into train and test sets (another similar way to set it up)
seed1 = 7
test_size1 = 0.33
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=test_size1, random_state=seed1)
# fit model no training data (verbose in false to hide the metrics values)
model1 = XGBClassifier()
# model1.n_estimators = 1000
model1.n_estimators = model.best_ntree_limit
model1.learning_rate = 0.001
# model1.early_stopping_rounds = 0.1*model1.n_estimators
eval_set1 = [(X_train1, y_train1), (X_test1, y_test1)]
model1.fit(X_train1, y_train1, early_stopping_rounds=10, eval_metric=["error","logloss"], eval_set=eval_set1, verbose=False)
# make predictions for test data
y_pred1 = model1.predict(X_test1)
predictions1 = [round(value) for value in y_pred1]
# evaluate predictions
accuracy1 = accuracy_score(y_test1, predictions1)
print("Accuracy1: %.2f%%" % (accuracy1 * 100.0))
model1
# retrieve performance metrics
results1 = model1.evals_result()
epochs1 = len(results1['validation_0']['logloss'])
x_axis = range(0, epochs1)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results1['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results1['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.xlabel('Number of training epochs')
pyplot.title('XGBoost Log Loss')
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results1['validation_0']['error'], label='Train')
ax.plot(x_axis, results1['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.xlabel('Number of training epochs')
pyplot.title('XGBoost Classification Error')
pyplot.show()
print ("best iteration : ",model.best_iteration," with score : ",model.best_score," so next, limit estimators to : ",model.best_ntree_limit)
print ("Observation1 : lower accuracy is obtain and the logloss still hasnt stabilised")
print ("Proposition1 : lets try upping the learning rate")
# -----------------------------------------------------------------------------------------------
# fitting a new model2 -----------------------
print("objective : a high learning rate in an infinite estimators test")
# split data into X2 and y2
X2 = dataset[:,0:8]
Y2 = dataset[:,8]
# split data into train and test sets (another similar way to set it up)
seed2 = 7
test_size2 = 0.33
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=test_size2, random_state=seed2)
# fit model no training data (verbose in false to hide the metrics values)
model2 = XGBClassifier()
model2.n_estimators = 1000
# model2.n_estimators = model2previous.best_ntree_limit
model2.learning_rate = 0.1
# model2.early_stopping_rounds = 0.1*model2.n_estimators
eval_set2 = [(X_train2, y_train2), (X_test2, y_test2)]
model2.fit(X_train2, y_train2, early_stopping_rounds=10, eval_metric=["error","logloss"], eval_set=eval_set2, verbose=False)
# make predictions for test data
y_pred2 = model2.predict(X_test2)
predictions2 = [round(value) for value in y_pred2]
# evaluate predictions
accuracy2 = accuracy_score(y_test2, predictions2)
print("Accuracy2: %.2f%%" % (accuracy2 * 100.0))
model2
# retrieve performance metrics
results2 = model2.evals_result()
epochs2 = len(results2['validation_0']['logloss'])
x_axis = range(0, epochs2)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results2['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results2['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.xlabel('Number of training epochs')
pyplot.title('XGBoost Log Loss')
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results2['validation_0']['error'], label='Train')
ax.plot(x_axis, results2['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.xlabel('Number of training epochs')
pyplot.title('XGBoost Classification Error')
pyplot.show()
print ("best iteration : ",model.best_iteration," with score : ",model.best_score," so next, limit estimators to : ",model.best_ntree_limit)
print ("Observation2 : with higher learning rate, quicker learning & stability already at ~ 26. but not with the best accuracy accross the board")
print ("Proposition2 : try coming back to a lower learning rate but upping the early stop rounds to give alg the chance to find a better result")
# -----------------------------------------------------------------------------------------------
# fitting a new model3 -----------------------
print("objective : lower the learning rate and extend the early stopping rounds still in infinite test")
# split data into X3 and y3
X3 = dataset[:,0:8]
Y3 = dataset[:,8]
# split data into train and test sets (another similar way to set it up)
seed3 = 7
test_size3 = 0.33
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, Y3, test_size=test_size3, random_state=seed3)
# fit model no training data (verbose in false to hide the metrics values)
model3 = XGBClassifier()
model3.n_estimators = 1000
# model3.n_estimators = model3previous.best_ntree_limit
model3.learning_rate = 0.01
# model3.early_stopping_rounds = 0.1*model3.n_estimators
eval_set3 = [(X_train3, y_train3), (X_test3, y_test3)]
model3.fit(X_train3, y_train3, early_stopping_rounds=50, eval_metric=["error","logloss"], eval_set=eval_set3, verbose=False)
# make predictions for test data
y_pred3 = model3.predict(X_test3)
predictions3 = [round(value) for value in y_pred3]
# evaluate predictions
accuracy3 = accuracy_score(y_test3, predictions3)
print("Accuracy3: %.2f%%" % (accuracy3 * 100.0))
model3
# retrieve performance metrics
results3 = model3.evals_result()
epochs3 = len(results3['validation_0']['logloss'])
x_axis = range(0, epochs3)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results3['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results3['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.xlabel('Number of training epochs')
pyplot.title('XGBoost Log Loss')
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results3['validation_0']['error'], label='Train')
ax.plot(x_axis, results3['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.xlabel('Number of training epochs')
pyplot.title('XGBoost Classification Error')
pyplot.show()
print ("best iteration : ",model.best_iteration," with score : ",model.best_score," so next, limit estimators to : ",model.best_ntree_limit)
print ("Observation3 : normally a better accuracy should be obtained")
print ("Proposition3 : move one of the parameters following observations")
# -----------------------------------------------------------------------------------------------