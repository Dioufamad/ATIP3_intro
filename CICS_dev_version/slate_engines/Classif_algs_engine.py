###--------------------- V3_end1_This is the location of some functions caring for whatever-----------------------

###---------------------IMPORTS FOR CLASSIFICATION
import numpy as np # for np.shape(x) giving n rows and k cols of a array as (n,k)
import pandas as pd # for pushing dataframes functions in order to report or keep track
import locale
from sklearn.preprocessing import StandardScaler # a scaling function
# ---RF installation
from sklearn.ensemble import RandomForestClassifier
from math import sqrt # for mtry computation
from slate_engines.fs_engine import length_features_list # for numbers of fts computation in mtry
#--- XGBoost installation
from xgboost import XGBClassifier
#--- GBM installation
from sklearn.ensemble import GradientBoostingClassifier
#----SVM installation
from sklearn.svm import SVC
from sklearn.feature_selection import RFE # in test for RFE implementation

# #****************
# # initial place for DNN using Keras imports
# def env_specific_loads_for_DNN(aseed):
# 	global optimizers, l2, Dropout, Sequential, Dense, Activation, BatchNormalization
# 	# import numpy as np
# 	# import pandas as pd
# 	# *********DNN using Keras imports (done here and not on top to avoid multiprocessing issues
# 	# ----------a random seed initialisation to fixate some randomness due to intialisation of librairies or ibjects
# 	np.random.seed(aseed)  # first intention was to minimize variations in neural networks ## use 0 if not working
# 	import tensorflow.compat.v1 as tf
# 	tf.disable_v2_behavior()
# 	# from tensorflow import set_random_seed # as TensorFlow backend is used by Keras DNNs and TensorFlow has its own random number generator, that must also be seeded
# 	tf.set_random_seed(aseed)  ## use 0 if not working
# 	# ----DNN implementation
# 	from keras import optimizers  # for the optimizers
# 	from keras.regularizers import l2  # for the weight decay as L2 regularization
# 	from keras.layers import Dropout  # from the dropout regularization essay
# 	from keras.models import Sequential  # for the model structure
# 	from keras.layers import Dense  # for each input layer mostly hidden layers
# 	from keras.layers import Activation  # for activation functions
# 	# from sklearn.preprocessing import LabelEncoder # to change the Response values from string to classes 0 and 1 # not needed at the moment
# 	from keras.layers.normalization import BatchNormalization  # for batch normlisation betwee X.w and batchnormalised (X.w) + b
# 	# in case
# 	# from keras.constraints import maxnorm
# 	# from keras.optimizers import SGD
# 	# from keras.wrappers.scikit_learn import KerasClassifier
# 	# from sklearn.model_selection import cross_val_score
# 	# from sklearn.model_selection import StratifiedKFold
# 	# from sklearn.pipeline import Pipeline
# 	# *********
# #*********


###---------------------(COMPLEMENTARY) IMPORTS FOR REGRESSION

#====================================================================

# ---------------------Variables to initialise------------------------------------------
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') #for setting the characters format
#====================================================================

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>functions for proper Classification algorithms
# -------------------------------here is the stock steps for a machine learning algorithm use code until prediction
# def classifier_introduction(classifier_version,num_cores,aseed):
# 	if classifier_version == "XGBoost_C_1" :
# 		model = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, subsample=0.8, n_estimators=700, colsample_bytree=0.8, silent=1, nthread=num_cores, seed=aseed)  # stock the method
# 	return model
# def classifier_model_training(model,train_x,train_y):
# 	model_fitted = model.fit(train_x, train_y)  # fit the method to the training data to get a classifier and predict with it later
# 	return model_fitted
# def classifier_model_prediction(model_fitted,test_x):
# 	model_prediction = model_fitted.predict(test_x)
# 	return model_prediction
# For many algorithms, this process is quite similar so we unify those.
# A condittion on the identity of the algorithm presently used will give us access to specific treatments following the algorithms.
# Algorithms requiring a much specific treatment will be isolated in their own function with classifier introduction-training-prediction in one direct action

#-------------------------------functions for non complex code for classifiers
def classifier_introduction(tag_alg,tag_alg_mark, trainframe_x, trainframe_y,aseed,encoded_classes,num_threads_heavy):
	np.random.seed(aseed)  # initiated again as a mesure of security so that even if random state not given, the none value make it catch this as a random seed that stays the same acrross repetitons of the same experiement
	# num_threads following situations :
	# num_threads_heavy is 38 for a 40 cores machine. used to exploit multiprocessing when authorized
	if tag_alg == "RF": # case of RF as alg
		# common hyperparams values
		mtry = int(round(sqrt(length_features_list(trainframe_x, 0))))  # see docs as it is the default value # int() because #mtry/max_features in rf of sklearn requires  int
		# obtain classes names
		class0 = encoded_classes[0]
		class1 = encoded_classes[1]
		# deprecated old way of obtaining the classes names in order is # sorted_unique_classes_in_y = sorted(set(trainframe_y.iloc[:, 0]))	# class0 = sorted_unique_classes_in_y[0] # class1 = sorted_unique_classes_in_y[1]
		class0_in_y = trainframe_y.iloc[:, 0].value_counts()[class0]  # num of res
		class1_in_y = trainframe_y.iloc[:, 0].value_counts()[class1]  # num of sens
		prop_class0_in_y = 1.0 * class0_in_y / len(trainframe_x.index)  # proportion of class "res" #we divide by the length of trainframe_x because its the dataframe that lastly had info attached to the num of samples ; its the same than dividing by length trainframe_y really
		prop_class1_in_y = 1.0 * class1_in_y / len(trainframe_x.index)  # proportion of class "sen"
		if tag_alg_mark == "Mark1Vpar": #codename = RF_default (see Sl_algs_descriptor.txt for params value)
			model = RandomForestClassifier(random_state=aseed)
		elif tag_alg_mark == "Mark2Vpar":  ##!!!## new linh params # codename = RF_Linh_params
			# ntrees = 10 # testing
			ntrees = 1000
			model = RandomForestClassifier(max_features=mtry, n_estimators=ntrees, class_weight={class0: prop_class1_in_y, class1: prop_class0_in_y},random_state=aseed)
		elif tag_alg_mark == "Mark3Vpar":  ##!!!## new linh params # codename = RF_Linh_params_ntreesIs100
			ntrees = 100
			model = RandomForestClassifier(max_features=mtry, n_estimators=ntrees, class_weight={class0: prop_class1_in_y, class1: prop_class0_in_y},random_state=aseed)
		elif tag_alg_mark == "Mark4Vpar":  ##!!!## new linh params # codename = RF_Linh_params_ntreesIs10
			ntrees = 10
			model = RandomForestClassifier(max_features=mtry, n_estimators=ntrees, class_weight={class0: prop_class1_in_y, class1: prop_class0_in_y},random_state=aseed)
		# Explanations :
		# n_estimators is ntrees
		# the class_weight is 2 classes and each one attached the other class proportion
		# the importance is by default because the gini criterion already by default
		# Eg : randomForest(ol2_train_x, ol2_train_y, mtry=mtry_mdl_allfts, importance=TRUE, ntree=ntree, classwt=c(Res=prop.sens_mdl_allfts, Sen=prop.res_mdl_allfts))
	elif tag_alg == "XGB": # case of XGBoost as alg
		if tag_alg_mark == "Mark1Vpar":  # codename = XGB_default (see Sl_algs_descriptor.txt for params value)
			model = XGBClassifier(seed=aseed)
		if tag_alg_mark == "Mark2Vseq":  # codename = XGB_Stephan_params (see Sl_algs_descriptor.txt for params value)
			model = XGBClassifier(max_depth=6, learning_rate=0.05, subsample=0.8, n_estimators=700,
														 colsample_bytree=0.8, silent=1,
														 nthread=num_threads_heavy, seed=aseed)
		if tag_alg_mark == "Mark2Vpar":  # codename = XGB_Stephan_params (see Sl_algs_descriptor.txt for params value)
			model = XGBClassifier(max_depth=6, learning_rate=0.05, subsample=0.8, n_estimators=700,
														 colsample_bytree=0.8, silent=1,
														 seed=aseed) # nthread=num_cores_heavy not used because it risks undeertaking the allocated cores to il1 xproc
	elif tag_alg == "GBM":  # case of GBM as alg
		if tag_alg_mark == "Mark1Vpar":  # codename = GBM_default (see Sl_algs_descriptor.txt for params value)
			model = GradientBoostingClassifier(random_state=aseed)
	return model

def classifier_model_training(model, trainframe_x, trainframe_y):
	# the fit function in the case of most algs take 2 arrays in the next specifications otherwise error can be sent :
	# (the array_x is multi_dimensional ie shape(n_samples,#fts) and the array_y is 1_dimensional ie shape(n_samples,)
	train_x_as_array = np.array(trainframe_x) # a multi_d array is needed
	train_y_as_array = np.array(trainframe_y.iloc[:, 0].tolist())  # the array of train_y has to be 1_d ie created from a list made from the response column info
	model_fitted = model.fit(train_x_as_array, train_y_as_array)  # fit the method to the training data to get a classifier and predict with it later
	return model_fitted

def classifier_model_prediction(model_fitted, testframe_x, prediction_type):
	# same as in training the function used here ask for an array_x that is multi_dimensional ie shape(n_samples,#fts)
	test_x_as_array = np.array(testframe_x)
	if prediction_type == "prob":
		model_prediction = model_fitted.predict_proba(test_x_as_array) # putting brackts aroung fix the need to reshpe error if working with series
	elif prediction_type == "pred":
		model_prediction = model_fitted.predict(test_x_as_array)  # putting brackts aroung fix the need to reshpe error
	# show an output as proof
	print("1 model prediction done : ", model_prediction)
	return model_prediction

def classifier_as_SVM_intro_train_pred(tag_alg_mark,trainframe_x, trainframe_y,feature_val_type,binary_classes_le,testframe_x, prediction_type,aseed):
	np.random.seed(aseed)  # initiated again as a mesure of security
	#---- choosing hyperparams combinaison and building the model
	if tag_alg_mark == "Mark1Vpar":  # codename = SVM linear
		kernel_choice = "linear"
		activate_probs = True
		model = SVC(kernel=kernel_choice,probability=activate_probs)
	elif tag_alg_mark == "Mark2Vpar":  # codename = SVM RBF
		kernel_choice = "rbf"
		kernel_coef_choice = 'auto'
		activate_probs = True
		model = SVC(kernel=kernel_choice, gamma=kernel_coef_choice,probability=activate_probs)
	# ---- preprocessing the data before training and testing following type of profile : scale values of features (only if not categorical values)
	if feature_val_type == "real" : # case of reals values profiles
		scaler = StandardScaler()
		scaler.fit(trainframe_x)
		train_x_as_array = scaler.transform(trainframe_x)
		test_x_as_array = scaler.transform(testframe_x)
	# train_x_as_array = StandardScaler().fit_transform(trainframe_x)
	else:
		train_x_as_array = np.array(trainframe_x)
		test_x_as_array = np.array(testframe_x)
	# for the responses classes, the values have to be encoded (using here an encoding done just after data_management)
	# binary_classes_le.classes_ gives an array of the classes
	train_y_as_array = np.array(binary_classes_le.transform(trainframe_y.iloc[:, 0].tolist()))  # get array from frames of thruths with values encoded
	#---- training
	model_fitted = model.fit(train_x_as_array, train_y_as_array)
	# model hyperparams values is this following :
	# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
	#     decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
	#     kernel='linear', max_iter=-1, probability=False, random_state=None,
	#     shrinking=True, tol=0.001, verbose=False)
	#---- prediction
	if prediction_type == "prob":
		model_prediction = model_fitted.predict_proba(test_x_as_array) # putting brackts aroung fix the need to reshpe error if working with series
	elif prediction_type == "pred":
		model_prediction = model_fitted.predict(test_x_as_array)  # putting brackts aroung fix the need to reshpe error
	# show an output as proof
	print("1 model prediction done : ", model_prediction)


	# elif tag_alg_mark == "Mark3Vseq":  # codename = SVM RFE
	# 	kernel_type = "linear"
	# 	use_rfe = True
	# elif tag_alg_mark == "Mark4Vseq":  # codename = SVM RBF RFE
	# 	kernel_type = "rbf"
	# 	use_rfe = True

	# #---putting in place rfe
	# if use_rfe == True:
	# 	# choosing rfe hyperparams
	# 	if trainframe_x.shape[1] < trainframe_x.shape[0]: # only in omc ncols < nrows
	#
	# 		if trainframe_x.shape[1]:
	#
	# 		else:
	# 			n_fts_to_select = None
	#
	# 		step_value = 1
	#
	# 	else:
	# 		n_fts_to_select = None
	# 		step_value = 0.2
	#
	#
	# 	rfe = RFE(estimator=model_as_estimator_or_not_yet, n_features_to_select=1, step=1) # create rfe object to rank fts
	# training
	# rfe.fit(X, y)
	# prediction

	return model_prediction

def classifier_as_Keras_DNN_intro_train_pred(tag_alg_mark,trainframe_x, trainframe_y,feature_val_type,binary_classes_le,testframe_x, prediction_type,aseed,num_threads_heavy):
	# ~~~~~~~~~~~~ creating model with given values of hyperparams
	# ---- necessary imports and random env init :
	np.random.seed(aseed)
	import tensorflow as tf
	tf.set_random_seed(aseed)
	from tensorflow.python.keras import optimizers
	from tensorflow.python.keras.regularizers import l2
	from tensorflow.python.keras.layers import Dropout
	from tensorflow.python.keras.models import Sequential
	from tensorflow.python.keras.layers import Dense
	from tensorflow.python.keras.layers import Activation
	from tensorflow.python.keras.layers.normalization import BatchNormalization
	from tensorflow.python.keras import backend as K
	# common hyperparams values
	layers = (512, 256, 64)
	input_layer_dim = trainframe_x.shape[1]
	kernel_initializer_type = 'normal'
	kernel_regularizer_decay_val = 1e-05
	bn = True
	input_hidden_layers_activation_type = 'relu'
	output_layer_activation_type = 'sigmoid'
	add_dropout = True
	dropout_val = 0.6
	loss_function = 'binary_crossentropy'
	optimizer_lr = 0.1
	epochs_num = 100
	batch_size_default_val = 100
	# ~~~~~~~ modified hyperparams values :
	# optional to use dropout on the input layer
	# epsilon = 1e-05 to satisfy Theano demands for portability, mode = 0 feature-wise normalization,
	# axis is left to default instead of 1 for samples (for better management )
	# by default if weights=None:
	# beta_init = 'zero', gamma_init = 'one'
	# beta_regularizer=None,gamma_regularizer=None
	# beta_constraint = None, gamma_constraint = None
	# moving_mean_initializer = 'zeros', moving_variance_initializer = 'ones'
	# eliminate mode for tf.keras and simplify params because unknown territory (bn_epsilon=1e-05, bn_mode=0, bn_momentum=0.9, bn_weights=None, # before)
	# eliminate and simplify some params because unknown territory (, optimizer_epsilon=None, optimizer_decay=0.0 # before)
	if tag_alg_mark == "Mark1Vseq": # codename = DNN_recommended_zhou2019_v1 with session management for multithreaded runs on automatic choice, verbose = 1, batch = half_pop
		# the hyperparams values :
		batch_size_strategy = "half_pop"
		verbose_in_fit = 1
		# num_cores_intra_op = 19
		# num_cores_inter_op = 19
		# ~~~~ the session management V1 num cores = all available
		config_par_autodetect_log = tf.ConfigProto()
		# intra_op_parallelism_threads=num_cores_intra_op, inter_op_parallelism_threads=num_cores_inter_op not set
		# allow_soft_placement=True, log_device_placement=True not set
		session = tf.Session(config=config_par_autodetect_log)
		K.set_session(session)
	elif tag_alg_mark == "Mark2Vseq": # codename = DNN_recommended_zhou2019_v1 with session management for multithreaded runs on automatic choice, verbose = 0, batch = half_pop
		# the hyperparams values :
		batch_size_strategy = "half_pop"
		verbose_in_fit = 0
		# num_cores_intra_op = 19
		# num_cores_inter_op = 19
		# ~~~~ the session management V2 num cores = all available
		config_par_autodetect_log = tf.ConfigProto()
		# intra_op_parallelism_threads=num_cores_intra_op, inter_op_parallelism_threads=num_cores_inter_op not set
		# allow_soft_placement=True, log_device_placement=True not set
		session = tf.Session(config=config_par_autodetect_log)
		K.set_session(session)
	elif tag_alg_mark == "Mark3Vseq": # codename = DNN_recommended_zhou2019_v1 with session management for multithreaded runs on automatic choice, verbose = 1 and batch = one as full pop
		# the hyperparams values :
		batch_size_strategy = "one_batch"
		verbose_in_fit = 1
		# num_cores_intra_op = 19
		# num_cores_inter_op = 19
		# ~~~~ the session management V2 num cores = all available
		config_par_autodetect_log = tf.ConfigProto()
		# intra_op_parallelism_threads=num_cores_intra_op, inter_op_parallelism_threads=num_cores_inter_op not set
		# allow_soft_placement=True, log_device_placement=True not set
		session = tf.Session(config=config_par_autodetect_log)
		K.set_session(session)
	elif tag_alg_mark == "Mark4Vseq": # codename = DNN_recommended_zhou2019_v1 with session management for multithreaded runs on automatic choice, verbose = 0 and batch = one as full pop
		# the hyperparams values :
		batch_size_strategy = "one_batch"
		verbose_in_fit = 0
		# num_cores_intra_op = 19
		# num_cores_inter_op = 19
		# ~~~~ the session management V2 num cores = all available
		config_par_autodetect_log = tf.ConfigProto()
		# intra_op_parallelism_threads=num_cores_intra_op, inter_op_parallelism_threads=num_cores_inter_op not set
		# allow_soft_placement=True, log_device_placement=True not set
		session = tf.Session(config=config_par_autodetect_log)
		K.set_session(session)
	else : # the default params and session management as in a Mark0Vseq
		# the hyperparams values :
		batch_size_strategy = "half_pop"
		verbose_in_fit = 1
		num_cores_intra_op = int(np.floor(num_threads_heavy / 2)) # one half of 38 cores if 40-2 are given or as in tests # 19 initialy
		num_cores_inter_op = int(np.floor(num_threads_heavy / 2)) # other half of 38 cores if 40-2 are given or as in tests int(np.floor(19.99))
		# ~~~~ the session management V4.1 num cores = 10
		config_par_autodetect_log = tf.ConfigProto(intra_op_parallelism_threads=num_cores_intra_op,
												   inter_op_parallelism_threads=num_cores_inter_op)
		# allow_soft_placement=True, log_device_placement=True not set
		session = tf.Session(config=config_par_autodetect_log)
		K.set_session(session)

	#------- Introduce the model
	model = Sequential()
	# the input layer
	model.add(Dense(layers[0], input_dim=input_layer_dim, kernel_initializer=kernel_initializer_type, kernel_regularizer=l2(kernel_regularizer_decay_val)))
	if bn:
		model.add(BatchNormalization()) # eliminate mode for tf.keras and simplify params because unknown territory (epsilon=bn_epsilon, mode=bn_mode, momentum=bn_momentum, weights=bn_weights # before)
	model.add(Activation(input_hidden_layers_activation_type))
	if add_dropout:
		model.add(Dropout(dropout_val))
	# the hidden layers
	for i in layers[1:]:
		model.add(Dense(i, kernel_initializer=kernel_initializer_type, kernel_regularizer=l2(kernel_regularizer_decay_val)))
		if bn:
			model.add(BatchNormalization()) # eliminate mode for tf.keras and simplify params because unknown territory (epsilon=bn_epsilon, mode=bn_mode, momentum=bn_momentum, weights=bn_weights # before)
		model.add(Activation(input_hidden_layers_activation_type))
		if add_dropout:
			model.add(Dropout(dropout_val))
	# the output layer
	model.add((Dense(1, kernel_initializer=kernel_initializer_type)))
	if bn:
		model.add(BatchNormalization()) # eliminate mode for tf.keras and simplify params because unknown territory (epsilon=bn_epsilon, mode=bn_mode, momentum=bn_momentum, weights=bn_weights # before)
	model.add(Activation(output_layer_activation_type))
	# Compile the model
	adagrad_mark1_as_opt = optimizers.Adagrad(lr=optimizer_lr)  # eliminate and simplify some params because unknown territory (, epsilon=optimizer_epsilon, decay=optimizer_decay # before)
	model.compile(loss=loss_function, optimizer=adagrad_mark1_as_opt)
	# # testing this
	# print("1 dnn model architecture made : ")
	# # testing this

	# ----preprocessing the data before training and testing following type of profile : scale values of features (only if not categorical values)
	# from sklearn.preprocessing import StandardScaler  # to scale again the real values features ## already imported
	if feature_val_type == "real" : # case of reals values profiles
		scaler = StandardScaler()
		scaler.fit(trainframe_x)
		train_x_as_array = scaler.transform(trainframe_x)
		test_x_as_array = scaler.transform(testframe_x)
		# train_x_as_array = StandardScaler().fit_transform(trainframe_x)
	else:
		train_x_as_array = np.array(trainframe_x)
		test_x_as_array = np.array(testframe_x)
	# for the responses classes, the values have to be encoded (using here an encoding done just after data_management)
	# binary_classes_le.classes_ gives an array of the classes
	train_y_as_array = np.array(binary_classes_le.transform(trainframe_y.iloc[:, 0].tolist()))  # get array from frames of thruths with values encoded

	# ~~~~~~~~~~~~ training model with given values of hyperparams
	# choose the batch size
	if batch_size_strategy == "half_pop":
		chosen_batch_size = int(np.floor(trainframe_x.shape[0] / 2))
	elif batch_size_strategy == "one_batch":
		chosen_batch_size = trainframe_x.shape[0]
	else:
		chosen_batch_size = batch_size_default_val
	model.fit(train_x_as_array, train_y_as_array, epochs=epochs_num, batch_size=chosen_batch_size, verbose=verbose_in_fit)  # giving this to a variable only stock a history object
	# do this to store the model fitted
	model_fitted = model
	# # testing this
	# print("1 dnn model trained : ")
	# # testing this

	# ~~~~~~~~~ predicting with model fitted
	# same as in training the function used here ask for an array_x that is multi_dimensional ie shape(n_samples,#fts)

	if prediction_type == "prob":
		prob_class_pos_array_column = model_fitted.predict(test_x_as_array)  # model_fitted.predict(test_x_as_array) is a (1,1) shape array
		prob_class_neg_array_column = 1 - prob_class_pos_array_column # apply a fucntion to each lines of the data
		model_prediction = np.concatenate((prob_class_neg_array_column,prob_class_pos_array_column),axis=1)
		# model_prediction = np.array([[prob_class_neg, prob_class_pos]])  # based on a = np.array([[1, 1], [2, 2], [3, 3]]) is a (3,2) shape array
	elif prediction_type == "pred":
		code_pred_class_array_column = model_fitted.predict_classes(test_x_as_array)  # model_fitted.predict(test_x_as_array) is a (1,1) shape array
		# code_pred_class_array_column = 1 - code_pred_class_array_column # 1 - 1 = 0 and  1 - 0 = 1 o from 1 to 0 or vice versa
		# binary_classes_le is an encoding done just after data_management
		model_prediction = binary_classes_le.inverse_transform([code_pred_class_array_column]) ## to finish fixing ## make sure arry is right shape # if to be used for future exploitation compare and make in same shape than regular output of model.predict for most models
	# testing this
	K.clear_session()  # ~~~~end of session management # testing this
	print("1 dnn model prediction done : ",model_prediction)
	# print(model_prediction)
	# testing this

	return model_prediction

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>helper functions for classifiers
def prediction_calling(prediction_array,prediction_call_col,thres,encoded_classes,initial_seed_value):
	for r in list(range(np.shape(prediction_array)[0])): # range 1 (line 1 is the 1st line) and line n (n=num_of_rows)
		if prediction_array[r,0] > thres: # 1 for class in col one and it is class cited 1st in classwt
			prediction_call_col.append(encoded_classes[0]) # predictions of all samples in the val set of the fold # affect class 1 if prob column 1 is > thres because col 1 is class 1
		elif prediction_array[r,0] == thres:
			# a seed is given to be able to use a random seed that stays the same acrross repetitons of the same experiement
			np.random.seed(initial_seed_value) # (actual_seed_class_class_calling)
			prediction_call_col.append(np.random.choice(encoded_classes)) # take one at random but restart the seed runner to hahve another chaznce to dodge this situation (will happens its dodged)
		else : # only case left
			prediction_call_col.append(encoded_classes[1])
	return prediction_call_col # end of calling pos or neg

def raw_predictions_pusher(raw_predictions_array,presently_explored_fold,list_of_indexes_of_test,encoded_classes,list_to_push_into,prediction_call_col,aseed):
	df_to_push = pd.DataFrame()
	# creating the 2nd col content before the 1st col because the 1st col uses it to be created
	content_Fold_column = np.repeat(presently_explored_fold, len(list_of_indexes_of_test))  # the longest column is the indexes in the fold column so start from it
	name_Fold_column = "Fold"
	# cretae the 2nd col content
	content_Seed_column = np.repeat(aseed,len(content_Fold_column))
	name_Seed_column = "Seed"
	# the first column is ready so add it
	df_to_push[name_Seed_column] = content_Seed_column
	# the 2nd column is ready so add it
	df_to_push[name_Fold_column] = content_Fold_column
	# create the 3rd col content
	content_test_samples_index_column = list_of_indexes_of_test
	name_test_samples_index_column = "Testset_samples_indexes"
	# the 3rd column is ready so add it
	df_to_push[name_test_samples_index_column] = content_test_samples_index_column
	# create the 4th col content
	content_proba_class0_column = raw_predictions_array[:, 0]
	name_proba_class0_column = "Proba_Class0_ie_" + encoded_classes[0]
	# the 4th column is ready so add it
	df_to_push[name_proba_class0_column] = content_proba_class0_column
	# create the 5th col content
	content_proba_class1_column = raw_predictions_array[:, 1]
	name_proba_class1_column = "Proba_Class1_ie_" + encoded_classes[1]
	# the 5th column is ready so add it
	df_to_push[name_proba_class1_column] = content_proba_class1_column
	# create the 6th and last col content
	content_class_called_column = prediction_call_col
	name_class_called_column = "Class_called"
	# the 6th and last col is ready so add it
	df_to_push[name_class_called_column] = content_class_called_column
	list_to_push_into.append(df_to_push)
	return list_to_push_into
	# a list of many df, each one the x lines of raw preds with x the number of samples in the test
	# to make pd.concat on it later and have in that sole df all the raw preds of one type of model (either allfts or omc of il1 on all the data/ol2)
#back story :
# we could also create a dict and make a df with it but it will order our columns by alphabetical order and we dont want that
	# joiner = {"Fold":content_Fold_column, "Test_sample_index": content_PDXindex_col, mdl_fitted_classes[0]:content_class0_column,mdl_fitted_classes[1]:content_class1_column}
	# df_to_push = pd.DataFrame(joiner)
## stitch together 4 cols ("Fold", "PDXindex","Res","Sen") to form a table : 1-the fold x the # of lines in it; 2-the sample name; 3-2 cols that are the content of the predicrtions in the val set
	# get the names of samples attached to the predictions made
	# stich foldsxtimes # of samples and the predictions with their names before
	# name the cols
# push into a keep tracker of predictions probs in folds, these formatted results of prediction for a fold


def called_predictions_pusher(prediction_call_col,list_of_indexes_of_test,list_to_push_into):
	indexes_and_calls_zipped = list(zip(list_of_indexes_of_test,prediction_call_col)) # create a list in the fashion [(index,prediction called)]
	list_to_push_into = list_to_push_into + indexes_and_calls_zipped # add the list content to a collector to make a longer list of duos and make a df of 2cols later
	return list_to_push_into # give the collector back
#back story :
# name the elts of the pred of all samples in val set as the names of the rows in the test set (one sample pred get back the # of row it had)
# add the predictions of all samples ( one really) to a collector

def multi_models_called_predictions_pusher_to_df(prediction_call_col_mdl1,prediction_calls_col_other_mdls,list_of_indexes_of_test):
	mdl1calls_othermdlscalls_zipped = list(zip(prediction_call_col_mdl1,prediction_calls_col_other_mdls,list_of_indexes_of_test))  # create a list in the fashion [(index,prediction called)]
	frame_from_the_union = pd.DataFrame(mdl1calls_othermdlscalls_zipped)
	return frame_from_the_union

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>functions for proper Regression algorithms

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


