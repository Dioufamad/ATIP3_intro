###--------------------- This is the location of some functions caring for whatever-----------------------

###---------------------IMPORTS FOR CLASSIFICATION
# import numpy as np # for computations on indexes when ding CV with any # folds value
# import itertools for CV with any # of folds value in stratification function
import locale
from multiprocessing import cpu_count, Process, JoinableQueue, current_process, Manager # next 3 lines are for the multiprocessing of the inner loop 1 (as the heavier part of the script, we need to distribute it over many processes)
import time
# import queue # imported for using queue.Empty exception if need be
from operator import itemgetter # to sort lists of tuples using directly one position in the index elements
# imports for new stratified CV
import numpy as np
from sklearn.model_selection import StratifiedKFold
###---------------------IMPORTS COMPLEMENTARY FOR REGRESSION

# ---------------------Variables to initialise------------------------------------------
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') #for setting the characters format
#====================================================================

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>classification used functions
#-------------------------------supplier of col when col is a dict
def add_entry_in_dict(dict_to_supply,key,value):
	try:
		dict_to_supply[key].append(value)
	except KeyError:  # if the key does not exit (case of the first append for a key) # except means "in case of this following event, do this : # maybe change later by "if key not if list of keys of the dict"
		dict_to_supply[key] = [value]
	return
#-------------------------------MULTIPROCESSING
#~~~~~~~~~~a multiprocessing handler
def il1_multiprocessing_handler(folds_of_IL1_col, number_of_processes,a_list_as_col_of_sorted_results_wo_their_id,job_function):
	#~~~~~~~~~~~Step 1 : the needed material
	# global _MP_STOP_SIGNAL # pulling the global _MP_STOP_SIGNAL defined much earlier on top
	# each_task_number_gallery = folds_of_IL1  ## put here the number of samples to create one process for each (ie 37 samples each) (can be superior to processes used, the rest will just wait
	# number_of_processes = tag_num_xproc  ## put here the number of cores minus 2 # test with 4 and oppose it to more samples eg. 6
	#~~~~~~~~~~~~~~~~~Step 2 : create a queue and supply it with tasks
	queue_of_tasks_to_accomplish = JoinableQueue()
	# charging the queue of tasks to do
	for one_fold_tools in folds_of_IL1_col:
		queue_of_tasks_to_accomplish.put((job_function, (one_fold_tools,)))  # add content to a queue
		if queue_of_tasks_to_accomplish.qsize() == len(folds_of_IL1_col):
			print(queue_of_tasks_to_accomplish.qsize(), "tasks charged in queue out of", len(folds_of_IL1_col), "results expected :")
			print("All tasks have been charged!")
		else:
			print(queue_of_tasks_to_accomplish.qsize(), "tasks charged in queue out of", len(folds_of_IL1_col), "results expected. Expecting additional content...")
	#~~~~~~~~~~~Step 3 : creating the processes for the group of tasks to accomplish and starting them
	# a list to stock them in case of a similar action on all of them
	processes = []
	# a shared list between the environnement of each process for them to deposit in it their results
	bench_of_results = Manager().list()  # a collector for the result of each fold carried out to sort later and make an ordered df
	# create the worker that has to carry out the function/job to do in each task (il1) using a queue of tasks to accomplish, each task in the queue is carrying out a fold of il1
	_MP_STOP_SIGNAL = False  # multi processing stop signal
	def il1_worker(queue_of_tasks, shared_list, all_folds_tools_list_col):
		while _MP_STOP_SIGNAL == False:
			job, args = queue_of_tasks.get()
			the_tuple_collected_from_the_task = job(*args)
			shared_list.append(the_tuple_collected_from_the_task)
			if len(shared_list) == len(all_folds_tools_list_col):
				print(len(shared_list), "results caught out of the", len(all_folds_tools_list_col), "results expected :")
				print("All IL1 results have been caught!")
			else:
				print(len(shared_list), "results caught out of the", len(all_folds_tools_list_col), "results expected. Expecting additional content...")
			print("A task has ended.")
			queue_of_tasks.task_done()
	# defined the multiprocessing function that will encapsulate the function launching the jobs in the queue
	for an_index in range(number_of_processes):
		p = Process(target=il1_worker, args=(queue_of_tasks_to_accomplish, bench_of_results, folds_of_IL1_col,))
		processes.append(p)  # add the process with the mission defined to a galerie
		p.start()  # ...and start it
		# If not sleeping, launching some jobs twice and omitting some.
		time.sleep(0.1)
	_MP_STOP_SIGNAL = True  # multi processing stop signal being shifted
	queue_of_tasks_to_accomplish.join()  # block until all tasks are done (wait for every process to finish) " necessary because using joinable queue
	#~~~~~~~~Step 4 : collecting the results before shutting down all processes for fear to lose the one holding the shared list
	print("************collecting the results************")
	bench_of_results_list_of_tuples_sorted_on_task_id = sorted(bench_of_results, key=itemgetter(0))
	for a_tuple in bench_of_results_list_of_tuples_sorted_on_task_id:
		a_list_as_col_of_sorted_results_wo_their_id.append(a_tuple[1])
	print("************finished collecting the results************")
	#~~~~Step 5 : Solution deployed for checking on processing shut down
	print("Curating child processes spawn for carrying out IL1...")
	print("The lenghth of the list holding the processes is", len(processes))

	for a_process_left in processes:  # loop on the process holding list and terminate them if they are alive
		print("After IL1, the", a_process_left, "(pid", a_process_left.pid, ") lifeline is", a_process_left.is_alive())
		if a_process_left.is_alive():
			a_process_left.terminate()  # takes effect only after the join
			a_process_left.join()
			print("The", a_process_left, "(pid", a_process_left.pid, ") is killed. Proof : his lifeline is", a_process_left.is_alive())
	count_alive_processes = 0  # next is a loop on the processes holding list to count alive and dead processes...
	count_dead_processes = 0
	for a_process_found in processes:
		if a_process_found.is_alive():
			count_alive_processes += 1
		else:
			count_dead_processes += 1
	print("Out of", number_of_processes, " processes used by IL1,", count_alive_processes, "are alive and", count_dead_processes, "are dead.")
	if count_alive_processes == 0:  # ... A response is displayed following all processes are dead or if they are alive (alive are displayed)
		print("All child processes successfully terminated")
	else:
		print("Caution :", count_alive_processes, "child processes are not terminated")
		print("These are the alive processes : ")
		for john_doe_process in processes:
			if john_doe_process.is_alive():
				print(john_doe_process, "(pid", john_doe_process.pid, ")")
	###end of the processes finishing function ##!! isolate in a function
	print("~~~~~~~~~~~~~ IL1 multiprocessed loop is over~~~~~~~~~~~~~~~~~~~~~")

#~~~~~~~~~~a sequential processing handler
def il1_sequential_processing_handler(folds_of_IL1_col,a_list_as_col_of_sorted_results_wo_their_id,job_function):
	# a col of the resulting tuple of the job_function
	batch_run_col_list = []
	# loop on folds (produce sequentially the results)
	for a_fold_tools_list in folds_of_IL1_col: # testing a_fold_tools_list = folds_of_IL1_col[0]
		resulting_tuple = job_function(a_fold_tools_list)
		batch_run_col_list.append(resulting_tuple)
		# supply info on evolution of the batch runs
		if len(batch_run_col_list) == len(folds_of_IL1_col):
			print(len(batch_run_col_list), "results caught out of the", len(folds_of_IL1_col), "results expected :")
			print("All IL1 results have been caught!")
		else:
			print(len(batch_run_col_list), "results caught out of the", len(folds_of_IL1_col), "results expected. Expecting additional content...")
		print("A task has ended.")
	# collect the results in order
	print("************collecting the results************")
	bench_of_results_list_of_tuples_sorted_on_task_id = sorted(batch_run_col_list, key=itemgetter(0))
	for a_tuple in bench_of_results_list_of_tuples_sorted_on_task_id:
		a_list_as_col_of_sorted_results_wo_their_id.append(a_tuple[1])
	print("************finished collecting the results************")


###########################################################################
###########################################################################
# new Stratified CV
def stratKfolds_making(num_folds,aseed,dframe,index_starting_fts_cols,Resp_col_name,describe):
	# keeping the old index in a dict
	dframe_real_index = dframe.index
	# keep in a dict mimes of a eventual new index as keys and real indexes as values to get them later
	dict_mime_and_real_indexes = {}
	for i in range(len(dframe_real_index)):
		dict_mime_and_real_indexes[i] = dframe_real_index[i]
	# print("The temporary index will be during folds making :") ## test
	# print(dict_mime_and_real_indexes.keys())
	# print("...while the real index was : ")
	# print(dict_mime_and_real_indexes.values())

	# np arrays are made to use it for folds making but it will use rows numbers so it seems as a new index
	# creating the array needed by the sklearn function
	dframe_x = dframe[list(dframe)[index_starting_fts_cols:]]
	arr_dframe_x = np.array(dframe_x)
	dframe_y = dframe.loc[:, [Resp_col_name]]
	arr_dframe_y = np.array(dframe_y)
	# before the function from sklearn is used, check if the number of folds asked is not exceeding the numbers of samples. if so force it to be num samples instead
	if num_folds > dframe.shape[0]:
		num_folds = dframe.shape[0]
	# the from sklearn function is used
	skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=aseed)
	print("This following cross-validation has been done : ")
	print(skf)
	# initilisations
	fold_id = 0
	folds_list_col = [] # a list to host for each fold a list [id_fold,list_tr_ind,list_tst_ind]

	for train_index, test_index in skf.split(arr_dframe_x, arr_dframe_y):

		one_fold_tools_list = []
		# 1
		fold_id += 1
		one_fold_tools_list.append(fold_id)
		# 2
		train_index_as_list = list(train_index)
		# lets correct train_index
		for i in range(len(train_index_as_list)):
			if train_index_as_list[i] != dict_mime_and_real_indexes[train_index_as_list[i]]:
				train_index_as_list[i] = dict_mime_and_real_indexes[train_index_as_list[i]]
		one_fold_tools_list.append(train_index_as_list)
		# 3
		test_index_as_list = list(test_index)
		# lets correct test_index
		for i in range(len(test_index_as_list)):
			if test_index_as_list[i] != dict_mime_and_real_indexes[test_index_as_list[i]]:
				test_index_as_list[i] = dict_mime_and_real_indexes[test_index_as_list[i]]
		one_fold_tools_list.append(test_index_as_list)
		if describe == True:
			print("The fold :", one_fold_tools_list[0])
			print("- training set indexes", one_fold_tools_list[1])
			print("- testing set indexes", one_fold_tools_list[2])
		folds_list_col.append(one_fold_tools_list)
	return folds_list_col


###########################################################################
###########################################################################


#=====================================================functions to stratify the folds in a cross-validation (uncomment to use)
# #-------------------------------streatgy to select the indexes of the x samples in the test set and y samples in the training set
# # knowing the size we want for our test set and the size of our dataframe, we can compute indexes in the dataframe to select for the testset
# # those idexes have to be at most possible equal distance from each other, not precisely but only to ensure that they are quite distributed in the data
# # so the basics of such a computation is " number of spaces between trees is number of trees minus 1"
# # we can't use the whole 0 to whatever last index space otherwise when the space value is rounded and multiplied in a suire pattern we can arrive to the last without having enough intervals
# # solution : set a start and an end, then calculate x spaces between them
# # our boundaries are : start (1), end (length dframe-1).
# # so the space to add each time from second to before-the-last step is = last step-first step /trees-1 => (length dframe-1)-1/trees-1
# #------------------------------------creating a spaceur
# def testset_indexes_spacer(dframe,testset_size):
# 	last_index = len(dframe) - 1
# 	first_index = 1
# 	num_spaces = testset_size - 1 # trees - 1
# 	tsi_spacer = float(last_index - first_index) / num_spaces
# 	return tsi_spacer
# #-------------------------------using the spaceur to elect the test set indexes
# def testset_indexes_selector(dframe, testset_size, tsi_spacer):
# 	selector1 = []
# 	first_index = 1
# 	selector1.append(first_index)
# 	num_spaces = testset_size - 1
# 	for i in list(range(1,num_spaces)):
# 		# int(np.ceil(valuex) means round valuex to sup value if possible and make and integer of that
# 		selector1.append(int(np.ceil(tsi_spacer * i)))
# 		# print i # for testing
# 	last_index = len(dframe) - 2
# 	selector1.append(last_index)
# 	# print selector1 # for testing
# 	return selector1
# #-------------------------------deduce it from the whole indexes of the dataframe to elect the training set indexes
# def trainingset_indexes_selector(dframe, selector1):
# 	# enlever l'index des samples du test de tous les index du dataframe to elect indexes of train set
# 	selector2 = set(list(range(len(dframe)))) - set(selector1)
# 	# soprt the resulting list to have them in order like a proper index
# 	selector2 = sorted(selector2)
# 	return selector2
# #-------------------------------using created indexes to elect the train and test set
# def set_creator_w_rows_index(dframe, selector):
# 	# df.iloc[set,:] means all columns but only rows with indexes in set
# 	elected_set = dframe.iloc[selector, :]
# 	# remaking the order or indexes
# 	elected_set = elected_set.reset_index(drop="True")
# 	return elected_set

# def stratification1(n_splits_in_cv, resp_col_name, cv_frame):
# 	# Takes 3 args:
# 	## - n_splits_in_cv : the number of splits in the cv experience (eg 5 for 5XCV)
# 	## - resp_col_name : the name of the response column in the frame to cross-validate on (e.g. "LOG_IC50")
# 	## - cv_frame : a matrix with observations in rows, columns as features, to cross validate on
#
# 	# lets initialize the element to return : a list of n_splits_in_cv sets to return with each set containing 2 list :
# 	# one of indexes of to use as training set of the fold and one for idexes to use as test set of the fold
# 	mygenerator = []
#
# 	n_samples = len(cv_frame)
# 	mysorted = cv_frame.sort_values(resp_col_name, axis=0, ascending=True, inplace=False, kind='mergesort') # sort following the values of the column given as response
# 	# indices = range(len(mysorted)) # range of 0 to length of new sorted table # a range start with  0
#
# 	# Get fold lengths
# 	n_splits = n_splits_in_cv
# 	fold_sizes = (n_samples // n_splits) * np.ones(n_splits, dtype=np.int)  # Equal partitioning in an array of one row each cells having size of a fold until number of folds
# 	# now potentially there is a rest on the division that are samples not enough to make a fold in the tail of the df. lets distribute that number to include them in the folds
# 	fold_sizes[:n_samples % n_splits] += 1  # increase by one until all remaining data samples are filled in (from the start add 1 in each cell of fold size until the rest is entirely distributed
#
# 	lister = []
# 	for fold_size in fold_sizes:  # Create the folds # for each fold of specific size (because they can be different), replace it by a list of same number of zeros
# 		#  with zeroes
# 		lister.append([0] * fold_size)
#
# 	for arun in list(range(min(fold_sizes))): # loop on the range 0 to smallest fold size-1 (places in the initialized with zero in folds list
# 		bucket = np.random.choice(list(range(0, n_splits)), n_splits, replace=False) # create a bucket of indexes for each one of the places arun ## range (0,j) is same as range(j)
#
# 		for element in list(range(len(bucket))):
# 			lister[element][arun] = bucket[element] + arun * n_splits # put fold_size * the column index to get same index in each cell. That get the same indexes everywhere, to stratify add to each column, the same vactor but that is changing
# 	# the previous deals with the fold in minimal size but the fold with more than the minimal size are not filled with an index, lets do it
# 	# Always add to the end, starting from the first
# 	# Is always the last item anyway or min fold size would just go higher
# 	mydifference = len(mysorted) - n_splits * min(fold_sizes) # calculate a controler of all samples are distributed or not (a diff exists between the initial df and the one with # of fold * # of splits)
#
# 	if mydifference > 0: # condittion to fill additionnal cells for indexes higher than min_fold_size*n_splits
# 		bucket = np.random.choice(list(range(0, mydifference)), mydifference, replace=False)  # the bucket contained the remaining to disparatly add on the tailing folds
# 		for something in list(range(len(bucket))):
# 			lister[something][-1] = bucket[something] + min(fold_sizes) * n_splits      # give by default min_fold_size*n_splits add disparatly add the values in the bucket
# 	else:
# 		pass     # in the case of a df with min folds does it all, do nothing
#
# 	listerset = set(tuple(sublist) for sublist in lister) # a set of all sublists in lister #will serve as all_indexes_groups to deduce the indexes of test_group from all_indexes_group
#
# 	for thingy in itertools.combinations(lister, n_splits - 1):  # Itertools will shuffle! Lister lengths: first no longer longest. # producing of the 4 out 5 groups used for training. idea is also to produces 5 groups that lacks a group out of the 5 and that group changes (the idea of the test fold in a CV)
# 		# print "here is one thingy"
# 		# print thingy
# 		# print "here is on sthingy"
# 		setthingy = set(tuple(subsection) for subsection in thingy)  # We need a set for the most efficient difference call
# 		# print setthingy
# 		# lets choose a group for being a test_indexes in a fold
# 		thingy = sorted([item for subl in thingy for item in subl])  # Flatten & sort training set # unpack thingy and sort it  ## two for loop like it is done vertically are used after a value to unpack it from list if list
# 		# print "here is flattened and sorted thingy"
# 		# print thingy
# 		# lets choose a group for being a training_indexes in a fold
# 		diff = sorted([myitem for subset in listerset.difference(setthingy) for myitem in subset])  # Set difference (test) + flatten # a list of the difference unpacked from their list to a list
# 		# print "here is what must be the group put aside but that exist in lister (already in flattened and sorted form")
# 		# print diff
# 		# print "here is what will go into the generator subsequently"
# 		# print thingy
# 		# print diff
# 		# print "here is what will go into the generator as a whole"
# 		# print ((thingy, diff))
# 		mygenerator.append((thingy, diff))  # Created generator object # lets add to it a tuple of 2 lists : one larger with the training_indexes and one shorter with the test_indexes, in that order
# 	return mygenerator
#========================================================(uncomment to use)

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< classification used functions  (uncomment to use)

