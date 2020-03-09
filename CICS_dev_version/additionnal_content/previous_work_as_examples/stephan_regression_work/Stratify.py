import numpy as np
import itertools
import locale
# ---------------------Variables to initialise------------------------------------------
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') #for setting the characters format
# ---------------------------------------------------------------------------------------------------
# -------------------------------------- Function Definitions ---------------------------------------
# ---------------------------------------------------------------------------------------------------
def stratify(n_splits, response, nframe):
	# As input, takes a matrix with observations in rows, columns as features
	# One feature must be listed as the response variable (e.g. LOG_IC50)
	mygenerator = []
	n_samples = len(nframe)
	mysorted = nframe.sort_values(response, axis=0, ascending=True, inplace=False, kind='mergesort') # sort following the values of the column given as response
	indices = range(len(mysorted)) # range of 0 to length of new sorted table # a range start with  0

	# Get fold lengths
	n_splits = n_splits
	fold_sizes = (n_samples // n_splits) * np.ones(n_splits, dtype=np.int)  # Equal partitioning in an array of one row each cells having size of a fold until number of folds
    # now potentially there is a rest on the division that are samples not enough to make a fold in the tail of the df. lets distribute that number to include them in the folds
	fold_sizes[:n_samples % n_splits] += 1  # increase by one until all remaining data samples are filled in (from the start add 1 in each cell of fold size until the rest is entirely distributed
	current = 0  # not needed

	arun = 0 # not needed
	lister = []
	for fold_size in fold_sizes:  # Create the folds # for each fold of specific size (because they can be different), replace it by a list of same number of zeros
		#  with zeroes
		lister.append([0] * fold_size)

	for arun in range(min(fold_sizes)): # loop on the range 0 to smallest fold size-1 (places in the initialized with zero in folds list
		bucket = np.random.choice(range(0, n_splits), n_splits, replace=False) # create a bucket of indexes for each one of the places arun ## range (0,j) is same as range(j)

		for element in range(len(bucket)):
			lister[element][arun] = bucket[element] + (arun) * n_splits # put fold_size * the column index to get same index in each cell. That get the same indexes everywhere, to stratify add to each column, the same vactor but that is changing
    # the previous deals with the fold in minimal size but the fold with more than the minimal size are not filled with an index, lets do it
	# Always add to the end, starting from the first
	# Is always the last item anyway or min fold size would just go higher
	mydifference = len(mysorted) - n_splits * min(fold_sizes) # calculate a controler of all samples are distributed or not (a diff exists between the initial df and the one with # of fold * # of splits)

	if mydifference > 0: # condittion to fill additionnal cells for indexes higher than min_fold_size*n_splits
		bucket = np.random.choice(range(0, mydifference), mydifference, replace=False)  # the bucket contained the remaining to disparatly add on the tailing folds
		for something in range(len(bucket)):
			lister[something][-1] = bucket[something] + min(fold_sizes) * n_splits      # give by default min_fold_size*n_splits add disparatly add the values in the bucket
	else:
		pass     # in the case of a df with min folds does it all, do nothing

	listerset = set(tuple(sublist) for sublist in lister) # a set of all sublists in lister #will serve as all_indexes_groups to deduce the indexes of test_group from all_indexes_group

	for thingy in itertools.combinations(lister, n_splits - 1):  # Itertools will shuffle! Lister lengths: first no longer longest. # producing of the 4 out 5 groups used for training. idea is also to produces 5 groups that lacks a group out of the 5 and that group changes (the idea of the test fold in a CV)
        # print "here is one thingy"
        # print thingy
        # print "here is on sthingy"
		setthingy = set(tuple(subsection) for subsection in thingy)  # We need a set for the most efficient difference call
        # print setthingy
        # lets choose a group for being a test_indexes in a fold
		thingy = sorted([item for subl in thingy for item in subl])  # Flatten & sort training set # unpack thingy and sort it  ## two for loop like it is done vertically are used after a value to unpack it from list if list
        # print "here is flattened and sorted thingy"
        # print thingy
        # lets choose a group for being a training_indexes in a fold
		diff = sorted([myitem for subset in listerset.difference(setthingy) for myitem in subset])  # Set difference (test) + flatten # a list of the difference unpacked from their list to a list
        # print "here is what must be the group put aside but that exist in lister (already in flattened and sorted form")
        # print diff
        # print "here is what will go into the generator subsequently"
        # print thingy
        # print diff
        # print "here is what will go into the generator as a whole"
        # print ((thingy, diff))
		mygenerator.append((thingy, diff))  # Created generator object # lets add to it a tuple of 2 lists : one larger with the training_indexes and one shorter with the test_indexes, in that order

	return mygenerator