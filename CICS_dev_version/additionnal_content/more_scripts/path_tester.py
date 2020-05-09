# Part 1
from pathlib import Path
d = Path(__file__)
e = Path()
print(d)
print(e)
report = open(str(e) + "/" + "outputs" + "/" + "Output_following_" + "writing" + ".o", 'w')
# data_parent_folder = "slate_data"
# drugs_data_folder = "table_of_treatments_details"
# profile_data_folder = "datasets_to_process_folder/real_val_prof_test"
# data_profiles_path = e+"/"+data_parent_folder+"/"+profile_data_folder # the profiles data
# data_drugs_path = e+"/"+data_parent_folder+"/"+drugs_data_folder # the drugs data

# # Part 2
# import os
# import argparse #to manage the arguments of the script
# from pathlib import Path # to manage paths as into arguments
# our_args_parser = argparse.ArgumentParser(prog='ClassHD',description="Welcome in the Classification benchmark on HD data.", epilog="Thank you and adress for contributions")
# ### mandatory folders to use if arguments for input files not given
# basedir = os.getcwd() #for setting the working directory
# data_parent_folder = "slate_data" # the folder containing all data
# drugs_data_folder = "table_of_treatments_details" # all drugs info are consolidated here so mandatory folder for drugs data
# ### an interchangeable folder for the cases profiles data : uncomment one and data_path1 is modified
# profile_data_folder = "datasets_to_process_folder/real_val_prof_test" # folder for real val prof test
# # profile_data_folder = "datasets_to_process_folder/cat_val_prof_test" # folder for cat val prof test
# # profile_data_folder = "datasets_to_process_folder/odd_prof_with_real_val_test" # folder for odd prof with real val test
# # profile_data_folder = "datasets_to_process_folder/odd_prof_with_cat_val_test" # folder for odd prof with cat val test
# ### Paths needed to extract data (uncomment to activate a data_path1 for data matrix and a data_path2 for names of the drugs tested
# data_profiles_path = basedir+"/"+data_parent_folder+"/"+profile_data_folder # the profiles data
# data_drugs_path = basedir+"/"+data_parent_folder+"/"+drugs_data_folder #
# our_args_parser.add_argument("-cla_drugs_path","--Classif_drugs_folder", type=Path, default=data_drugs_path, help="(path) (default is PDX test data) for classification, path to the drugs data (in quotes, a path, starting by a folder located in cwd, ending by the name of the folder containing all profiles files to analyse)")
# our_args_parser.add_argument("-cla_profiles_path","--Classif_profiles_folder", type=Path, default=data_profiles_path, help="(path) (default is PDX test data) for classification, path to the profiles data (in quotes, a path, starting by a folder located in cwd, ending by the name of the folder containing all profiles files to analyse)")
# ############## parsing them....
# our_args = our_args_parser.parse_args()
# classif_profiles_folder = our_args.Classif_profiles_folder # classif_profiles_folder = data_profiles_path for testing
# classif_drugs_folder = our_args.Classif_drugs_folder # classif_drugs_folder = data_drugs_path for testing
# for root, directories, filenames in os.walk(classif_profiles_folder):
# 	for file in filenames:
# 		# print(os.path.join(root,file)) # for testing (display the full path of the file)
# 		print("profiles are in",file)
# for root, directories, filenames in os.walk(classif_drugs_folder):
# 	for file in filenames:
# 		# print(os.path.join(root,file)) # for testing (display the full path of the file)
# 		print("drugs are in",file)
# # "/home/diouf/ClassHD_work/actual_repo/ClassHD/CICS_dev_version/slate_data/datasets_to_process_folder/real_val_prof_test"
# # "/home/diouf/ClassHD_work/actual_repo/ClassHD/CICS_dev_version/slate_data/table_of_treatments_details"