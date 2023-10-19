import pandas as pd
import time
import pyRAPL
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

MEASUREMENT_REPETITIONS = range(3)
N_COLS = [10, 15, 20]
N_ROWS = [1000, 5000, 10000]

DEBUG = False

DATA_ANON_DIR = "/home/teamK/GreenLab/data/anonymized_training_data/"
DATA_ORIG_DIR = "/home/teamK/GreenLab/data/original_training_data/"

counter = 0

for rep in MEASUREMENT_REPETITIONS:

    for n_col in N_COLS:
        for n_row in N_ROWS:
            
            # Reconstruct the foldername
            folder = str(n_row) + "r_" + str(n_col) + "f"
            
            # Data Dirs.
            # e.g. C:\Users\kilia\Documents\GitHub\GreenLab\data\original_training_data\1000r_10f\
            dir_orig = DATA_ORIG_DIR + folder + "/"
            dir_anon = DATA_ANON_DIR + folder + "/"

            # Load original datasets
            X_orig_1 = pd.read_csv(dir_orig+"X_reduced_diabetes_binary1.csv")
            y_orig_1 = pd.read_csv(dir_orig+"y_reduced_diabetes_binary1.csv")

            X_orig_2 = pd.read_csv(dir_orig+"X_reduced_diabetes_binary2.csv")
            y_orig_2 = pd.read_csv(dir_orig+"y_reduced_diabetes_binary2.csv") 

            # Load anonymized datasets
            X_anon_1 = pd.read_csv(dir_anon+"X_reduced_diabetes_binary_anon1.csv")
            y_anon_1 = pd.read_csv(dir_anon+"y_reduced_diabetes_binary_anon1.csv")

            X_anon_2 = pd.read_csv(dir_anon+"X_reduced_diabetes_binary_anon2.csv")
            y_anon_2 = pd.read_csv(dir_anon+"y_reduced_diabetes_binary_anon2.csv")

            ############################# RF #################################
            
            #Hyperparams
            n_trees = 5000

            pyRAPL.setup()
            label = 'orig_1_RF_' + folder + "_" + str(rep) + "_"
            csv_output = pyRAPL.outputs.CSVOutput('/home/teamK/GreenLab/code/energy/'+label+".csv")
            with pyRAPL.Measurement(label, output=csv_output):
                orig1_RF = RandomForestClassifier(n_estimators=n_trees, random_state=42)
                orig1_RF.fit(X_orig_1, y_orig_1.values.ravel())
            csv_output.save()

            if DEBUG:
                counter += 1
                print("orig1_RF reached")

            pyRAPL.setup()
            label = label = 'orig_2_RF_' + folder + "_" + str(rep) + "_"
            csv_output = pyRAPL.outputs.CSVOutput('/home/teamK/GreenLab/code/energy/'+label+".csv")
            with pyRAPL.Measurement(label, output=csv_output):
                orig2_RF = RandomForestClassifier(n_estimators=n_trees, random_state=42)
                orig2_RF.fit(X_orig_2, y_orig_2.values.ravel())
            csv_output.save()

            if DEBUG:
                counter += 1
                print("orig2_RF reached")

            pyRAPL.setup()
            label = label = 'anon_1_RF_' + folder + "_" + str(rep) + "_"
            csv_output = pyRAPL.outputs.CSVOutput('/home/teamK/GreenLab/code/energy/'+label+".csv")
            with pyRAPL.Measurement(label, output=csv_output):
                anon1_RF = RandomForestClassifier(n_estimators=n_trees, random_state=42)
                anon1_RF.fit(X_anon_1, y_anon_1.values.ravel())
            csv_output.save()

            if DEBUG:
                counter += 1
                print("anon1_RF reached")

            pyRAPL.setup()
            label = label = 'anon_2_RF_' + folder + "_" + str(rep) + "_"
            csv_output = pyRAPL.outputs.CSVOutput('/home/teamK/GreenLab/code/energy/'+label+".csv")
            with pyRAPL.Measurement(label, output=csv_output):
                anon2_RF = RandomForestClassifier(n_estimators=n_trees, random_state=42)
                anon2_RF.fit(X_anon_2, y_anon_2.values.ravel())
            csv_output.save()

            if DEBUG:
                counter += 1
                print("anon2_RF reached")

            ######################### kNN ####################################
            
            #Hyperparams
            k = 75

            pyRAPL.setup()
            label = label = 'orig_1_KNN_' + folder + "_" + str(rep) + "_"
            csv_output = pyRAPL.outputs.CSVOutput('/home/teamK/GreenLab/code/energy/'+label+".csv")
            with pyRAPL.Measurement(label, output=csv_output):
                orig1_kNN = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree')
                orig1_kNN.fit(X_orig_1, y_orig_1.values.ravel())
            csv_output.save()

            if DEBUG:
                counter += 1
                print("orig1_kNN reached")

            pyRAPL.setup()
            label = label = 'orig_2_KNN_' + folder + "_" + str(rep) + "_"
            csv_output = pyRAPL.outputs.CSVOutput('/home/teamK/GreenLab/code/energy/'+label+".csv")
            with pyRAPL.Measurement(label, output=csv_output):
                orig2_kNN = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree')
                orig2_kNN.fit(X_orig_2, y_orig_2.values.ravel())
            csv_output.save()

            if DEBUG:
                counter += 1
                print("orig2_kNN reached")

            pyRAPL.setup()
            label = label = 'anon_1_KNN_' + folder + "_" + str(rep) + "_"
            csv_output = pyRAPL.outputs.CSVOutput('/home/teamK/GreenLab/code/energy/'+label+".csv")
            with pyRAPL.Measurement(label, output=csv_output):
                anon1_kNN = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree')
                anon1_kNN.fit(X_anon_1, y_anon_1.values.ravel())
            csv_output.save()

            if DEBUG:
                counter += 1
                print("anon1_kNN reached")

            pyRAPL.setup()
            label = label = 'anon_2_KNN_' + folder + "_" + str(rep) + "_"
            csv_output = pyRAPL.outputs.CSVOutput('/home/teamK/GreenLab/code/energy/'+label+".csv")
            with pyRAPL.Measurement(label, output=csv_output):
                anon2_kNN = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree')
                anon2_kNN.fit(X_anon_2, y_anon_2.values.ravel())
            csv_output.save()            

            if DEBUG:
                counter += 1
                print("anon2_kNN reached")


    print("Finished with repetition " + str(rep))

# 3 repetitions * 2 algorithms * 2 differently randomized datasets * 2 conditions (anon vs. orig) * 9 col/row combinations = 216 repetitions
if DEBUG:
    print(counter)


for rep in MEASUREMENT_REPETITIONS:
    pyRAPL.setup()
    label = label = "idle_" + str(rep) + "_"
    csv_output = pyRAPL.outputs.CSVOutput('/home/teamK/GreenLab/code/energy/'+label+".csv")
    with pyRAPL.Measurement(label, output=csv_output):
        time.sleep(30)
    csv_output.save()
