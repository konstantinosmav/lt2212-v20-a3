import os
import sys
import argparse
import numpy as np
import pandas as pd
import nltk
import glob
import csv
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()
    dirs = glob.glob("{}/*".format(args.inputdir))
    list_of_subdirs = [] #subdirs and files should have the same l
    list_of_files = []
    i = 0
    map_word_to_index = {}
    for subdir in dirs:
        author = subdir.split("/")[-1]
        files = glob.glob("{}/*".format(subdir))
        #list_of_subdirs.append(author)
        for file in files:
            list_of_subdirs.append(author)
            
            txt = ''
            with open(file, "r") as f:
                for line in f:
                    txt +=  line
            text = txt.lower().split()        
            words = [word for word in text if word.isalpha()]
            for w in words:
                if w not in map_word_to_index:
                    map_word_to_index[w] = i
                    i += 1
            list_of_files.append(words)    
                
                 
          
    oles = map_word_to_index.keys()
    #print(map_word_to_index)
    
    columns = len(oles)
    rows = len(list_of_files) 
    features = np.zeros((rows,columns))
    count = 0
    for each_file in list_of_files:
        #print(each_file)
        vector = np.zeros(columns)
        for w in each_file:
            idx = map_word_to_index[w]
            vector[idx] +=1
        features[count, :] = vector
        count += 1
    X = features
    svd = TruncatedSVD(n_components=args.dims, n_iter=6, random_state=42)
    dim_red = svd.fit_transform(X)
    y = list_of_subdirs 


    nsplit = args.testsize/100
    X_train, X_test, y_train, y_test = train_test_split(dim_red, y, test_size=nsplit, random_state=42, shuffle = True) 
        # build table
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    y_train = pd.DataFrame(y_train)
    y_train= y_train.assign(train_test= "train") #maybe insert is better
    y_test = pd.DataFrame(y_test)
    y_test = y_test.assign(train_test= "test")
    X_ttds = pd.concat([X_train, X_test])
    #X_ttds= X_ttds.rename(columns = {0 : "vector_representation"})
    y_ttds = pd.concat([y_train, y_test])
    y_ttds = y_ttds.rename(columns = {0 : "author"})
    
    final_table = pd.concat([X_ttds, y_ttds], axis = 1)
    
    col0= final_table.pop("author")
    col1 = final_table.pop("train_test")
    
    
    final_table.insert(0,col1.name,col1)
    final_table.insert(1, col0.name, col0)
    
    print("Writing to {}...".format(args.outputfile))
    final_table.to_csv(args.outputfile)


    print("Done!")

