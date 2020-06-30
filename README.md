# LT2212 V20 Assignment 3


PART 1
Regarding tokenization, I lowercased all words and filtered out the numerical values and punctuation leaving only alphabetical characters. 
To call a3_features.py, you need to pass the following arguments:
1)	The folder to where the text files are: this can either be a full path(e.g. /home/gusmavko@GU.GU.SE/lt2212-v20-a3-master/enron_sample) or just the folder name if the script and the data are in the same directory(enron_sample).
2)	The name of the output file where you store the feature table(e.g. output_file.csv).
3)	The number of dimensions you wish your feature table to have(e.g. 100).
4)	The number you want the test set to be. This is an optional argument, so the default value is set to 20, but you can change it by adding it as an argument the following way: --test 20.
Here is an example you can use to run script a3_features.py:
Python3 a3_features.py /home/gusmavko@GU.GU.SE/lt2212-v20-a3-master/enron_sample output_file.csv 100 -–test 50

