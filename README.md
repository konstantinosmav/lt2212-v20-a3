# LT2212 V20 Assignment 3


PART 1

Regarding tokenization, I lowercased all words and filtered out the numerical values and punctuation leaving only alphabetical characters. 
To call a3_features.py, you need to pass the following arguments:
1)	The folder to where the text files are: this can either be a full path(e.g. /home/gusmavko@GU.GU.SE/lt2212-v20-a3-master/enron_sample) or just the folder name if the script and the data are in the same directory(enron_sample).
2)	The name of the output file where you store the feature table(e.g. output_file.csv).
3)	The number of dimensions you wish your feature table to have(e.g. 100).
4)	The number you want the test set to be. This is an optional argument, so the default value is set to 20, but you can change it by adding it as an argument the following way: --test 20.
Here is an example you can use to run script a3_features.py:
python3 a3_features.py /home/gusmavko@GU.GU.SE/lt2212-v20-a3-master/enron_sample output_file.csv 100 -–test 50


PART 2-3

Sampling data: 
For sampling, I used .drop() to get the vector representation of the texts and .iloc() to get the train and test labels. After that, I choose randomly an index to get the vector representation of one text. Then, I also choose randomly another text that has an equal probability (50%) of having either the same (1) or a different label (0) than the first text. The same process if followed for both the training and the testing of the Neural Network. In the testing part, I used classification_report from sklearn.metrics to evaluate the model.
I used BCE as criterion since we had to do binary classification. For optimizer, I chose Adam since with Adam the performance of the learning rate is less sensitive than the learning rate with SGD.


To run a3_model.py, you need to pass the following arguments:
1)	The feature csv file that was created when running the a3_features.py file. 
2)	The size of the training data used to train the Neural Network –it is an optional argument. The default value is set to 210.
3)	The size of the testing data for the Neural Network- also an optional argument. Default value is set to 70
4)	The number of epochs that the entire data will be passed through the Neural Network. It is set to 3 by default.
5)	The size of the hidden layer. Default is 0.
6)	The name of the activation function, one can choose between either relu or tanh. 
All in all, the default settings are all kept to rather low values so that I could experiment a lot with training and testing. 
Here is an example of how to run a3_model.py:

python a3_model.py output_file.csv --trainsize 500 --testsize 90 --epoch 7 --hidden 2
 --activation tanh 


