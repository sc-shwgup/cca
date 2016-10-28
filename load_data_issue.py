import glob
import numpy
import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter
import pandas
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


def load_data_issue_and_labels():

    seed = 7
    numpy.random.seed(seed)
    
 

    # load dataset
    df = pandas.read_csv("./Consumer_Complaints.csv")
    df = pandas.DataFrame(df)
    df = df[['Consumer complaint narrative','Issue']]
    non_empty_complaints = df[pandas.notnull(df['Consumer complaint narrative'])]
    non_empty_issue = non_empty_complaints[pandas.notnull(non_empty_complaints['Issue'])]
    Y = non_empty_issue['Issue'].astype('str')
    X = non_empty_issue['Consumer complaint narrative'].astype('str')

    unique_list = list(non_empty_issue.Issue.unique())
    
    balanced_issue = pandas.DataFrame()

    for val in unique_list:
        df_temp=[]
        df_unique = []
        df_temp = non_empty_issue[non_empty_issue['Issue']==val]
        if len(df_temp)>1000:
            df_temp = pd.concat([df_temp]*12,ignore_index = True)
            df_unique = df_temp.drop_duplicates()
            if len(df_unique)< 10000:
                df_duplicates = df_temp.sample(10000-len(df_unique))
                df_unique = pandas.concat([df_unique,df_duplicates])
            balanced_issue = pandas.concat([balanced_issue,df_unique])


    unique_list = list(balanced_issue.Issue.unique())


    df_train = pandas.DataFrame()
    df_test = pandas.DataFrame()
    df_testing = pandas.DataFrame()



    for val in unique_list:
        df_temp=[]
        df_temp = balanced_issue[balanced_issue['Issue']==val]
        train_end = int(0.6 * len(df_temp))
        dev_end = train_end + int(0.2 * len(df_temp))
        df_train = df_train.append(df_temp[0:train_end])
        df_random_train = df_train.iloc[np.random.permutation(len(df_train))]
        df_test = df_test.append(df_temp[train_end:dev_end])
        df_random_test= df_test.iloc[np.random.permutation(len(df_test))]
        df_testing = df_testing.append(df_temp[dev_end:])
    X_train = df_random_train['Consumer complaint narrative'].astype('str')
    Y_train = df_random_train['Issue'].astype('str')
    X_test = df_random_test['Consumer complaint narrative'].astype('str')
    Y_test = df_random_test['Issue'].astype('str')

    encoder = LabelEncoder()
    encoder.fit(Y_train)
    encoded_Y = encoder.transform(Y_train)
    n_values = numpy.max(encoded_Y) + 1
    dummy_y_train = numpy.eye(n_values)[encoded_Y]


    
     # encode test class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y_test)
    encoded_Y = encoder.transform(Y_test)
    ll = encoder.inverse_transform([i for i in range(0,len(unique_list))])
    # convert integers to dummy variables (i.e. one hot encoded)
    n_values = numpy.max(encoded_Y) + 1
    dummy_y_test = numpy.eye(n_values)[encoded_Y]
    df_testing.to_csv("./issueTest.csv")
    X = pandas.concat([X_train,X_test]).astype('str')
    encoder = dict(zip(ll,[i for i in range(0,len(unique_list))]))
    num_classes = len(unique_list)
    return [X,X_train,X_test,dummy_y_train,dummy_y_test,num_classes,encoder]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

                            
