import pandas as pd
import os
from sklearn.model_selection import train_test_split


def get_interim_data():
    data_full_path = '../../data/interim/Iris.csv'
    data = pd.read_csv(data_full_path)
    return data


def rename_response_variable(data):
    map_name = {'Iris-setosa': 'Sertosa', 'Iris-versicolor': 'Versicolor', 'Iris-virginica': 'Virginica'}
    unique_val_set = set(data['Species'].unique())
    for k in map_name.keys():
        if k not in unique_val_set:
            print('Names not found in response values. Returning without renaming')
            return

    data.replace({'Species': map_name}, inplace=True)
    print('Names successfully renamed using map function')


def create_features_subset(data):
    subset_features = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'Species']]
    save_processed_data(subset_features, 'Iris_subset.csv')
    return subset_features


##Build features by transforming the data for sepal length and sepal width

def save_processed_data(data, filename):
    data_full_path = '../../data/processed/' + filename
    if os.path.isfile(data_full_path):
        print('Processed data found for ' + filename + '... not saving copy')
    else:
        print('Processed data not found for ' + filename + '... creating a copy and saving at processed location')
        base_name = os.getcwd()
        os.chdir('../../data/processed')
        data.to_csv(filename, index=False)
        os.chdir(base_name)


def create_train_test_split(data, test_size=0.2, save_data=False, train_filename=None, test_filename=None):
    train, test = train_test_split(data, test_size=test_size)
    if save_data:
        save_processed_data(train, str(train_filename))
        save_processed_data(test, str(test_filename))


if __name__ == '__main__':
    data = get_interim_data()
    rename_response_variable(data)
    subset_data = create_features_subset(data)
    #create_train_test_split(subset_data,0.25,True,'train_subset.csv','test_subset.csv')
    data.shape
