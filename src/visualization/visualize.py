import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_interim_data():
    data_full_path = '../../data/interim/Iris.csv'
    interim_data = pd.read_csv(data_full_path)
    interim_data.drop('Id',axis=1,inplace=True)
    return interim_data


def get_data_information(data):
    # Get number of rows and columns
    shape = data.shape
    print()
    print('Number of rows: ' + str(shape[0]) + '. \nNumber of columns: ' + str(shape[1]) + '.')

    print()
    print('Column header names are:')
    # Get column names
    for columns_headers in data.columns:
        print(columns_headers)
    print()
    print('Column data types are:')
    # get column types
    print(data.dtypes)

    print()
    print("Getting descriptive statistics for the data:")
    print(data.describe())

    print()
    # Check for any missing values present in the columns
    null_columns = data.columns[data.isnull().any()].tolist()
    if len(null_columns) == 0:
        print('No missing values present in dataframe')
    else:
        print(null_columns)

def rename_response_variable(data):
    map_name = {'Iris-setosa': 'Sertosa', 'Iris-versicolor':'Versicolor','Iris-virginica':'Virginica'}
    unique_val_set = set(data['Species'].unique())
    for k in map_name.keys():
        if k not in unique_val_set:
            print('Names not found in response values. Returning without renaming')
            return

    data.replace({'Species':map_name},inplace=True)
    print('Names successfully renamed using map function')
    assign_index_within_subgroups(data)

def assign_index_within_subgroups(data):
    # Add column to assign index for each element belonging to the different species
    data['GroupIndex'] = data.groupby('Species').cumcount() + 1

def plot_features_by_species(data):
    fig,axes = plt.subplots(nrows=2,ncols=2)
    feature_list = data.columns.tolist()
    ignore_features = ['Species','GroupIndex']
    interested_features = list(set(feature_list) - set(ignore_features))
    axes = axes.ravel()
    for feature,ax in zip(interested_features,axes):
        pivot_data = data.pivot(index='GroupIndex',columns='Species',values=feature)
        pivot_data.plot(title = feature,ax=ax,legend=False)

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center')
    #fig.subplots_adjust(hspace=1)
    plt.show()

    ######
    #From the plots of the features for each of the species
    #it can be verified that petal length and petal width are easily distinguishable
    #for each of the species and are strong features
    #sepal length and width overlap for the three species and need to be studied further



def get_feature_correlation(data):
    sns.pairplot(data, hue='Species')
    corr_df = data.corr()
    #the corr_df explains the correlation between the features and
    #it can be seen that there is a high correlation between
    #petal length and petal width. Ignoring one of the features before training the model
    return corr_df

if __name__ == '__main__':
    data = get_interim_data()
    #get_data_information(data)
    rename_response_variable(data)
    #plot_features_by_species(data)
    corr_df = get_feature_correlation(data)
    data.shape


