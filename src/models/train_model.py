from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def get_interim_data():
    data_full_path = '../../data/interim/Iris.csv'
    interim_data = pd.read_csv(data_full_path)
    interim_data.drop('Id', axis=1, inplace=True)
    return interim_data


def encode_label(data):
    data['Species'] = LabelEncoder().fit_transform(data['Species'])


def build_pipeline_for_single_classifier(x, y, classifier = LogisticRegression()):
    pipeline = Pipeline([
        ('normalizer', StandardScaler()),
        ('clf', classifier)
    ])
    scores = cross_validate(pipeline,X_train,y_train)
    return scores,pipeline

def build_and_assess_classifiers(x,y,pipeline):
    clfs = [LogisticRegression(), SVC(), KNeighborsClassifier(n_neighbors=3), DecisionTreeClassifier(),
            RandomForestClassifier(), GradientBoostingClassifier()]


    for classifier in clfs:
        pipeline.set_params(clf=classifier)
        clf_scores = cross_validate(pipeline, x, y)
        print('-------------------------------------------------------------------------------')
        print(str(classifier))
        print('-------------------------------------------------------------------------------')
        print(str(classifier.get_params()))
        print('-------------------------------------------------------------------------------')
        for key, values in clf_scores.items():
            print(key, ' mean ', values.mean())
            print(key, ' std ', values.std())

def set_parameters_for_hyper_parameter_tuning(pipeline,classifier = 'RandomForest'):
    if classifier == 'RandomForest':
        pipeline.set_params(clf=RandomForestClassifier())
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=400, num=10)]
        # Number of features to consider at every split
        max_features = ['log2', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the grid
        grid_search_param = {'clf__n_estimators': n_estimators,
                       'clf__max_features': max_features,
                       'clf__max_depth': max_depth,
                       'clf__min_samples_split': min_samples_split,
                       'clf__min_samples_leaf': min_samples_leaf,
                       'clf__bootstrap': bootstrap}
    elif classifier == 'DecisionTrees':
        pipeline.set_params(clf=DecisionTreeClassifier())
        # Number of features to consider at every split
        max_features = ['log2', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Create the grid
        grid_search_param = {'clf__max_features': max_features,
                             'clf__max_depth': max_depth,
                             'clf__min_samples_split': min_samples_split,
                             'clf__min_samples_leaf': min_samples_leaf}

    return grid_search_param

def perform_hyper_parameter_tuning(pipeline, grid_search_param, type = 'GridSearch'):
    if type == 'GridSearch':
        cv_grid = GridSearchCV(pipeline, param_grid=grid_search_param)
    elif type == 'RandomSearch':
        cv_grid = RandomizedSearchCV(pipeline, param_distributions=grid_search_param,
                                     n_iter=100, cv=5, verbose=2, random_state=20, n_jobs=-1)

    cv_grid.fit(X_train, y_train)
    print('CV Best params are: ')
    print(cv_grid.best_params_)

    print()
    print('CV Best estmiator is: ')
    print(cv_grid.best_estimator_)

    print()
    print("CV Best score is: ")
    print(cv_grid.best_score_)

    y_predict = cv_grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    print("Accuracy of the best classifier after CV is %.3f%%" % (accuracy * 100))



data = get_interim_data()
encode_label(data)
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1].values, data['Species'], test_size=0.3,
                                                    random_state=38)
scores,pipeline = build_pipeline_for_single_classifier(x = X_train,y = y_train)
print(scores['test_score'].mean())

#build_and_assess_classifiers(X_train,y_train,pipeline)

### Random forest classifier performs best and we now proceed with hyper parameter tuning


grid = set_parameters_for_hyper_parameter_tuning(pipeline,classifier='DecisionTrees')

perform_hyper_parameter_tuning(pipeline,grid,'RandomSearch')



