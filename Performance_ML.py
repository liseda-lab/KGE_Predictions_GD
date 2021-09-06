# -*- coding: utf-8 -*-
'''
    File name: Performance_ML.py
    Authors: Susana Nunes, Rita T. Sousa, Catia Pesquita
    Python Version: 3.7
'''
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


################################################
##             FILE MANIPULATION              ##
################################################

# Read the data file with a specific operation
def process_operations_file(file_operation_path):
    dataset = open(file_operation_path, 'r')
    data = dataset.readlines()
    labels = []
    list_labels = []
    pairs = []
    embs = []
    final_kg = []
    for line in data[1:]:
        gene, disease, embedding, label = line.strip(";\n").split(";")
        embs.append(embedding)
        labels.append(int(label))
        list_labels.append([(gene, disease), int(label)])
        pairs.append((gene, disease))

    for emb in embs:  # divide the string of values several separate values
        emb = emb.split(',')
        emb = [float(value) for value in emb]
        final_kg.append(emb)

    dataset.close()
    return list_labels, pairs, labels, final_kg


################################################
##         TEST AND TRAIN CREATION            ##
################################################

# Creates X train/test e Y train/test:
def create_X_and_Y_lists(file_operation_path, file_indexes_path):
    index_file = open(file_indexes_path, 'r')
    indexes = index_file.readlines()
    list_labels, pairs, labels, KG = process_operations_file(file_operation_path)
    indexesList = []

    for index in indexes:
        ind = index.strip("\n")
        indexesList.append(int(ind))
    final_Xlist = [KG[index] for index in indexesList]
    final_Ylist = [labels[index] for index in indexesList]

    index_file.close()
    return final_Xlist, final_Ylist


################################################
##                PREDICTIONS                 ##
################################################


def predictions(predicted_labels, list_labels, predictions_prob_test):
    print('....................... DOING PREDICTIONS .......................')
    probs = []
    for probability in predictions_prob_test:
        prob = np.ndarray.tolist(probability)
        probs.append(prob[1])

    waf = metrics.f1_score(list_labels, predicted_labels, average='weighted')
    fmeasure_noninteract, fmeasure_interact = metrics.f1_score(list_labels, predicted_labels, average=None)
    precision = metrics.precision_score(list_labels, predicted_labels)
    recall = metrics.recall_score(list_labels, predicted_labels)
    accuracy = metrics.accuracy_score(list_labels, predicted_labels)
    auc = metrics.roc_auc_score(list_labels, probs, average='weighted')

    return waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy, auc


def writePredictions(predictions, ytest_or_train, path_output):
    print('....................... WRITING PREDICTIONS .......................')
    file_predictions = open(path_output, 'w')
    file_predictions.write('Predicted_output' + '\t' + 'Expected_Output' + '\n')

    for i in range(len(ytest_or_train)):
        file_predictions.write(str(predictions[i]) + '\t' + str(ytest_or_train[i]) + '\n')

    file_predictions.close()


################################################
##             ML ALGORITHMS                  ##
################################################

def performance_XGBoost(X_train, X_test, y_train, y_test, path_output_predictions):
    print('....................... XGBOOST INICIALIZED .......................')
    probs = []
    xgb_model = xgb.XGBClassifier()

    clf = GridSearchCV(xgb_model,
                       {'max_depth': [2, 4, 6],
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.1, 0.01, 0.001]}, verbose=1)

    clf.fit(np.array(X_train), np.array(y_train))

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    parameters = clf.best_params_

    predictions_test = clf.predict(np.array(X_test))
    predictions_train = clf.predict(np.array(X_train))
    predictions_prob_test = clf.predict_proba(np.array(X_test))
    predictions_prob_train = clf.predict_proba(np.array(X_train))
    for probability in predictions_prob_train:
        prob = np.ndarray.tolist(probability)
        probs.append(prob[1])
    for probability in predictions_prob_test:
        prob = np.ndarray.tolist(probability)
        probs.append(prob[1])

    y = y_train + y_test

    writePredictions(predictions_train.tolist(), y_train, path_output_predictions + '_predictionsTrainSet.txt')
    writePredictions(predictions_test.tolist(), y_test, path_output_predictions + '_predictionsTestSet.txt')
    writePredictions(probs, y, path_output_predictions + '_predictions_prob.txt')

    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy, auc = predictions(predictions_test,
                                                                                                 y_test,
                                                                                                 predictions_prob_test)

    print('....................... XGBOOST FINALIZED .......................')
    return waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy, auc, parameters


def performance_RandomForest(X_train, X_test, y_train, y_test, path_output_predictions):
    print('....................... RANDOM FOREST INICIALIZED .......................')
    probs = []
    rf_model = RandomForestClassifier()

    regressor = GridSearchCV(rf_model,
                             {'max_depth': [2, 4, 6, None],
                              'n_estimators': [50, 100, 200]})

    regressor.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(regressor.best_params_)
    parameters = regressor.best_params_

    predictions_test = regressor.predict(X_test)
    predictions_train = regressor.predict(X_train)
    predictions_prob_test = regressor.predict_proba(np.array(X_test))
    predictions_prob_train = regressor.predict_proba(np.array(X_train))
    for probability in predictions_prob_train:
        prob = np.ndarray.tolist(probability)
        probs.append(prob[1])

    for probability in predictions_prob_test:
        prob = np.ndarray.tolist(probability)
        probs.append(prob[1])

    y = y_train + y_test

    writePredictions(predictions_train, y_train, path_output_predictions + '_predictionsTrainSet.txt')
    writePredictions(predictions_test, y_test, path_output_predictions + '_predictionsTestSet.txt')
    writePredictions(probs, y, path_output_predictions + '_predictions_prob.txt')

    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy, auc = predictions(predictions_test,
                                                                                                 y_test,
                                                                                                 predictions_prob_test)
    print('....................... RANDOM FOREST FINALIZED .......................')
    return waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy, auc, parameters


##############################################
##              Run Performance             ##
##############################################

operations_list = [('Data_Concatenated.csv', 'Concatenation'), ('Data_Average.csv', 'Average'),
                   ('Data_Weighted_L1.csv', 'Weighted-L1'), ('Data_Weighted_L2.csv', 'Weighted-L2')]
# RandomForest
for operation in operations_list:  # Embedding Operation
    Xtrain, Ytrain = create_X_and_Y_lists(operation[0], 'Indexes_split/PairsIndexes__splitTrain.txt')
    Xtest, Ytest = create_X_and_Y_lists(operation[0], 'Indexes_split/PairsIndexes__splitTest.txt')
    path_output_predictions = 'RandomForest_Run_' + str(operation[1])

    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, \
        accuracy, auc, parameters = performance_RandomForest(Xtrain, Xtest, Ytrain, Ytest, path_output_predictions)

    file_xg = open('Performance_RandomForest' + str(operation[1]) + '.txt', 'a')
    file_xg.write('WAF: ' + str(waf) + '\n')
    file_xg.write('fmeasure noninteract: ' + str(fmeasure_noninteract) + '\n')
    file_xg.write('fmeasure interact: ' + str(fmeasure_interact) + '\n')
    file_xg.write('Precision: ' + str(precision) + '\n')
    file_xg.write('Recall: ' + str(recall) + '\n')
    file_xg.write('Accuracy: ' + str(accuracy) + '\n')
    file_xg.write('Auc: ' + str(auc) + '\n')
    file_xg.write('Best parameters set found on development set:' + str(parameters) + '\n')
    file_xg.close()

# XGBoost
for operation in operations_list:  # Embedding Operation
    Xtrain, Ytrain = create_X_and_Y_lists(operation[0], 'Indexes_split/PairsIndexes__splitTrain.txt')
    Xtest, Ytest = create_X_and_Y_lists(operation[0], 'Indexes_split/PairsIndexes__splitTest.txt')
    path_output_predictions = 'XGBoost_' + str(operation[1])

    waf, fmeasure_noninteract, fmeasure_interact, precision, \
        recall, accuracy, auc, parameters = performance_XGBoost(Xtrain, Xtest, Ytrain, Ytest, path_output_predictions)

    file_xg = open('Performance_XGboost' + str(operation[1]) + '.txt', 'a')
    file_xg.write('WAF: ' + str(waf) + '\n')
    file_xg.write('fmeasure noninteract: ' + str(fmeasure_noninteract) + '\n')
    file_xg.write('fmeasure interact: ' + str(fmeasure_interact) + '\n')
    file_xg.write('Precision: ' + str(precision) + '\n')
    file_xg.write('Recall: ' + str(recall) + '\n')
    file_xg.write('Accuracy: ' + str(accuracy) + '\n')
    file_xg.write('Auc: ' + str(auc) + '\n')
    file_xg.write('Best parameters set found on development set:' + str(parameters) + '\n')
    file_xg.close()

#NayveBayes
for operation in operations_list:  # Embedding Operation
    Xtrain, Ytrain = create_X_and_Y_lists(operation[0], 'Indexes_split/PairsIndexes__splitTrain.txt')
    Xtest, Ytest = create_X_and_Y_lists(operation[0], 'Indexes_split/PairsIndexes__splitTest.txt')
    path_output_predictions = 'NayvesBayes_Run_' + str(operation[1])

    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, \
        accuracy = performance_NayveBayes(Xtrain, Xtest, Ytrain, Ytest, path_output_predictions)

    file_xg = open('Performance_NayveBayes' + str(operation[1]) + '.txt', 'a')
    file_xg.write('WAF: ' + str(waf) + '\n')
    file_xg.write('fmeasure noninteract: ' + str(fmeasure_noninteract) + '\n')
    file_xg.write('fmeasure interact: ' + str(fmeasure_interact) + '\n')
    file_xg.write('Precision: ' + str(precision) + '\n')
    file_xg.write('Recall: ' + str(recall) + '\n')
    file_xg.write('Accuracy: ' + str(accuracy) + '\n')
    file_xg.close()


# Multi-Layer Perceptron
for operation in operations_list:  # Embedding Operation
    Xtrain, Ytrain = create_X_and_Y_lists(operation[0], 'Indexes_split/PairsIndexes__splitTrain.txt')
    Xtest, Ytest = create_X_and_Y_lists(operation[0], 'Indexes_split/PairsIndexes__splitTest.txt')
    path_output_predictions = 'MLP_Run_' + str(operation[1])

    waf, fmeasure_noninteract, fmeasure_interact, \
        precision, recall, accuracy, parameters = performance_MLP(Xtrain, Xtest, Ytrain, Ytest, path_output_predictions)

    file_xg = open('Performance_MLP' + str(operation[1]) + '.txt', 'a')
    file_xg.write('WAF: ' + str(waf) + '\n')
    file_xg.write('fmeasure noninteract: ' + str(fmeasure_noninteract) + '\n')
    file_xg.write('fmeasure interact: ' + str(fmeasure_interact) + '\n')
    file_xg.write('Precision: ' + str(precision) + '\n')
    file_xg.write('Recall: ' + str(recall) + '\n')
    file_xg.write('Accuracy: ' + str(accuracy) + '\n')
    file_xg.write('Best parameters set found on development set:' + str(parameters) + '\n')
    file_xg.close()
