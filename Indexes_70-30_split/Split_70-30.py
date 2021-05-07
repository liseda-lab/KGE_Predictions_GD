# -*- coding: utf-8 -*-
'''
    File name: Split_70-30.txt
    Authors: Susana Nunes, Rita T. Sousa, Catia Pesquita
    Python Version: 3.7
'''
import numpy as np
from sklearn import metrics
from operator import itemgetter
from sklearn.model_selection import StratifiedShuffleSplit


################################################
##        DATASET FILE MANIPULATION           ##
################################################

# CSV file reading function
def process_dataset_file(file_dataset_path):
    dataset = open(file_dataset_path, 'r')
    data = dataset.readlines()
    labels = []
    labels_list = []
    pairs = []

    for line in data[1:]:
        nr, gene, disease, label = line.strip(";\n").split(";")
        labels.append(int(label))
        labels_list.append([(gene, disease), int(label)])
        pairs.append((gene, disease))

    dataset.close()
    return labels_list, pairs, labels


####################################
##         70%-30% SPLIT          ##
####################################

# Function that creates partitions: 10 datasets with positions
def run_partition(file_dataset_path, filename_output):
    labels_list, pairs, labels = process_dataset_file(file_dataset_path)
    index_partition = 1

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)

    for indexes_partition_train, indexes_partition_test in sss.split(pairs, labels):
        file_crossValidation_train = open(
            filename_output + 'Indexes__crossvalidationTrain' + str(index_partition) + '.txt', 'w')
        file_crossValidation_test = open(
            filename_output + 'Indexes__crossvalidationTest' + str(index_partition) + '.txt', 'w')
        for index in indexes_partition_train:
            file_crossValidation_train.write(str(index) + '\n')
        for index in indexes_partition_test:
            file_crossValidation_test.write(str(index) + '\n')
        file_crossValidation_train.close()
        file_crossValidation_test.close()

        index_partition = index_partition + 1

file_dataset_path='Dataset_Pairs_Label.csv'
run_partition(file_dataset_path, 'Pairs_')

