# -*- coding: utf-8 -*-
'''
    File name: Median-Calculation.py
    Authors: Susana Nunes, Rita T. Sousa, Catia Pesquita
    Python Version: 3.7
'''
from statistics import median


################################################
##        Performance FILE MANIPULATION       ##
################################################

# CSV file reading function
def process_dataset_file(file_performance_path, file_output):
    performance = open(file_performance_path, 'r')
    data = performance.readlines()
    waf_median = []
    precision_median = []
    recall_median = []
    for line in data:
        if 'WAF' in line:
            wafs = line.replace('WAF: ', '').strip()
            waf_median.append(float(wafs))
        if 'Precision' in line:
            precisions = line.replace('Precision: ', '').strip()
            precision_median.append(float(precisions))
        if 'Recall' in line:
            recalls = line.replace('Recall: ', '').strip()
            recall_median.append(float(recalls))

    numbers = True

    while numbers:
        if len(waf_median) == 10:
            waf = median(waf_median)
            print('WAF:', waf)
        else:
            print('ERROR:wrong number of waf values')
            numbers = False

        if len(precision_median) == 10:
            precision = median(precision_median)
            print('precision:', precision)
        else:
            print('ERROR:wrong number of precision values')
            numbers = False

        if len(recall_median) == 10:
            recall = median(recall_median)
            print('recall:', recall)

        else:
            print('ERROR:wrong number of recall values')
            numbers = False

        file_results = open(file_output, 'w')
        file_results.write('Median WAF: ' + str(waf).replace('.', ',') + '\n')
        file_results.write('Median Precision: ' + str(precision).replace('.', ',') + '\n')
        file_results.write('Median Recall: ' + str(recall).replace('.', ','))
        file_results.close()

    performance.close()
    return waf, precision, recall


################################################
##             RUN METRICS                    ##
################################################

## EXAMPLE OF FILE WE CAN CALCULATE THE MEDIAN
file_performance_path= 'Performance_RandomForestHadamard.txt'
file_output = 'Median_Results_RandomForestHadamard.txt'
process_dataset_file(file_performance_path, file_output)
