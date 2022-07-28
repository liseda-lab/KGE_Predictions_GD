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

# CSV file reading function that will retrieve the WAF, Precision and Recall
def process_dataset_file(file_performance_path, file_output):
    performance = open(file_performance_path, 'r')
    data = performance.readlines()
    waf_median = []
    precision_median = []
    recall_median = []
    for line in data:
        if 'WAF' in line:
            waf = line.replace('WAF: ', '').strip()
            waf_median.append(float(waf))
        if 'Precision' in line:
            precision = line.replace('Precision: ', '').strip()
            precision_median.append(float(precision))
        if 'Recall' in line:
            recall = line.replace('Recall: ', '').strip()
            recall_median.append(float(recall))

    if len(waf_median) == 10:
        waf = median(waf_median)
    else:
        print('missing values')

    precision = median(precision_median)
    recall = median(recall_median)

    print('WAF:', waf)
    print('precision:', precision)
    print('recall:', recall)

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

