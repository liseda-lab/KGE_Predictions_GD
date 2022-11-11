# -*- coding: utf-8 -*-
'''
    File name: MedianResults_Baseline.py
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
        if 'WAF in Test Set' in line:
            waf = line.replace('WAF in Test Set: ', '').strip()
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

    print('WAF in Test Set:', waf)
    print('Precision:', precision)
    print('Recall:', recall)

    file_results = open(file_output, 'w')
    file_results.write('Median WAF: ' + str(waf).replace('.',',') + '\n')
    file_results.write('Median Precision: ' + str(precision).replace('.', ',') + '\n')
    file_results.write('Median Recall: ' + str(recall).replace('.', ','))
    file_results.close()
    performance.close()
    return waf, precision, recall

################################################
##             RUN METRICS                    ##
################################################
file_performance_path = 'Performance_Baseline_Measure1-BMA_Seco2004.txt'
file_output ='Results_Baseline_Measure1-BMA_Seco2004.txt'
process_dataset_file(file_performance_path, file_output)

file_performance_path = 'Performance_Baseline_Measure2-BMA_Resnik1995.txt'
file_output ='Results_Baseline_Measure2-BMA_Resnik1995.txt'
process_dataset_file(file_performance_path, file_output)

file_performance_path = 'Performance_Baseline_Measure3-simGIC_Seco2004.txt'
file_output ='Results_Baseline_Measure3-simGIC_Seco2004.txt'
process_dataset_file(file_performance_path, file_output)

file_performance_path = 'Performance_Baseline_Measure4-simGIC_Resnik1995.txt'
file_output ='Results_Baseline_Measure4-simGIC_Resnik1995.txt'
process_dataset_file(file_performance_path, file_output)

file_performance_path = 'Performance_Baseline_Measure5-MAX_Seco2004.txt'
file_output ='Results_Baseline_Measure5-MAX_Seco2004.txt'
process_dataset_file(file_performance_path, file_output)

file_performance_path = 'Performance_Baseline_Measure6-MAX_Resnik1995.txt'
file_output ='Results_Baseline_Measure6-MAX_Resnik1995.txt'
process_dataset_file(file_performance_path, file_output)