# -*- coding: utf-8 -*-
'''
    File name: Run_OpenKE_Embeddings.py
    Authors: Susana Nunes, Rita T. Sousa, Catia Pesquita
    Python Version: OpenKE implementation in https://github.com/thunlp/OpenKE
    -default parameters.
'''

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import os
import json
import rdflib
import sys
from OpenKE import config
from OpenKE import models
from Run_KG_TransE-DistMult import *

#####################
##    Functions    ##
#####################
def construct_model(dic_nodes, dic_relations, list_triples, path_output, n_embeddings, model_embedding):
    """
    Construct and embedding model and compute embeddings.
    :param dic_nodes: dictionary with KG nodes and respective ids;
    :param dic_relations: dictionary with type of relations in the KG and respective ids;
    :param list_triples: list with triples of the KG;
    :param path_output: OpenKE path;
    :param n_embeddings: dimension of embedding vectors;
    :param models_embedding: embedding model;
    :return: write a json file with the embeddings for all nodes and relations.
    """
    entity2id_file = open(path_output + "entity2id.txt", "w")
    entity2id_file.write(str(len(dic_nodes))+ '\n')
    for entity , id in dic_nodes.items():
        entity = entity.replace('\n' , ' ')
        entity = entity.replace(' ' , '__' )
        entity = entity.encode('utf8')
        entity2id_file.write(str(entity) + '\t' + str(id)+ '\n')
    entity2id_file.close()

    relations2id_file = open(path_output + "relation2id.txt", "w")
    relations2id_file.write(str(len(dic_relations)) + '\n')
    for relation , id in dic_relations.items():
        relation  = relation.replace('\n' , ' ')
        relation = relation.replace(' ', '__')
        relation = relation.encode('utf8')
        relations2id_file.write(str(relation) + '\t' + str(id) + '\n')
    relations2id_file.close()

    train2id_file = open(path_output + "train2id.txt", "w")
    train2id_file.write(str(len(list_triples)) + '\n')
    for triple in list_triples:
        train2id_file.write(str(triple[0]) + '\t' + str(triple[2]) + '\t' + str(triple[1]) + '\n')
    train2id_file.close()

    # Input training files from data folder.
    con = config.Config()
    con.set_in_path(path_output)
    con.set_dimension(n_embeddings)

    print(
        '--------------------------------------------------------------------------------------------------------------------')
    print('MODEL: ' + model_embedding)

    # Models will be exported via tf.Saver() automatically.
    con.set_export_files(path_output + model_embedding + "/model.vec.tf", 0)
    # Model parameters will be exported to json files automatically.
    con.set_out_files(path_output + model_embedding + "/embeddings.vec.json")

    if model_embedding == 'ComplEx':
        con.set_work_threads(8)
        con.set_train_times(1000)
        con.set_nbatches(100)
        con.set_alpha(0.5)
        con.set_lmbda(0.05)
        con.set_bern(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.ComplEx)

    elif model_embedding == 'distMult':
        con.set_work_threads(8)
        con.set_train_times(1000)
        con.set_nbatches(100)
        con.set_alpha(0.5)
        con.set_lmbda(0.05)
        con.set_bern(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.DistMult)

    elif model_embedding == 'HOLE':
        con.set_work_threads(4)
        con.set_train_times(500)
        con.set_nbatches(100)
        con.set_alpha(0.1)
        con.set_bern(0)
        con.set_margin(0.2)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.HolE)

    elif model_embedding == 'RESCAL':
        con.set_work_threads(4)
        con.set_train_times(500)
        con.set_nbatches(100)
        con.set_alpha(0.1)
        con.set_bern(0)
        con.set_margin(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.RESCAL)

    elif model_embedding == 'TransD':
        con.set_work_threads(8)
        con.set_train_times(1000)
        con.set_nbatches(100)
        con.set_alpha(1.0)
        con.set_margin(4.0)
        con.set_bern(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransD)

    elif model_embedding == 'TransE':
        con.set_work_threads(8)
        con.set_train_times(1000)
        con.set_nbatches(100)
        con.set_alpha(0.001)
        con.set_margin(1.0)
        con.set_bern(0)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransE)

    elif model_embedding == 'TransH':
        con.set_work_threads(8)
        con.set_train_times(1000)
        con.set_nbatches(100)
        con.set_alpha(0.001)
        con.set_margin(1.0)
        con.set_bern(0)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransH)

    elif model_embedding == 'TransR':
        con.set_work_threads(8)
        con.set_train_times(1000)
        con.set_nbatches(100)
        con.set_alpha(1.0)
        con.set_lmbda(4.0)
        con.set_margin(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransR)

    # Train the model.
    con.run()



def write_embeddings(path_model_json, path_embeddings_output, ents, dic_nodes):
    """
    Writing embeddings.
    :param path_model_json: json file with the embeddings for all nodes and relations;
    :param path_embeddings_output: embedding file path;
    :param ents: list of entities for which embeddings will be saved;
    :param dic_nodes: dictionary with KG nodes and respective ids;
    :return: writes an embedding file with format "{ent1:[...], ent2:[...]}".
    """
    with open(path_model_json, 'r') as embeddings_file:
        data = embeddings_file.read()
    embeddings = json.loads(data)
    embeddings_file.close()

    ensure_dir(path_embeddings_output)
    with open(path_embeddings_output, 'w') as file_output:
        file_output.write("{")
        first = False
        for i in range(len(ents)):
            ent = ents[i]
            if first:
                if "ent_embeddings" in embeddings:
                    file_output.write(", '%s':%s" % (str(ent), str(embeddings["ent_embeddings"][dic_nodes[str(ent)]])))
                else:
                    file_output.write(
                        ", '%s':%s" % (str(ent), str(embeddings["ent_re_embeddings"][dic_nodes[str(ent)]])))
            else:
                if "ent_embeddings" in embeddings:
                    file_output.write("'%s':%s" % (str(ent), str(embeddings["ent_embeddings"][dic_nodes[str(ent)]])))
                else:
                    file_output.write(
                        "'%s':%s" % (str(ent), str(embeddings["ent_re_embeddings"][dic_nodes[str(ent)]])))
                first = True
        file_output.write("}")
    file_output.close()



def construct_embeddings_2ontos(ontology_1_file_path, ontology_2_file_path, annotations_1_file_path,
                 annotations_2_file_path,  dataset_file, model_embedding, n_embeddings, path_embeddings_output):

    Graph = buildGraph_2ontos(ontology_1_file_path, ontology_2_file_path, annotations_1_file_path,
                 annotations_2_file_path)
    dic_nodes, dic_relations, list_triples = buildIds(Graph)
    dict_labels, ents = process_dataset(dataset_file)

    path_output = './OpenKE/'
    path_model_json = './OpenKE/' + model_embedding + "/embeddings.vec.json"

    construct_model(dic_nodes, dic_relations, list_triples, path_output, n_embeddings, model_embedding)
    write_embeddings(path_model_json, path_embeddings_output, ents, dic_nodes)

def construct_embeddings_1onto(ontology_file_path, annotations_file_path,  dataset_file, model_embedding, n_embeddings, path_embeddings_output):

    Graph = buildGraph_1onto(ontology_file_path, annotations_file_path)
    dic_nodes, dic_relations, list_triples = buildIds(Graph)
    dict_labels, ents = process_dataset(dataset_file)

    path_output = './OpenKE/'
    path_model_json = './OpenKE/' + model_embedding + "/embeddings.vec.json"

    construct_model(dic_nodes, dic_relations, list_triples, path_output, n_embeddings, model_embedding)
    write_embeddings(path_model_json, path_embeddings_output, ents, dic_nodes)




#############################
##    Calling Functions    ##
#############################


if __name__ == "__main__":
    #Example of running KG with one ontology for distMult
    ontology_file_path = "HPO-full.owl"
    annotations_file_path = "annotations_HPO.tsv"
    entities_file = "OpenKE_entities.csv"
    n_embeddings = 200
    
    model_embedding = "distMult"
    #model_embedding = "TransE"
    
   
    path_embeddings_output = "/output/Run_HPOfull_Embeddings_DistMult"
    #path_embeddings_output = "/output/Run_HPOfull_Embeddings_TransE"


    construct_embeddings_1onto(ontology_file_path, annotations_file_path, dataset_file, model_embedding, n_embeddings,
                         path_embeddings_output)















