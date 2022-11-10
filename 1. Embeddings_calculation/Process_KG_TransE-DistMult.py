# -*- coding: utf-8 -*-
'''
    File name: Process_KG_TransE-DistMult.py
    Authors: Susana Nunes, Rita T. Sousa, Catia Pesquita
    Python Version: 3.7
'''

import rdflib
from rdflib.namespace import RDF, OWL


#####################
##    Functions    ##
#####################

def process_dataset(file_dataset_path):
    """
    Process a dataset file.
    :param file_dataset_path: dataset file path with the correspondent entity pairs. The format of each line of the dataset files is "Nr Ent1 Ent2 label";
    :return: one dictionary and one list. "dict_labels" is a dictionary with entity pairs and respective label. "ents" is a list of entities for which embeddings will be computed.
    """
    dataset = open(file_dataset_path, 'r')
    dict_labels = {}
    ents =[]

    for line in dataset[1:]:
        nr, gene, disease, label = line.strip(";\n").split(";")

        url_ent1 = "http://purl.obolibrary.org/obo/" + gene
        url_ent2 = "http://purl.obolibrary.org/obo/" + disease

        dict_labels[(url_ent1, url_ent2)] = label

        if url_ent1 not in ents:
            ents.append(url_ent1)
        if url_ent2 not in ents:
            ents.append(url_ent2)

    dataset.close()
    return dict_labels, ents

def buildGraph_2ontos(ontology_1_file_path, ontology_2_file_path, annotations_1_file_path,
                 annotations_2_file_path): #For 2 ontologies

    Kg = rdflib.Graph()
    Kg.parse(ontology_1_file_path, format='xml')
    Kg.parse(ontology_2_file_path, format='xml')

    file_annot_hpo = open(annotations_1_file_path, 'r')
    for annot in file_annot_hpo:
        ent, hpo_term_list = annot[:-1].split('\t')

        url_ent = "http://purl.obolibrary.org/obo/" + ent  # url are the initials of hpo
        

        for url_hpo_term in hpo_term_list.split(';'):
            Kg.add((rdflib.term.URIRef(url_ent), rdflib.term.URIRef('http://purl.obolibrary.org/obo/hasAnnotation'),
                    rdflib.term.URIRef(url_hpo_term)))

    file_annot_go = open(annotations_2_file_path, 'r')
    for annot in file_annot_go:
        ent, go_term_list = annot[:-1].split('\t')

        url_ent = "http://purl.obolibrary.org/obo/" + ent  # url  are the initials of GO

        for url_go_term in go_term_list.split(';'):
            Kg.add((rdflib.term.URIRef(url_ent), rdflib.term.URIRef('http://purl.obolibrary.org/obo/hasAnnotation'),
                    rdflib.term.URIRef(url_go_term)))

    print('KG created')
    file_annot_hpo.close()
    file_annot_go.close()

    return Kg

def buildGraph_1onto(ontology_file_path, annotations_file_path): #For one ontology only
    kg = rdflib.Graph()
    kg.parse(ontology_file_path, format='xml')

    file_annot = open(annotations_file_path, 'r')

    for annot in file_annot:
        ent, hp_term_list = annot[:-1].split('\t')

        url_ent = "http://purl.obolibrary.org/obo/" + ent  # url são are the initials of HPO

        for url_hp_term in hp_term_list.split(';'):
            kg.add((rdflib.term.URIRef(url_ent), rdflib.term.URIRef('http://purl.obolibrary.org/obo/hasAnnotation'),
                    rdflib.term.URIRef(url_hp_term)))

    file_annot.close()
    return kg

def buildIds(g):
    """
    Assigns ids to KG nodes and KG relations.
    :param g: knowledge graph;
    :return: 2 dictionaries and one list. "dic_nodes" is a dictionary with KG nodes and respective ids. "dic_relations" is a dictionary with type of relations in the KG and respective ids. "list_triples" is a list with triples of the KG.
    """
    dic_nodes = {}
    id_node = 0
    id_relation = 0
    dic_relations = {}
    list_triples = []

    for (subj, predicate, obj) in g:
        if str(subj) not in dic_nodes:
            dic_nodes[str(subj)] = id_node
            id_node = id_node + 1
        if str(obj) not in dic_nodes:
            dic_nodes[str(obj)] = id_node
            id_node = id_node + 1
        if str(predicate) not in dic_relations:
            dic_relations[str(predicate)] = id_relation
            id_relation = id_relation + 1
        list_triples.append([dic_nodes[str(subj)] , dic_relations[str(predicate)] , dic_nodes[str(obj)]])

    return dic_nodes, dic_relations, list_triples
    


