# -*- coding: utf-8 -*-
'''
    File name: owlready2_GOannots.py
    Authors: Susana Nunes, Rita T. Sousa, Catia Pesquita
    Python Version: 3.7
'''
from owlready2 import *
import Bio.UniProt.GOA as GOA
from goatools import obo_parser


##############################################
##   TREAT OBO FILE - Parse obo to graph    ##
##############################################

# parses the gene ontology file in obo format
def create_obo_graph(GO_obo_file):
    graph_obo = obo_parser.GODag(GO_obo_file)

    return graph_obo

##############################################
##  TREAT ANOTATIONS FILE - Create 1stdict  ##
##############################################

# Reads annotations file (entity \t GO) and creates a dictionary {entity_id{term:[GO list]}}
def create_annots_dictionary(annotations_file):
    annots = open(annotations_file, 'r')
    annots_file = annots.readlines()
    annots_dict = {}
    for annot in annots_file:
        ent, term_list = annot[:-1].split('\t')
        annots_dict[ent] = {'terms': []}
        for term in term_list.split(';'):
            annots_dict[ent]['terms'].append('GO:' + term)

    return annots_dict


##############################################
##    OBO + ANOTATIONS - Create 2nd dict    ##
##############################################

# Reads 1st dictionary created and produces a new dictionary
# that separates the term by the 2 aspects in GO from the graph of the obo file
# (cellular component, biological process, molecular function)

def create_final_annots_dict(annots_dict, graph_obo):

    annotations = annots_dict
    final_annots_dict = {}
    for entity_id in annotations:
        final_annots_dict[entity_id] = {'CC': [], 'BP': [], 'MF': []}
        for GO_term in annotations[entity_id]['terms']:
            if GO_term in graph_obo:
                go_term = graph_obo[GO_term]
                aspect = go_term.namespace
                GO_term = GO_term.replace(':', '_')
                if aspect == 'biological_process':
                    final_annots_dict[entity_id]['BP'].append(GO_term)
                elif aspect == 'molecular_function':
                    final_annots_dict[entity_id]['MF'].append(GO_term)
                elif aspect == 'cellular_component':
                    final_annots_dict[entity_id]['CC'].append(GO_term)

    return final_annots_dict

#############################
#    Create instances       #
#############################

#Creates an instance of the tipe Entity (Disease or Gene), and atributes the annotations
def ontology_instance(gene_id, annotations):
    instance = Gene(gene_id)

    for term in annotations[gene_id]['BP']:
        go_term = go.search_one(iri='http://purl.obolibrary.org/obo/' + term)
        instance.is_a.append(RO_0002331.some(go_term))
    for term in annotations[gene_id]['CC']:
        go_term = go.search_one(iri='http://purl.obolibrary.org/obo/' + term)
        instance.is_a.append(BFO_0000050.some(go_term))
    for term in annotations[gene_id]['MF']:
        go_term = go.search_one(iri='http://purl.obolibrary.org/obo/' + term)
        instance.is_a.append(RO_0002327.some(go_term))

    return gene_id

##############################################
##                RUN PROGRAM               ##
##############################################

###############
##   Data    ##
###############
GO_obo_file = 'go-basic.obo'
annotations_file = 'AnnotationsGO_without_url.tsv'
ontology_file = 'GO-full.owl'

###############
## functions ##
###############

#STEP 1: Creates a graph through the ontology in obo format 
graph_obo = create_obo_graph(GO_obo_file)

#STEP 2: Creates a initial dictionary with the genes 
my_annots = create_annots_dictionary(annotations_file)

#STEP 3: Creates a final dictionary diivided by the three GO categories
annots_dict = create_final_annots_dict(my_annots, graph_obo)

#STEP 4: Loads the ontology in owl format
go = get_ontology(ontology_file).load()

#STEP 5: Adds the namespaces
obo = go.get_namespace("http://purl.obolibrary.org/obo/")

#STEP 6: Projects the classes and relations that will be used in the instances

with go:
    class Gene(Thing):
        namespace = obo

    class RO_0002327(Thing >> Thing):  # enables
        namespace = obo
        label = 'enables'

    class BFO_0000050(Thing >> Thing):  # part of
        namespace = obo
        label = 'part of'

    class RO_0002331(Thing >> Thing):  # involved in
        namespace = obo
        label = 'involved in'

#STEP 7: Creates the instances and annotations
for entity_id in annots_dict:
    ontology_instance(entity_id, annots_dict)

#STEP 8: Saves the owl file with annotations
go.save(file="GO_with_annotations.owl.owl", format="rdfxml")

