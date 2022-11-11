# -*- coding: utf-8 -*-
'''
    File name: owlready2_HPannots.py
    Authors: Susana Nunes, Rita T. Sousa, Catia Pesquita
    Python Version: 3.7
'''

from owlready2 import *
import Bio.UniProt.GOA as GOA
from goatools import obo_parser


###############################################
## TREAT 2 ANOTATIONS FILES - Create 2 dicts ##
###############################################

# Reads the two annotations file (genes and diseases of HPO) and creates a dictionary {gene/disease_id{term:[HPO list]}}
def create_annots_dictionary(annotations_file_genes, annotations_file_diseases):
    gene_annots_file = open(annotations_file_genes, 'r')
    gene_annots = gene_annots_file.readlines()
    disease_annots_file = open(annotations_file_diseases, 'r')
    disease_annots = disease_annots_file.readlines()
    genes_dict = {}
    diseases_dict = {}

    for annot in gene_annots:
        ent, term_list = annot[:-1].split('\t')
        genes_dict[ent] = {'terms': []}
        for term in term_list.split(';'):
            genes_dict[ent]['terms'].append('HP_' + term)

    for annot in disease_annots:
        ent, term_list = annot[:-1].split('\t')
        diseases_dict[ent] = {'terms': []}
        for term in term_list.split(';'):
            diseases_dict[ent]['terms'].append('HP_' + term)

    return genes_dict, diseases_dict


#############################
#    Create instances       #
#############################

#Creates an instance of the type Disease or Gene and atributes the respectives annotations
def ontology_instance_gene(gene_id, gene_annotations):
    instance = Gene(gene_id)

    for term in gene_annotations[gene_id]['terms']:
        hpo_term = hpo.search_one(iri='http://purl.obolibrary.org/obo/' + term)
        instance.is_a.append(RO_0002327.some(hpo_term))

    return gene_id

def ontology_instance_disease(disease_id, disease_annotations):
    instance = Disease(disease_id)
    for term in disease_annotations[disease_id]['terms']:
        hpo_term = hpo.search_one(iri='http://purl.obolibrary.org/obo/' + term)
        instance.is_a.append(RO_0002327.some(hpo_term))

    return disease_id

##############################################
##                RUN PROGRAM               ##
##############################################

###############
##   Data    ##
###############

annotations_file_genes = 'AnnotationsHP_genes_without_url.tsv' 
annotations_file_diseases = 'AnnotationsHP_diseases_without_url.tsv'
ontology_file = 'HPfull.owl'


###############
## functions ##
###############

#STEP 1: creates dictionaries with anotations for genes and for diseases
genes_dict, diseases_dict = create_annots_dictionary(annotations_file_genes, annotations_file_diseases)

#STEP 2: Loads the ontology in owl format
hpo = get_ontology(ontology_file).load()

#STEP 3: adds the namespaces of the uris used in the ontology
obo = hpo.get_namespace("http://purl.obolibrary.org/obo/")

#STEP 4: projects classes and relations that will be used in the instances of genes 
with hpo:
    class Gene(Thing):
        namespace = obo

    class RO_0002327(Thing >> Thing):  # enables
        namespace = obo
        label = 'enables'


#STEP 5: creates all the instances and the annotations in the genes dictionary
for gene_id in genes_dict:
    ontology_instance_gene(gene_id, genes_dict)

#STEP 6: projects classes and relations that will be used in the instances of  diseases
with hpo:
    class Disease(Thing):
        namespace = obo

    class RO_0002327(Thing >> Thing):  # enables
        namespace = obo
        label = 'enables'

#STEP 7: creates all the instances and the annotations in the diseases dictionary
for disease_id in diseases_dict:
    ontology_instance_disease(disease_id, diseases_dict)

#STEP 8: Saves owl file with annotations
hpo.save(file="HPfull_with_annotations.owl", format="rdfxml")
