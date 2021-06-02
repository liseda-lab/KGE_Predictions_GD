IN CONSTRUCTION ....

# Predicting Gene-Disease Associations with Knowledge Graph Embeddings over Multiple Ontologies

## Introduction
Ontology-based approaches for predicting gene-disease associations include the more classical semantic similarity methods and more recently knowledge graph embeddings. While semantic similarity is typically restricted to hierarchical relations within the ontology, knowledge graph embeddings consider their full breadth. However, embeddings are produced over a single graph and complex tasks such as gene-disease association may require additional ontologies. We investigate the impact of employing richer semantic representations that are based on more than one ontology, able to represent both genes and diseases and consider multiple kinds of relations within the ontologies. Our experiments demonstrate the value of employing knowledge graph embeddings based on random-walks and highlight the need for a closer integration of different ontologies.

- This document provides the implementation described in the **short paper**: http://arxiv.org/abs/2105.04944 

## Dataset and Annotations
The dataset **Dataset_Pairs_Label.csv** has a total of 2716 genes, 1807 diseases, and 8189 disease-genes relations from DisGeNET and 8189 negative samples. GO annotations were downloaded from Gene Ontology Annotation (GOA) database for the human species. HP annotations were downloaded from the HP database, providing links between genes or diseases to HP terms. 

## Baseline





## Running Embeddings
TransE and DistMult implementation in https://github.com/thunlp/OpenKE with default parameters.

RDF2Vec implementation in  https://github.com/IBCNServices/pyRDF2Vec, the sequences are generated using the Weisfeiler-Lehman algorithm with walks depth 8 and a limited number of 500. The corpora of sequences were used to build a Skip-Gram model with the default parameters.

OPA2Vec implementation in https://github.com/bio-ontology-research-group/opa2vec with default parameters.


OWL2Vec* implementation in https://github.com/KRR-Oxford/OWL2Vec-Star with RDF2Vec parameters.



## Vector Operations
## Running Perfomance 






