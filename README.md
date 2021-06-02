IN CONSTRUCTION ....

# Predicting Gene-Disease Associations with Knowledge Graph Embeddings over Multiple Ontologies

## Introduction
Ontology-based approaches for predicting gene-disease associations include the more classical semantic similarity methods and more recently knowledge graph embeddings. While semantic similarity is typically restricted to hierarchical relations within the ontology, knowledge graph embeddings consider their full breadth. However, embeddings are produced over a single graph and complex tasks such as gene-disease association may require additional ontologies. We investigate the impact of employing richer semantic representations that are based on more than one ontology, able to represent both genes and diseases and consider multiple kinds of relations within the ontologies. Our experiments demonstrate the value of employing knowledge graph embeddings based on random-walks and highlight the need for a closer integration of different ontologies.

- This document provides the implementation described in the **short paper**: http://arxiv.org/abs/2105.04944 

## Dataset and Annotations
__Dataset_Pairs_Label.csv__ has a total of 2716 genes, 1807 diseases, and 8189 disease-genes relations from DisGeNET and 8189 negative samples. GO annotations were downloaded from Gene Ontology Annotation (GOA) database for the human species. HP annotations were downloaded from the HP database, providing links between genes or diseases to HP terms. 

## Baseline
Uses the SSMC tool (more details available in https://github.com/liseda-lab/SSMC) and six different semantic similarity measures:
- BMA ICSeco 
- BMA ICResnik 
- SimGIC ICSeco 
- SimGIC ICResnik
- Max ICSeco
- MAX ICResnik 

SSMC accepts as input a JSON file with a series of mandatory (and optional) user defined configurations. An example has been provided in Configuration.json in the SSMC Tool folder. 
The association prediction performed in __Threshold_Baseline.py__ (SS_Baseline folder)is expressed as a classification problem where a score for a gene-disease pair exceeding a certain threshold indicates a positive association. 

## Running Embeddings

KGEs with 200 features and five different methods that cover different approaches for KGE:
- Translational Distance (TransE)
- Semantic Matching (DistMult) 
- Random Walk-based (RDF2Vec, OPA2Vec, OWL2Vec*)

-- TransE and DistMult implementation in https://github.com/thunlp/OpenKE with default parameters.

-- RDF2Vec implementation in  https://github.com/IBCNServices/pyRDF2Vec, the sequences are generated using the Weisfeiler-Lehman algorithm with walks depth 8 and a limited number of 500. The corpora of sequences were used to build a Skip-Gram model with the default parameters.

-- OPA2Vec implementation in https://github.com/bio-ontology-research-group/opa2vec with default parameters.

-- OWL2Vec* implementation in https://github.com/KRR-Oxford/OWL2Vec-Star with RDF2Vec parameters.


## Vector Operations
Each gene-disease pair corresponds to two vectors, g and d, associated with a gene and a disease, respectively. We defined an operator over the corresponding vectors in order to generate a representation r(g,d). Several choices for the operator were considered:
- Concatenation
- Average
- Hadamard
- Weighted-L1
- Weighted-L2 

We measured the cosine similarity between the vectors carrying out the same approach used in the __Baseline__ with a SS threshold.

## Running Perfomance 
The resulting vectors were then the input to four different ML algorithms: 
- Random Forest
- eXtreme Gradient Boosting
- Naïve Bayes
- Multi-Layer Perceptron 

Grid search was employed in __Performance_ML.py__ to obtain optimal parameters for RF, XGB, and MLP. 


## Authors
- __Susana Nunes__
- __Rita T. Sousa__
- __Catia Pesquita__

For any comments or help needed with this implementation, please send an email to: snunes@lagise.di.fc.ul.pt

## Acknowledgments
This work was supported by FCT through LASIGE Research Unit (ref. UIDB/00408/2020 and ref. UIDP/00408/2020. It was also partially supported by the KATY project which has received funding from the European Union’s Horizon 2020 research and innovation program under grant agreement No 101017453.
