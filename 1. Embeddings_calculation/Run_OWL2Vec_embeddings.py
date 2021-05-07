# -*- coding: utf-8 -*-
'''
    File name: Run_OWL2Vec_Embeddings.py
    Authors: Susana Nunes, Rita T. Sousa, Catia Pesquita
    Python Version: 3.7
    Original OWL2Vec* implementation in https://github.com/KRR-Oxford/OWL2Vec-Star
    -RDF2Vec parameters
'''
import os
import time
import argparse
import random
import multiprocessing
import gensim
import configparser
from owl2vec.RDF2Vec_Embed import get_rdf2vec_walks
from owl2vec.Label import pre_process_words, URI_parse
from owl2vec.Onto_Projection import Reasoner, OntologyProjection


########################################
#####     OWL2Vec Embeddings       #####
########################################
def run_OWL2Vec(ontology_file, IC_dict, sampler, walker, walk_depth, n_walks, Mix_Type, embed_size, window_arg, iteration, negative_arg, min_count_arg, seed_arg, epoch, URI_Doc_arg = 'yes', Lit_Doc_arg = 'yes', Mix_Doc_arg ='yes', cache_dir='./cache', ontology_projection="no", pre_entity_file="", pre_axiom_file="", pre_annotation_file="",  pre_train_model=""):

    start_time = time.time()
    if (ontology_projection =='yes') or (pre_entity_file == "") or (pre_axiom_file == "") or (pre_annotation_file == ""):
        print('\n Access the ontology ...')
        projection = OntologyProjection(ontology_file, reasoner=Reasoner.STRUCTURAL, only_taxonomy=False,
                                    bidirectional_taxonomy=True, include_literals=True, avoid_properties=set(),
                                    additional_preferred_labels_annotations=set(),
                                    additional_synonyms_annotations=set(),

                                    memory_reasoner='13351')
    else:
        projection = None

    # Ontology projection
    if (ontology_projection =='yes'):
        print('\nCalculate the ontology projection ...')
        projection.extractProjection()
        onto_projection_file = os.path.join(cache_dir, 'projection.ttl')
        projection.saveProjectionGraph(onto_projection_file)
        ontology_file = onto_projection_file
    else:
        ontology_file = ontology_file


    # Extract and save seed entities (classes and individuals)
    # Or read entities specified by the user
    if pre_entity_file != "":
        entities = [line.strip() for line in open(pre_entity_file).readlines()]
    else:
        print('\nExtract classes and individuals ...')
        projection.extractEntityURIs()
        classes = projection.getClassURIs()
        individuals = projection.getIndividualURIs()
        entities = classes.union(individuals)
        with open(os.path.join(cache_dir, 'entities.txt'), 'w') as f:
            for e in entities:
                f.write('%s\n' % e)

    # Extract axioms in Manchester Syntax if it is not pre_axiom_file is not set
    if pre_axiom_file == "":
        print('\nExtract axioms ...')
        projection.createManchesterSyntaxAxioms()
        with open(os.path.join(cache_dir, 'axioms.txt'), 'w') as f:
            for ax in projection.axioms_manchester:
                f.write('%s\n' % ax)

    # If pre_annotation_file is set, directly read annotations
    # else, read annotations including rdfs:label and other literals from the ontology
    #   Extract annotations: 1) English label of each entity, by rdfs:label or skos:preferredLabel
    #                        2) None label annotations as sentences of the literal document
    uri_label, annotations = dict(), list()

    if pre_annotation_file != "":
        with open(pre_annotation_file) as f:
            for line in f.readlines():
                tmp = line.strip().split()
                if tmp[1] == 'http://www.w3.org/2000/01/rdf-schema#label':
                    uri_label[tmp[0]] = pre_process_words(tmp[2:])
                else:
                    annotations.append([tmp[0]] + tmp[2:])

    else:
        print('\nExtract annotations ...')
        projection.indexAnnotations()
        for e in entities:
            if e in projection.entityToPreferredLabels and len(projection.entityToPreferredLabels[e]) > 0:
                label = list(projection.entityToPreferredLabels[e])[0]
                uri_label[e] = pre_process_words(words=label.split())
        for e in entities:
            if e in projection.entityToAllLexicalLabels:
                for v in projection.entityToAllLexicalLabels[e]:
                    if (v is not None) and (not (e in projection.entityToPreferredLabels and v in projection.entityToPreferredLabels[e])):
                        annotation = [e] + v.split()
                        annotations.append(annotation)

        with open(os.path.join(cache_dir, 'annotations.txt'), 'w') as f:
            for e in projection.entityToPreferredLabels:
                for v in projection.entityToPreferredLabels[e]:
                    f.write('%s preferred_label %s\n' % (e, v))
            for a in annotations:
                f.write('%s\n' % ' '.join(a))


    # read URI document
    # two parts: walks, axioms (if the axiom file exists)
    walk_sentences, axiom_sentences, URI_Doc = list(), list(), list()
    if URI_Doc_arg == 'yes':
        print('\nGenerate URI document ...')

        walks_ = get_rdf2vec_walks(dic_IC =IC_dict, onto_file=ontology_file, sampler_type= sampler, n_walks=n_walks, walker_type=walker, walk_depth=walk_depth, classes=entities)

        print('Extracted %d walks for %d seed entities' % (len(walks_), len(entities)))
        walk_sentences += [list(map(str, x)) for x in walks_]

        axiom_file = os.path.join(cache_dir, 'axioms.txt')
        if os.path.exists(axiom_file):
            for line in open(axiom_file).readlines():
                axiom_sentence = [item for item in line.strip().split()]
                axiom_sentences.append(axiom_sentence)
        print('Extracted %d axiom sentences' % len(axiom_sentences))
        URI_Doc = walk_sentences + axiom_sentences


    # Some entities have English labels
    # Keep the name of built-in properties (those starting with http://www.w3.org)
    # Some entities have no labels, then use the words in their URI name
    def label_item(item):
        if item in uri_label:
            return uri_label[item]
        elif item.startswith('http://www.w3.org'):
            return [item.split('#')[1].lower()]
        elif item.startswith('http://'):
            return URI_parse(uri=item)
        else:
            return [item.lower()]


    # read literal document
    # two parts: literals in the annotations (subject's label + literal words)
    #            replacing walk/axiom sentences by words in their labels
    Lit_Doc = list()
    if Lit_Doc_arg == 'yes':
        print('\nGenerate literal document ...')
        for annotation in annotations:
            processed_words = pre_process_words(annotation[1:])
            if len(processed_words) > 0:
                Lit_Doc.append(label_item(item=annotation[0]) + processed_words)
        print('Extracted %d annotation sentences' % len(Lit_Doc))

        for sentence in walk_sentences:
            lit_sentence = list()
            for item in sentence:
                lit_sentence += label_item(item=item)
            Lit_Doc.append(lit_sentence)

        for sentence in axiom_sentences:
            lit_sentence = list()
            for item in sentence:
                lit_sentence += label_item(item=item)
            Lit_Doc.append(lit_sentence)

    # read mixture document
    # for each axiom/walk sentence, all): for each entity, keep its entity URI, replace the others by label words
    #                            random): randomly select one entity, keep its entity URI, replace the others by label words
    Mix_Doc = list()
    if Mix_Doc_arg == 'yes':
        print('\nGenerate mixture document ...')
        for sentence in walk_sentences + axiom_sentences:
            if Mix_Type == 'all':
                for index in range(len(sentence)):
                    mix_sentence = list()
                    for i, item in enumerate(sentence):
                        mix_sentence += [item] if i == index else label_item(item=item)
                    Mix_Doc.append(mix_sentence)
            elif Mix_Type == 'random':
                random_index = random.randint(0, len(sentence) - 1)
                mix_sentence = list()
                for i, item in enumerate(sentence):
                    mix_sentence += [item] if i == random_index else label_item(item=item)
                Mix_Doc.append(mix_sentence)

    print('URI_Doc: %d, Lit_Doc: %d, Mix_Doc: %d' % (len(URI_Doc), len(Lit_Doc), len(Mix_Doc)))
    all_doc = URI_Doc + Lit_Doc + Mix_Doc

    print('Time for document construction: %s seconds' % (time.time() - start_time))
    random.shuffle(all_doc)

    # learn the language model (train a new model or fine tune the pre-trained model)
    start_time = time.time()
    if not os.path.exists(pre_train_model):
        print('\nTrain the language model ...')
        model_ = gensim.models.Word2Vec(all_doc, size=int(embed_size),
                                    window=int(window_arg),
                                    workers=multiprocessing.cpu_count(),
                                    sg=1, iter=int(iteration),
                                    negative=int(negative_arg),
                                    min_count=int(min_count_arg), seed=int(seed_arg))


    else:
        print('\nFine-tune the pre-trained language model ...')
        model_ = gensim.models.Word2Vec.load(pre_train_model)
        if len(all_doc) > 0:
            model_.min_count = int(min_count_arg)
            model_.build_vocab(all_doc, update=True)
            model_.train(all_doc, total_examples=model_.corpus_count, epochs=int(epoch))

    #model_.save(embedding_dir)
    embeddings = [model_.wv[str(entity)] for entity in entities]
    print('Time for learning the language model: %s seconds' % (time.time() - start_time))
    print('Model saved. Done!')

    return(embeddings)


########################################
#####    Embeddings Creation       #####
########################################
def write_embeddings(embedding_file_path, entities, embeddings):

    with open(embedding_file_path, 'w') as file:
        file.write("{")
        first = False
        for i in range(len(entities)):
            if first:
                file.write(", '%s':%s" % (str(entities[i]), str(embeddings[i].tolist())))
            else:
                file.write("'%s':%s" % (str(entities[i]), str(embeddings[i].tolist())))
                first=True
            file.flush()
        file.write("}")


########################################
#####        Run Embeddings        #####
########################################

# use URI/Literal/Mixture document (yes or no)
# they can be over witten by the command line parameters
URI_Doc_arg = "yes"
Lit_Doc_arg = "yes"
Mix_Doc_arg ="no"
# cache directory for storing files; if not set, it creates a default: ./cache/
cache_dir = './cache'
# use or not use the projected ontology
# default: no
ontology_projection = "no"
# the seed entities for generating the walks
# default: all the named classes and instances cached in cache/entities.txt
# comment it if the default python program is called to extract all classes and individuals as the seed entities
# pre_entity_file = ./cache/entities.txt
#pre_entity_file = ""
# the annotations and axioms can pre-calculated and saved in the some directory (e.g., the cache directory)
# OWL2Vec will use the pre-calculated files if set, or it will extract them by default
# comment them if the default python program is called to extract annotations and axioms
pre_annotation_file = ""
# pre_axiom_file = ./cache/axioms.txt
pre_axiom_file = ""
# walker and walk_depth must be set
# random walk or random walk with Weisfeiler-Lehman (wl) subtree kernel
walker = "wl"
walk_depth = 8
n_walks = 500
# the type for generating the mixture document (all or random)
# works when Mix_Doc is set to yes
Mix_Type = random
# the size for embedding
# it is set to the size of the pre-trained model if it is adopted
embed_size = 200
# for training the language model without pre-training
window_arg = 5
negative_arg = 25
min_count_arg = 1
seed_arg = 42
# number of iterations in training the language model
iteration = 10
# epoch for fine-tuning the pre-trained model
epoch = 100
# the directory of the pre-trained language model
# default: without pre-training
# comment it if no pre-training is needed
pre_train_model= "~/w2v_model/enwiki_model/word2vec_gensim"
samplers = "uniform"
IC_dict = {}
pre_entity_file = "entities.txt"
entities = [line.strip() for line in open(pre_entity_file).readlines()]
ontology_file = "HPO.owl"

embeddings = run_OWL2Vec(ontology_file, IC_dict, samplers, walker, walk_depth, n_walks, Mix_Type, embed_size,
                             window_arg, iteration, negative_arg, min_count_arg, seed_arg, epoch, URI_Doc_arg,
                             Lit_Doc_arg, Mix_Doc_arg, cache_dir, ontology_projection, pre_entity_file, pre_axiom_file,
                             pre_annotation_file, pre_train_model)


embedding_file_path = "Owl2Vec_Output_Embeddings.txt"
write_embeddings(embedding_file_path, entities, embeddings)
