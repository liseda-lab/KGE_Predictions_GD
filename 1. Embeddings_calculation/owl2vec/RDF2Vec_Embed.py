import rdflib
import sys
import numpy as np

sys.path.append('../pyrdf2vec/')
from pyrdf2vec.graphs import kg
from pyrdf2vec.samplers import (
    ObjFreqSampler,
    ObjPredFreqSampler,
    PageRankSampler,
    PredFreqSampler,
    UniformSampler,
    ICSampler,)
from pyrdf2vec.walkers import RandomWalker, WeisfeilerLehmanWalker
from pyrdf2vec.rdf2vec import RDF2VecTransformer

def construct_kg_walker(onto_file, dic_IC, sampler_type, n_walks, walker_type, walk_depth):
    g = rdflib.Graph()
    if onto_file.endswith('ttl') or onto_file.endswith('TTL'):
        g.parse(onto_file, format='turtle')
    else:
        g.parse(onto_file)
    KnowledgeGraph = kg.KG()
    for (s, p, o) in g:
        s_v, o_v = kg.Vertex(str(s)), kg.Vertex(str(o))
        p_v = kg.Vertex(str(p), predicate=True, vprev=s_v, vnext=o_v)
        KnowledgeGraph.add_vertex(s_v)
        KnowledgeGraph.add_vertex(p_v)
        KnowledgeGraph.add_vertex(o_v)
        KnowledgeGraph.add_edge(s_v, p_v)
        KnowledgeGraph.add_edge(p_v, o_v)

    if sampler_type.lower() == 'ic':
        sampler = ICSampler(dic_IC, inverse=True)
    elif sampler_type.lower() == 'icinverse':
        sampler = ICSampler(dic_IC)
    elif sampler_type.lower() == 'uniform':
        sampler = UniformSampler()

    if walker_type.lower() == 'random':
        walker = RandomWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)
    elif walker_type.lower() == 'wl':
        walker = WeisfeilerLehmanWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)

    else:
        print('walker %s not implemented' % walker_type)
        sys.exit()

    return KnowledgeGraph, walker


def get_rdf2vec_embed(dic_IC, onto_file, sampler_type, walker_type, walk_depth, embed_size, classes):
    kg, walker = construct_kg_walker(onto_file=onto_file, dic_IC=dic_IC, sampler_type = sampler_type, walker_type=walker_type, walk_depth=walk_depth)
    transformer = RDF2VecTransformer(walkers=[walker], vector_size=embed_size)
    instances = [rdflib.URIRef(c) for c in classes]
    walk_embeddings = transformer.fit_transform(graph=kg, instances=instances)
    return np.array(walk_embeddings)


def get_rdf2vec_walks(dic_IC, onto_file, sampler_type, n_walks, walker_type, walk_depth, classes):
    kg, walker = construct_kg_walker(onto_file=onto_file, dic_IC=dic_IC, sampler_type = sampler_type, n_walks=n_walks, walker_type=walker_type, walk_depth=walk_depth)
    instances = [rdflib.URIRef(c) for c in classes]
    walks_ = list(walker.extract(kg=kg, instances=instances))
    return walks_
