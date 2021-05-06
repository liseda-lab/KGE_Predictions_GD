'''
    File name: SSMC-script.py
    Author: David Carriço Teixeira
    Date created: 15/09/2020
    Python Version: 3.7
'''
import json,sys,logging
from owlready2 import *
import owlready2 
from math import log,inf
from statistics import mean
from itertools import product,chain
from collections import Counter
import pandas as p
from sklearn.cluster import KMeans,SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'figure.max_open_warning': 0})
sns.set()
logging.basicConfig(format='%(message)s',level = logging.INFO)



#path to configurations
settingsPath = "C:/Users/Admin/Desktop/TESE/Script SSM/Configuration.json"

#===============================================Ontology Handling and basic Functions=======================================================

def get_nodeList(ontology):
    """Returns all classes in a loaded Ontology"""
    return list(ontology.classes())

def get_roots(removeSingletons=True,removeObsoletes=True):
    """ Returns the original roots of all loaded ontologies and removes (optional) Singletons and Obsoletes"""
    roots=list(Thing.subclasses())
    roots=list([x for x in roots if len(x.label)> 0])
    logging.info("Found "+str(len(roots))+" roots")
    obsoletes,singletons=[],[]
    if removeObsoletes or removeSingletons:
        for x in roots:
            if removeObsoletes:
                if len(x.deprecated)==1:
                    obsoletes.append(x)
            if removeSingletons:
                if len(list(x.subclasses()))==0 and x not in obsoletes:
                    singletons.append(x)
        actualRoots=[node for node in roots if node not in obsoletes+singletons]
        logging.info("   > Removed "+str(len(obsoletes))+" obsolete classes")     
        logging.info("   > Removed "+str(len(singletons))+" singletons")
        logging.info("   > Retrieved "+str(len(actualRoots))+" roots")
    return list(actualRoots)

def get_leaves(nodeList):
    """Takes a list of ontology classes and returns a list of classes that are ontology leaves"""
    return [x for x in nodeList if len(list(x.subclasses()))==0]

def get_subclasses(node):
    """Finds the subclasses of a node"""
    return list(node.subclasses())

def get_ancestors(node, include_constructs = True):
    """Returns a list of node ancestors, may include properties (e.g. part of)"""
    return list(node.ancestors(include_self = True,include_constructs = include_constructs))

def get_descendants(node):
    """Returns a list of concepts desceding from the node excluding itself"""
    return list(node.descendants(include_self =True))

def get_parents(node):
    """Returns a list of concepts superseding the node"""
    return node.is_a


#==================================================Misc=========================================================================

def LoadInfo(settings):
    """Retrieves information from the submitted user's settings
       Requires dict settings with user defined parameters
       Returns ontologyInfo,ssmIDs,queries,normalize_values,reasoning
    """
    try:
        ontologyInfo=settings["Ontologies"]
        ssmIDs=settings["Similarity Settings"]
        queries=settings["Queries"]
        logging.info("   > Ontologies found: "+str(len(ontologyInfo)))
        logging.info("   > SSMs found: "+str(len(ssmIDs)))
        logging.info("   > Queries found: "+str(len(queries)))
    except KeyError as e:
        logging.info("\nERROR: The Settings File is lacking a "+str(e)+" entry")
        quit()
    if "Normalize" in list(settings):
        normalize_values=settings["Normalize"]
    else:
        normalize_values=True
    if "Reasoning" in list(settings):
        reasoning=settings["Reasoning"]
    else:
        reasoning=False
    return ontologyInfo,ssmIDs,queries,normalize_values,reasoning

def normalize(maxV,minV,value):
    """Normalizes a value from a max and a min value"""
    return (value-minV)/(maxV-minV)


#===========================================Ontology Editing and Reasoning=======================================================


def subclass_inference(nodeList,namespace):
    """
    Converts equivalent classes into is_a parent classes
    Requires a list nodeList of classes, and an ontology namespace
    """
    for node in nodeList:
        equivList=node.equivalent_to#get quivalent classes (does nothing in GO)
        if len(equivList)> 0:
            for x in str(equivList[0]).split(" & "):
                y=x.split(".")
                if len(y)==2:
                    ref=namespace.base_iri+y[1]
                    z=IRIS[ref]
                    if equivList not in node.is_a:
                        node.is_a.append(z)


#=====================================================Compute IC=================================================================

def get_IC(classes, icList, normalize_values,classDescendant,classAncestor, annotData,**args):
    """ Computation and normalization (optional) of IC
        Requires list classes of all ontology classes, list icList of ICs to compute, bool normalize_values, dicts classDescendant and classAncestor with descendants and ancestors of each class
        Returns dict computedICs with the IC of every class
    """
    computedICs={}
    for ic in icList:
        icMap = eval("ic"+ic)(nodeList=classes,classDescendant=classDescendant,classAncestor=classAncestor,annotData=annotData)   
        logging.info("   > "+ic)
        computedICs[ic] = icMap#Dict of IC dicts
        computedICs["-"+ic] = {k:(1-v) for k,v in icMap.items()}#Negative Version
    return computedICs

def icSanchez2011(nodeList,classDescendant,classAncestor,**args):
    """ Compute IC Sanchez 2011
        Requires list nodeList of classes and dicts classDescendant and classAncestor with descendants and ancestors of each class
        Returns a dict of computed IC values for classes
    """
    icMap=dict.fromkeys(nodeList,{})
    maxLeaves=len(get_leaves(nodeList))
    for node in nodeList:
        if node==Thing:
            leaves=maxLeaves
        else:
            leaves=len([x for x in classDescendant[node] if len(list(x.subclasses()))==0])
        subsumers=len(classAncestor[node])#Subsumers (Include itself)
        icScore=-log(((abs(leaves))/(abs(subsumers)+1)+1)/(maxLeaves+1))
        icMap[node] = icScore
    return icMap
    
def icSeco2004(nodeList,classDescendant,**args):
    """ Computes IC Seco 2004
        Requires list nodeList of classes and dicts classDescendant with the descendants of each class
        Returns a dict of the normalized computed IC values for classes
    """
    icMap=dict.fromkeys(nodeList,{})
    for node in nodeList:
        hyponyms=len(classDescendant[node])#get Hyponyms of a concept
        icScore=1-(np.log2(hyponyms)/np.log2(len(nodeList)))#IC_Seco (removed 1 form hyponims)
        icMap[node]= icScore

    return icMap

def icResnik1995(nodeList,classDescendant,annotData,**args):
    """ Computes intrinsic IC Resnic 1995
        Requires list nodeList of classes and dicts classDescendant
        Returns a normalized dict of computed IC values for classes
    """
    icMap=dict.fromkeys(nodeList,{}) 
    classOccurence={node:[x for x in annotData if node in annotData[x]] for node in nodeList}   
    for node in classOccurence:
        if len(list(node.subclasses()))==0:
            classOccurence[node]=classOccurence[node]+[node]  
    classInstances={node:len(set(list(itertools.chain.from_iterable([classOccurence[x] for x in classDescendant[node]])))) for node in nodeList}   
    
   # classInstances=[x for x in classInstances if x!=owl.Thing]#Exclude THING?
    
    #sorted(list(set(classInstances.values())))[-2]
    maximum=max(classInstances.values())+1
    classTemp={k:(v+1)/maximum for k,v in classInstances.items()}
    minimum=min(classTemp.values())
    for node in nodeList:
        icMap[node]=log(classTemp[node])/log(minimum)
    return icMap


#=====================================================PAIRWISE SS=================================================================


def get_Pairwise(c1,c2,icMap,pairwise,store,classAncestor,polar=False,pol='+'):#mudar para term sim
    """ Computation of pairwise SS values
        Requires a dict of IC values, two e1 and e2 nodes, a str Pairwise measure, and a dict store
        Returns and writes result to store
    """
    try:
        key=str(c1)+"-"+str(c2)
        return store[key]
    except KeyError:
        pwScore=eval("pairwise_"+ pairwise)(icMap=icMap,c1=c1,c2=c2,classAncestor=classAncestor,pol=pol)
        store[key]=pwScore
        return pwScore

def pairwise_Resnik1995(icMap,c1,c2,classAncestor,pol,polar=False,**args):
    """ Computaion of Resnik 1995 pairwise similarity
        Requires a dict of IC values, tow classes e1 and e2 , dict classAncestor with class ancestors
        Returns the MICA between e1 and e2 if there is one, or it returns 0 
    """
    if polar:
        if pol=="+":
            commonAncestors=list(set(classAncestor[c1]).intersection(set(classAncestor[c2])))#get Common Ancestors
            return max([icMap[x] for x in commonAncestors])#get and return MICA IC
        else:
            commonDescendants=list(set(classDescendant[c1]).intersection(set(classDescendant[c2])))
            return max([icMap[x] for x in commonDescendants])
    else:
        commonAncestors=list(set(classAncestor[c1]).intersection(set(classAncestor[c2])))#get Common Ancestors
        return max([icMap[x] for x in commonAncestors])#get and return MICA IC     

#=====================================================GROUPWISE SS=================================================================


def groupwise_simGIC(ssm,entityMatches,annotData,classAncestor,**args):
    """ Computation of SimGIC groupise semantic similarity
        Requires icMap dict of IC values, entityMatches dict of entity-to-entity combinations, dict of entity annotations
        Returns entityMatches with computed Groupwise Similarity scores
    """
    icMap=eval("classIC[\""+str(ssm["IC"])+"\"]")
    errorList=[]
    for k,entry in entityMatches.items():
        if k%(len(entityMatches.keys())/20)==0:#print progress
            logging.info("      "+str(k)+"/"+str(len(entityMatches.keys())))
        try:
            e1=list(itertools.chain.from_iterable([classAncestor[x] for x in annotData[entry[0]]]))
            e2=list(itertools.chain.from_iterable([classAncestor[x] for x in annotData[entry[1]]]))
        except KeyError as e:
            if str(e) not in errorList:#Entity is not annotated
                errorList.append(str(e))
            entityMatches[k]="NO DATA"
            continue
        sumIntersect=sum([icMap[x] for x in list(set(e1).intersection(set(e2)))])#Intersecting Concepts        
        sumUnion=sum([icMap[x] for x in list(set().union(e1,e2))])#All Concepts
        entityMatches[k].append(sumIntersect/sumUnion)
    if len(errorList)>0:
        logging.info("   WARNING: No annotations found for: "+"; ".join(list(set(errorList))))
    return entityMatches


def groupwise_MAX(ssm, entityMatches, annotData, pairwise, **args):
    icMap=eval("classIC[\""+str(ssm["IC"])+"\"]")
    errorList=[]
    store={}
    for k,entry in entityMatches.items():
        try:     
            e1=annotData[entry[0]]
            e2=annotData[entry[1]]
        except KeyError as e:
            if str(e) not in errorList:
                errorList.append(str(e))
            entityMatches[k]="NO DATA"
            continue
        b=[get_Pairwise(c1=x,c2=y,icMap=icMap,pairwise=pairwise,store=store,classAncestor=classAncestor) for x,y in list(product(e1,e2))]        
        entityMatches[k].append(max(b))
    return entityMatches    

def groupwise_BMA(ssm,entityMatches,annotData,pairwise,**args):
    """ Computation of BMA groupise semantic similarity
        Requires icMap dict of IC values, entityMatches dict of entity-to-entity combs, dict of entity annotations, pairwise str of pairwise measure
        Returns entityMatches with computed Groupwise Similarity scores
    """
    icMap=eval("classIC[\""+str(ssm["IC"])+"\"]")
    errorList=[]
    store={}
    for k,entry in entityMatches.items():
        try:     
            e1=annotData[entry[0]]
            e2=annotData[entry[1]]
        except KeyError as e:
            if str(e) not in errorList:
                errorList.append(str(e))
            entityMatches[k]="NO DATA"
            continue
        b=[get_Pairwise(c1=x,c2=y,icMap=icMap,pairwise=pairwise,store=store,classAncestor=classAncestor) for x,y in list(product(e1,e2))]
        if len(b)==1: 
            entityMatches[k].append(b[0])
        else:
            groupMatrix=np.array([[b[x:x+len(e2)]] for x in range(0,len(b)-1,len(e2))])
            sim1=mean([max(groupMatrix[x].tolist()[0]) for x in range(0,groupMatrix.shape[0])])
            sim2=mean([max(groupMatrix.transpose()[x].tolist()[0]) for x in range(0,groupMatrix.shape[2])])
            entityMatches[k].append((sim1+sim2)/2)
    return entityMatches



def groupwise_PolGIC(ssm,entityMatches,annotData,**args):
    """ Computation of SimGIC groupise semantic similarity with inclusion of negative annotations
        Requires icMap dict of IC values, entityMatches dict of entity-to-entity combinations, dict of entity annotations
        Returns entityMatches with computed Groupwise Similarity scores
    """
    icMapPos=eval("classIC[\""+str(ssm["IC"])+"\"]")
    icMapNeg=eval("classIC[\"-"+str(ssm["IC"])+"\"]")
    errorList=[]
    for k,entry in entityMatches.items():
        try:
            #Object 1
            e1_P=set(itertools.chain.from_iterable([classAncestor[x] for x in set(annotData["+"+entry[0]])]))
            e1_N=set(itertools.chain.from_iterable([classDescendant[x] for x in set(annotData["-"+entry[0]])]))
            #Object 2
            e2_P=set(itertools.chain.from_iterable([classAncestor[x] for x in set(annotData["+"+entry[1]])]))
            e2_N=set(itertools.chain.from_iterable([classDescendant[x] for x in set(annotData["-"+entry[1]])]))  
        except KeyError as e:
            if str(e) not in errorList:
                logging.info("   WARNING: No data found for object "+str(e))
                errorList.append(str(e))
            entityMatches[k]="NO DATA"
            continue   
        sharedPos=sum([icMapPos[x] for x in list(set(e1_P).intersection(set(e2_P)))])
        sharedNegs=sum([icMapPos[x] for x in list(set(e1_N).intersection(set(e2_N)))])    
        diff=sum([icMapPos[x] for x in list(e1_P.intersection(e2_N).union(e1_N.intersection(e2_P)))])
        union=sum([icMapPos[x] for x in list(set.union(e1_P,e2_P))]+[icMapPos[x] for x in list(set.union(e1_N,e2_N))]+2*[icMapPos[x] for x in list(set.union(e1_N,e2_N))])
        score=((sharedPos+sharedNegs+diff)/union)
        entityMatches[k].append(score)
    return entityMatches


def groupwise_PolBMA(ssm,entityMatches,annotData,pairwise,cPos=0.5,cNeg=0.5,**args):
    """ Computation of BMA groupise semantic similarity with inclusion of negative annotations
        Requires icMap dict of IC values, entityMatches dict of entity-to-entity combs, dict of entity annotations, pairwise str of pairwise measure
        Returns entityMatches with computed Groupwise Similarity scores
    """
    icMapPos=eval("classIC[\""+str(ssm["IC"])+"\"]") 
    store={}
    errorList=[]
    for k,entry in entityMatches.items():
        try:      
            e1_P=list(set(annotData["+"+entry[0]]))
            e1_N=list(set(annotData["-"+entry[0]]))
            e2_P=list(set(annotData["+"+entry[1]]))
            e2_N=list(set(annotData["-"+entry[1]]))  
        except KeyError as e:
            if str(e) not in errorList:#No annotations for an object(an empty list would be returned if there are no +/- annotations)
                logging.info("   WARNING: No data found for object "+str(e)[1:])
                errorList.append(str(e))
            entityMatches[k]="NO DATA"
            continue
        e1=e1_P+e1_N
        e2=e2_P+e2_N    
        pwSims=[get_Pairwise(c1=x,c2=y,icMap=icMapPos,pairwise=pairwise,store=store,classAncestor=classAncestor,polar=False) for x,y in list(product(e1,e2))]    
        if len(pwSims)==0:#one annotations from one prot
            score=0
        elif len(pwSims)==1:#each prot has one annotation
            if e1==e1_N or e2==e2_N:
                if e1==e1_N:
                    if e2==e2_N:
                        score=pwSims[0]#both terms are negative, 
                    else:
                        score=0#One is negative and the other positive
                else:
                    score=-pwSims[0]
            else:
                score=pwSims[0]#both terms are positive
        else:
            scores=[]
            simMatrix=p.DataFrame(np.array([pwSims[x:x+len(e2)] for x in range(0,len(pwSims)-1,len(e2))]),columns=e2,index=e1)#       
            for x in range(0,2):#Select the best matches and if they were positive, negative or mix
                #x=1 if targeting row annotations (from e1)
                #x=0 if targeting column annotations (from e2)
                pairedWith=list(simMatrix.idxmax(axis=x))
                simScores=list(simMatrix.max(axis=x))                         
                if x == 1:
                    PtoP=[simScores[x] for x in range(0,len(e1_P)) if pairedWith[x] in e2_P]
                    NtoN=[simScores[x] for x in range(len(e1_P),len(e1)) if pairedWith[x] in e2_N]    
                else:
                    PtoP=[simScores[x] for x in range(0,len(e2_P)) if pairedWith[x] in e1_P]
                    NtoN=[simScores[x] for x in range(len(e2_P),len(e2)) if pairedWith[x] in e1_N]    
                scores.append((sum(PtoP)+sum(NtoN))/len(simScores))
            score=sum(scores)/2  
        entityMatches[k].append(score)
    return entityMatches



#=====================================================Utility and CLustering==================================================================


def get_scoreMatrix(simScores,targetScore): 
    """Convert a score column from an entity similarity table to a similarity (i.e. affinity) matrix"""
    pivot=simScores.pivot(index='Entity A', columns='Entity B', values=targetScore)
    index = pivot.index.union(pivot.columns)
    #Generate Matrix
    scoreMatrix = pivot.reindex(index=index, columns=index)
    entities=list(scoreMatrix.index)
    #Convert to numpy array to fill upper and diagonal
    scoreMatrix=np.array(scoreMatrix)
    scoreMatrix=np.tril(scoreMatrix)+np.triu(scoreMatrix.T)
    np.fill_diagonal(scoreMatrix,1)
    return scoreMatrix,entities

def get_spectralClustering(scoreMatrix,k,path=None):
    """ Semantic Spectral Clustering
        Requires array scoreMatrix, int k number of clusters, str path (optional)
        Returns cluster labels for objects
    """

    na=len(np.argwhere(np.isnan(scoreMatrix))) 
    if na>0:
        logging.info("WARNING: Similarity Matrix has "+str(na)+" NaN values")
        scoreMatrix[np.isnan(scoreMatrix)] = 0
    scoreMatrix[scoreMatrix < 1e-308] = 0
    
    spectre = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels="discretize")
    spectre.fit_predict(scoreMatrix)

    return spectre.labels_


#============================================================Loader Functions==========================================================

def load_Ontologies(ontologyInfo,reasoning):
    """ Loads ontologies and sets their namespaces
        Requires list ontologyInfo, bool reasoning
        Returns list ontologyList, list ontologyNamespace
    """
    ontologies=list(ontologyInfo)
    ontologyNamespace=[]
    ontologyList=[]
    for onto in ontologies:
        #e.g. if ontologies[0] is "NCIT", then the string "NCIT" is converted to its loader() variable and hence used to refer to the ontology itself
        exec("vars()[\""+onto+"\"]=get_ontology(\""+ontologyInfo[onto]["Path"]+"\").load()")
        ontologyList.append(eval(onto))
        if "Prefix" in ontologyInfo[onto]:
            ontologyNamespace.append(eval(onto).get_namespace(ontologyInfo[onto]['Prefix']))
        else:
            ontologyNamespace.append(eval(onto).base_iri)
    #if True:#reasoning:
        #for onto in ontologyList:
           # logging.info("   > Applying Hermit reasoner to "+str(onto.name)+" ontology")
            #with onto: sync_reasoner()
    #else:
    #    for onto in ontologyList:
    #        if onto.name!='IOBC(edit).xrdf':
    #            logging.info("   > Infering Subclassess in "+ str(onto.name))
     #           subclass_inference(get_nodeList(onto),ontologyNamespace[ontologyList.index(onto)])
    return ontologyList, ontologyNamespace



def load_ClassInfo(ontologyList, ssmIDs, normalize_values,annotData):
    """ Loads classes and computes ancestors, descendants and IC (optional)
        Requires list ontologyList, list ssmIDs, bool normalize_values
        Returns list classes, dict classAncestor, dict classDescendant, dict classIC
    """

    classes=sum([get_nodeList(x) for x in ontologyList], [])+[Thing]#Add Thing
    logging.info("   > Found "+str(len(classes))+" classes")
    classAncestor={c:get_ancestors(c,include_constructs=False) for c in classes}#Include constructs adds more than is_a relationships (needs processing for more fucntionality and to recognize actual classes rather than properties) 
    logging.info("   > Computed ancestors")
    classDescendant={c:get_descendants(c) for c in classes}#Does not work with Thing
    classDescendant[Thing]=classes#All classes including Thing

    #classes=classes+[Thing]#Add again to classes
    logging.info("   > Computed descendants")
   
    icList=list(set(ssmIDs[x]["IC"] for x in ssmIDs if "IC" in ssmIDs[x]))
    if len(icList)>0:
        logging.info("\nComputing ICs")
        classIC = get_IC(classes,icList,normalize_values,classDescendant,classAncestor,annotData)
        return classes, classAncestor, classDescendant, classIC
    else:
        return classes, classAncestor, classDescendant, None

def load_Objects(pathToAnnot,annotData):
    """ Loads objects and connects their annotations with actual ontologies subclasses. If an object has invalid IRIs, these will be ignored.
        Requires str pathToAnnot, dict annotData
        Returns filled dict with annotated objects
    """

    with open(pathToAnnot) as annotFile:
        data = annotFile.readlines()
    polar = True if any("+" in x or "-" in x.split("\t")[1] for x in data) else False #Negative-Positives vs. Normal  
    mismatch=[]
    loaded=0
    for line in data:
        if line == "\n":
            continue
        content=line.rstrip().split("\t")
        try:
            if polar:
                positives,negatives=[],[]
                a=[positives.append(x[1:]) if x.startswith('+') else negatives.append(x[1:]) for x in content[1].split(';')]
                annotData["+"+content[0]]=[IRIS[x] if IRIS[x]!=None else mismatch.append(x) for x in set(positives)]#Positives to +[Object]
                annotData["-"+content[0]]=[IRIS[x] if IRIS[x]!=None else mismatch.append(x) for x in set(negatives)]#Negatives to -[Object]
            else:
                annotData[content[0]]=[(IRIS[x]) for x in content[1].split(';')]
                
        except IndexError:
            logging.info("Found object \""+ content[0]+"\" without valid annotations")
            continue
        except Exception as e:
            logging.info(e)
            break
        loaded+=1
    if any(None in x for x in annotData.values()):#Debbugging
        annotData={k:[x for x in v if x!=None] for k,v in annotData.items()}#Remove Nones
        logging.info("Found "+str(len(set(mismatch)))+" unrecognized annotation(s)")
    logging.info("   > Loaded "+str(loaded)+" objects from "+path)
    return annotData  



    
#============================================================Process Requests============================================================


def SSMC(settings,queries,annotData, classAncestor, classDescendant, classIC):
    """ Process submitted querries, compute SS between concepts or entities, cluster (optional) concepts or querries
        Requires dicts settings with user specifications, querries with all submitted querries, classDescendant and classAncestor with descendants and ancestors of each class, and classIC with computed class ICs
    """

    for x in queries:
        logging.info("\nStarting on querry "+x)
        querryIDs = queries[x]["SSM_ID"].split(",")
        querrySSMs = [settings["Similarity Settings"][y] for y in querryIDs]
    
        with open(queries[x]["Path"]) as querry:
            a=[[num]+line.strip("\n").split() for num, line in enumerate(querry, 0)]
        b={row[0]: row[1:] for row in a}#A dictionary of all combinations of objects or concepts (easier to convert into dataframe)
        header=["Entity A","Entity B"]
        logging.info("   > Loaded query file "+queries[x]["Path"])
        
        for ssm in querrySSMs:#Add possible arguments for a SSM for universal call (spares if conditions)
            for arg in ['IcMap','Pairwise','Double_Objects','cPos','cNeg']:
                if arg not in ssm:
                    ssm[arg]=None
            
            if "Groupwise" in ssm:
                #Groupwise Comparison between objects (handling all at once)
                logging.info("   > Calculating Groupwise Scores using: "+ssm["Groupwise"])
                header.append(ssm["Groupwise"]+"_"+ssm["IC"])
                logging.info(ssm["IC"])
                b=eval("groupwise_"+ssm["Groupwise"])(entityMatches=b,pairwise=ssm["Pairwise"],doubleObjects=ssm["Double_Objects"],ssm=ssm,annotData=annotData,classAncestor=classAncestor,classDescendant=classDescendant)
            else:
                #Pairwsie Comparison between concepts (one at a time)
                logging.info("   > Calculating Pairwise Scores using: "+ssm["Pairwise"])
                header.append("pairwise_"+ssm["Pairwise"])
                for k,entry in b.items():
                    pwScore=eval("pairwise_"+ssm["Pairwise"])(c1=IRIS[entry[0]],c2=IRIS[entry[1]],icMap=eval("classIC[\""+str(ssm["IC"])+"\"]"))
                    b[k].append(pwScore)

    
        #Make Similarity Score File
        b={k:v for k,v in b.items() if v != "NO DATA"}
        simMatrix=np.array(list(b.values()))
        simScores=p.DataFrame(simMatrix,index=np.arange(1, simMatrix.shape[0]+1),columns=np.arange(1, simMatrix.shape[1]+1))
        simScores.columns=header
        for y in header[2:]:
            simScores[y]=simScores[y].astype(float)
        if "Results" in queries[x]:
            simScores.to_csv(queries[x]["Results"])
            logging.info("   > Saved Scores to "+queries[x]["Results"])
    
    #Clustering (only when there is a parwise score bewteen every object)
        if "Clusters" in queries[x]:
            for ssm in queries[x]["Clusters"]["Score"].split(","): 
                #get target header from the index in IDs, retrieve first value and make the name            
                targetScore="_".join([x[1] for x in list(querrySSMs[querryIDs.index(ssm)].items()) if x[1] not in [None, True, False] and x[0]!="Pairwise"])     
                logging.info("   > Generating Similarity Matrix using: "+targetScore)
                scoreMatrix,entities=get_scoreMatrix(simScores,targetScore)#get Matrix from that score and the object index
                logging.info("   > Clustering")
                for k in queries[x]["Clusters"]["K"].split(","):
                    clusterPath=queries[x]["Clusters"]["Cluster Path"]+"\\"+targetScore+"_k"+k+".png"                
                    entityLabels=get_spectralClustering(scoreMatrix,int(k),clusterPath)
    logging.info("\nFinished")


    
with open(settingsPath) as data:
    settings=json.load(data)
logging.info("\nGathering Data from directory: \""+settingsPath+"\"")
ontologyInfo,ssmIDs,queries,normalize_values,reasoning=LoadInfo(settings)

logging.info("\nLoading ontologies")
ontologyList, ontologyNamespace = load_Ontologies(ontologyInfo,reasoning)
    
logging.info("\nLoading object data")
if settings["Object_Data"]:
    annotData={}
    for path in settings["Object_Data"].split(";"):
        annotData=load_Objects(path,annotData)
    
logging.info("\nLoading classes")
classes, classAncestor, classDescendant, classIC = load_ClassInfo(ontologyList,ssmIDs,normalize_values,annotData)


#Call main
SSMC(settings,queries,annotData, classAncestor, classDescendant, classIC)
