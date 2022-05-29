from __future__ import print_function
import os
import subprocess
import re
import SPARQLWrapper
import json
import sys
import tqdm
import parmap
import numpy as np
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
import seaborn as sns
print('SPARQLWrapper ver.', SPARQLWrapper.__version__)
from SPARQLWrapper import SPARQLWrapper, JSON, XML
import pandas as pd
print('Pandas ver.', pd.__version__)
from itertools import islice, product
import time
import kernelbuilder
from multiprocessing import Pool
from numba import cuda, jit, prange, vectorize, guvectorize


#parameters that can be edited
def setParam(P_threadcount = 24, P_split = 1000,  P_prefix = 'http://dbpedia.org/ontology/' ,  P_relation = 'owl:disjointWith', P_path = 'fragments/',
             P_corese_path = os.path.normpath(r"C:\Users\ballo\OneDrive - Université Nice Sophia Antipolis\corese\corese-server"), 
             P_rdfminer_path = os.path.normpath(r"C:\Users\ballo\OneDrive - Université Nice Sophia Antipolis\corese\RDFMining-main"),
             P_command_line = 'start /w cmd /k java -jar -Dfile.encoding=UTF8 -Xmx20G corese-server-4.1.5.jar -e -lp -debug -pp profile.ttl', 
             P_wds_Corese = 'http://localhost:8080/sparql', P_label_type = 'c', P_list_of_axioms = None, P_score = None,  P_dont_score = True):
    global threadcount    #number of process for multiprocessing avoid using logical cores
    global split          # divide the table you are working on into tasks, the more processors the more you can divide
    global prefix         #use this to reduce the search time and make thigs more readable
    global relation       #the axiom/relation we are extracting
    global path           #the path of kernel builder should be edited in the .py file itself
    global corese_path    #parameters to launch corese server
    global command_line
    global wds_Corese
    global allrelations  #the whole axiom dataset we are working with
    global label_type    #either classification of regression so either a score or a binary label
    global axiom_type    # disjoint, subclass, equivilent or same as 
    global list_of_axioms
    global score
    global rdfminer_path
    global dont_score
    dont_score = P_dont_score
    threadcount = P_threadcount
    split = P_split
    prefix = P_prefix
    relation = P_relation
    path = P_path
    corese_path = P_corese_path
    command_line = P_command_line
    wds_Corese = P_wds_Corese
    label_type = P_label_type
    rdfminer_path = P_rdfminer_path
    #finish for other axiom types
    if P_relation == 'owl:disjointWith':
        axiom_type = 'DisjointClasses'
    elif P_relation == 'rdfs:subClassOf':
        axiom_type = 'SubClassOf'
    
    list_of_axioms = P_list_of_axioms
    score = P_score
    



# Read a list of axioms, extract unique concepts to use in creating a precise concept similarity matrix (finish axiom types)
def clean_scored_atomic_axioms(labeltype = 'c', axiomtype = "SubClassOf", score_ = None,sample = True):
    
    valid = {'c', 'r'}
    if labeltype not in valid:
        raise ValueError("labeltype must be one of %r." % valid)
           
    scored_axiom_list = pd.read_json( rdfminer_path+'\\IO\\'+ score_)
    scored_axiom_list = scored_axiom_list[['axiom', 'numConfirmations', 'referenceCardinality', 'numExceptions', 'generality', 'possibility', 'necessity']]
    scored_axiom_list = scored_axiom_list.drop_duplicates('axiom')#drop duplicates
    scored_axiom_list = scored_axiom_list[scored_axiom_list.referenceCardinality != 0].reset_index(drop = True)#remove axioms whos concepts have no instances
    scored_axiom_list['left'], scored_axiom_list['right'] = zip(*(s.split(" ") for s in scored_axiom_list.iloc[:, 0].apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')).apply(lambda x: x.replace('SubClassOf','')) ))
    scored_axiom_list = scored_axiom_list[scored_axiom_list.left != scored_axiom_list.right].reset_index(drop = True)
    
    if axiomtype == "DisjointClasses":
        scored_axiom_list['label'] = np.where(scored_axiom_list['numExceptions']/scored_axiom_list['generality'] >= 0.05, 0, 1)# number of exceptions over generality gives most logical results
    else:
        scored_axiom_list = scored_axiom_list[(scored_axiom_list['necessity']/scored_axiom_list['possibility'] - 1 <= -0.2)
                                            & (scored_axiom_list['necessity']/scored_axiom_list['possibility'] - 1 >= 0.2)]# ARI possibility and necessity -1 if between 0.2 and -0.2 drop
        scored_axiom_list['label'] = np.where((scored_axiom_list['necessity']/scored_axiom_list['possibility'] - 1) <= 0, 0, 1)# ARI possibility and necessity -1
        
    
    #sample an equal ammount of negative and positive labels
    if sample:
        scored_axiom_list = sample_dataset(scored_axiom_list)
    
    
    if axiomtype == "DisjointClasses":
        a, b = zip(*(s.split(" ") for s in scored_axiom_list.iloc[:, 0].apply(lambda x: x.replace('DisjointClasses','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    else:
        a, b = zip(*(s.split(" ") for s in scored_axiom_list.iloc[:, 0].apply(lambda x: x.replace('SubClassOf','')).apply(lambda x: re.sub( '[()]','',x)).apply(lambda x: x.replace('<','"')).apply(lambda x: x.replace('>','"')) ))
    
    #list of all unique concepts in our axiom set
    concepts =  pd.Series(pd.Series(np.hstack([a,b])).drop_duplicates().values).sort_values()
    
    a= pd.Series(a).apply(lambda x: x.replace('"',''))
    b = pd.Series(b).apply(lambda x: x.replace('"',''))
    
    # extract the score for regression and a label for classification based on the type of axiom
   
        
    if labeltype == 'c':
        labeled_axioms =  pd.concat([a,b,scored_axiom_list['label']],axis = 1, keys = ["left","right","label"])
    else:
        if axiomtype == "DisjointClasses":
            labeled_axioms =  pd.concat([a,b,1-(scored_axiom_list['numExceptions']/scored_axiom_list['generality'])],axis = 1, keys = ["left","right","label"])# number of exceptions over generality gives most logical results
        else:
            labeled_axioms =  pd.concat([a,b,(scored_axiom_list['necessity']/scored_axiom_list['possibility'] - 1)],axis = 1, keys = ["left","right","label"]) # ARI possibility and necessity -1
    
    #create the list of axioms to be sent in the query
    concept_string = ",".join(concepts) 
    return concepts,concept_string, labeled_axioms




#corese_server = subprocess.Popen(command_line, shell=True, cwd=corese_path)
def sparql_service_to_dataframe(service, query):
    """
    Helper function to convert SPARQL results into a Pandas DataFrame.

    Credit to Ted Lawless https://lawlesst.github.io/notebook/sparql-dataframe.html
    """
    sparql = SPARQLWrapper(service)
    sparql.setMethod('POST')
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = sparql.query()

    processed_results = json.load(result.response)
    cols = processed_results['head']['vars']

    out = []
    for row in processed_results['results']['bindings']:
        item = []
        for c in cols:
            item.append(row.get(c, {}).get('value'))
        out.append(item)

    return pd.DataFrame(out, columns=cols)

#LOAD DATAFRAMES IF AVAILABLE
def readAllFiles():
    tic = time.perf_counter()
    try:
        df = pd.read_csv(path +'classessim.csv')
    except:
        print("classessim isnt saved")
        df = 0
    try:
        allrelations = pd.read_csv(path +'allrelations.csv')
    except:
        print("allrelations isnt saved")
        allrelations= 0
    try:
        kernelmatrix = pd.read_csv( path + 'kernelmatrixpivoted.csv')
    except:
        print("kernelmatrixpivoted isnt saved")
        kernelmatrix = 0
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds")
    return (df,allrelations,kernelmatrix)
    

#build the classes similarity table
def buildClassSim(listofconcepts = None):
    if listofconcepts == None:
        query = '''
    
        select * (kg:similarity(?class1, ?class2) as ?similarity)  where {
        ?class1 a owl:Class
        ?class2 a owl:Class
        filter (!isBlank(?class1)  && !isBlank(?class2))
        }
        ORDER BY DESC (?similarity)
    
        '''
    else:
        query = '''
    
        select * (kg:similarity(?class1, ?class2) as ?similarity)  where {
        ?class1 a owl:Class
        ?class2 a owl:Class
        filter (!isBlank(?class1)  && !isBlank(?class2))
         filter (str(?class1) IN ('''+ listofconcepts + ''') && str(?class2) IN ('''+ listofconcepts + ''') )
        }
        ORDER BY DESC (?similarity)
    
        '''

    #filter (?class1 <= ?class2)
    pd.set_option('display.max_colwidth', None) # if your Pandas version is < 1.0 then use -1 as second parameter, None otherwise
    pd.set_option('display.precision', 5)
    pd.set_option('display.max_rows', 99999999999)

    df = sparql_service_to_dataframe(wds_Corese, query)
    print(df.shape)


    # create the table of similarity between all the classes
    tic = time.perf_counter()
    #reduce prefix size for quicker comparison later
    #df['class1'] = df['class1'].apply(lambda x: x.replace(prefix,'dbo:').replace('http://www.w3.org/2002/07/owl#', 'owl:'))
    #df['class2'] = df['class2'].apply(lambda x: x.replace(prefix,'dbo:').replace('http://www.w3.org/2002/07/owl#', 'owl:'))
    df = df.astype({'similarity': 'float'})
    df = df.pivot_table(columns='class1', index='class2', values='similarity').reset_index()
    df.to_csv( path + 'classessim.csv', index=False)
    print('file classsim created and saved')
    print(df.shape)
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds")
    return df


def buildRelationsExisting(P_relation, P_set_axiom_number):
    N_relation = None
    if P_relation == 'owl:disjointWith':
        N_relation = 'rdfs:subClassOf'
    else:
        N_relation = 'owl:disjointWith'
        
    print("extracting existing axioms and sampling")
    tic = time.perf_counter()
    
    query = '''
    SELECT ?class1 ?class2 ?label WHERE {
    ?class1 a owl:Class
    ?class2 a owl:Class
    ?class1 ''' + P_relation + ''' ?class2
    filter (!isBlank(?class1)  && !isBlank(?class2))
    filter (?class1 != ?class2)
    bind(1.0 as ?label)
    BIND(RAND() AS ?random) .
    } ORDER BY ?random
    LIMIT ''' + str(P_set_axiom_number) + '''
    '''
    
    positiverelations = sparql_service_to_dataframe(wds_Corese, query)
    print("positive relations extracted")
    print(positiverelations.shape)
    
    query = '''
    SELECT ?class1 ?class2 ?label WHERE {
    ?class1 a owl:Class
    ?class2 a owl:Class
    ?class1 ''' + N_relation + ''' ?class2
    filter (!isBlank(?class1)  && !isBlank(?class2))
    filter (?class1 != ?class2)
    bind(0 as ?label)
    BIND(RAND() AS ?random) .
    } ORDER BY ?random
    LIMIT ''' +  str(positiverelations.shape[0]) + ''' 
    '''
    negativeiverelations = sparql_service_to_dataframe(wds_Corese, query)
    print("negativeive relations extracted")
    print(negativeiverelations.shape)
    
    # retrieving the existing axioms that are labeled as accepted
    allrelations = pd.concat([positiverelations, negativeiverelations], axis=0).sample(frac = 1, random_state = 1).reset_index(drop=True)
    print(allrelations.shape)
    
    allrelations = allrelations.astype({'label': 'float'})
    allrelations = allrelations.rename(columns={"class1": "left", "class2": "right"})
    allrelations = sample_dataset(allrelations)
    listofaxioms = axiom_type+"(<"+ allrelations["left"] + "> <" + allrelations["right"] + ">)"
    
    concepts =  pd.Series(pd.Series(np.hstack([allrelations["left"],allrelations["right"]])).drop_duplicates().values).sort_values()
    concepts =  "\"" + concepts.astype(str) + "\"" # just to add " " around the concepts to be considered strings when sent in the queries
    concept_string = ",".join(concepts) 
    print("concepts series")
    print(concepts.shape)
    
    
    
    print(allrelations.shape)
    listofaxioms.to_csv(rdfminer_path+'\IO\listofaxioms.txt', sep=',', index=False, header = False)
    #allrelations.to_csv( path +'allrelations.csv', index=False)
    print('allrelations created')
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds to extract the axioms and concepts")
    
    return allrelations, concept_string, concepts
  
  
#build the atomic relations table with random generated false axioms
def buildRelationsGenerated(P_relation,P_set_axiom_number ):
    N_relation = None
    if P_relation == 'owl:disjointWith':
        N_relation = 'rdfs:subClassOf'
    else:
        N_relation = 'owl:disjointWith'
        
    print("extracting existing axioms and sampling")
    tic = time.perf_counter()
    
    query = '''
    SELECT ?class1 ?class2 ?label WHERE {
    ?class1 a owl:Class
    ?class2 a owl:Class
    ?class1 ''' + P_relation + ''' ?class2
    filter (!isBlank(?class1)  && !isBlank(?class2))
    filter (?class1 != ?class2)
    bind(1.0 as ?label)
    BIND(RAND() AS ?random) .
    } ORDER BY ?random
    LIMIT ''' + str(P_set_axiom_number) + '''
    '''
    
    positiverelations = sparql_service_to_dataframe(wds_Corese, query)
    print("positive relations extracted")
    print(positiverelations.shape)
    
    query = '''
    SELECT ?class1 ?class2 ?label WHERE {
    ?class1 a owl:Class
    ?class2 a owl:Class
    ?class1 ''' + N_relation + ''' ?class2
    filter (!isBlank(?class1)  && !isBlank(?class2))
    filter (?class1 != ?class2)
    bind(0 as ?label)
    BIND(RAND() AS ?random) .
    } ORDER BY ?random
    LIMIT ''' +  str(positiverelations.shape[0]) + ''' 
    '''
    negativeiverelations = sparql_service_to_dataframe(wds_Corese, query)
    print("negativeive relations extracted")
    print(negativeiverelations.shape)
    
    # retrieving the existing axioms that are labeled as accepted
    allrelations = pd.concat([positiverelations, negativeiverelations], axis=0).sample(frac = 1, random_state = 1).reset_index(drop=True)
    print(allrelations.shape)
    
    allrelations = allrelations.astype({'label': 'float'})
    allrelations = allrelations.rename(columns={"class1": "left", "class2": "right"})
    allrelations = sample_dataset(allrelations)
    listofaxioms = axiom_type+"(<"+ allrelations["left"] + "> <" + allrelations["right"] + ">)"
    
    print(allrelations.shape)
    listofaxioms.to_csv(rdfminer_path+'\IO\listofaxioms.txt', sep=',', index=False, header = False)
    allrelations.to_csv( path +'allrelations.csv', index=False)
    print('file allrelations created and saved')
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds")
    return allrelations



def sample_dataset(labeled_axioms):
    
    positiverelations = labeled_axioms[labeled_axioms["label"] == 1]
    #sample the same number of negative relations as positive ones
    negativerelations= labeled_axioms[labeled_axioms["label"] == 0]
    
    #uncomment this
    if len(positiverelations)>= len(negativerelations):
          labeled_axioms = labeled_axioms.groupby("label").sample(n=len(negativerelations), random_state=1)
    else:
          labeled_axioms = labeled_axioms.groupby("label").sample(n=len(positiverelations), random_state=1)
        
    
    # # #shuffle    
    labeled_axioms = labeled_axioms.sample(frac = 1, random_state = 1).reset_index(drop = True)
    
    #delete this
    #labeled_axioms = negativerelations.sample(n = 5000, random_state = 5).reset_index(drop = True)
    #labeled_axioms = positiverelations
    return labeled_axioms




#build the atomic relations table with real axioms and their compliments
def buildRelationsCompliment():
    tic = time.perf_counter()
    query = '''

    select ?class1 ?class2 ?I where {

    ?class1 a owl:Class
    ?class2 a owl:Class
    ?class1 ''' + relation +'''?class2
    filter (!isBlank(?class1)  && !isBlank(?class2))
    bind(1.0 as ?I)

    }

    '''

    # generating all possible combinations of 2 atomic classes to create the false axioms with -1 p index
    allrelations = sparql_service_to_dataframe(wds_Corese, query)
    print(allrelations.shape)
    
    #allrelations['class1'] = allrelations['class1'].apply(lambda x: x.replace(prefix,'dbo:').replace('http://www.w3.org/2002/07/owl#', 'owl:'))
    #allrelations['class2'] = allrelations['class2'].apply(lambda x: x.replace(prefix,'dbo:').replace('http://www.w3.org/2002/07/owl#', 'owl:'))
    
    #sampling the negative synthasized axioms
    #allrelations = allrelations.astype({'I': 'float'})
    print(allrelations.shape)
    

    #create the compliments

    print(allrelations.shape)

    allrelations.to_csv( path +'allrelations.csv', index=False)
    print('file allrelations created and saved')
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds")
    return allrelations



#function used to calculate the kernel matrix can be used with fractions of the relationship table, should be written in a .py
#file and imported in this module, please change the path in  kernelbuilder.py if you wish to output csv files somewhere else

def matrixfractionAverageSim(start, end, size, df, allrelations):#the column name for the first column is class2
    rowlist = []
    for i in range(start, end):
        axiom1 = allrelations.iloc[i]
        a1_c1 =  axiom1['class1']
        a1_c2 = axiom1['class2']
        for j in range(i, size):
            axiom2 = allrelations.iloc[j]
            sim1 = df.loc[df['class2'] == a1_c1, axiom2['class1']].values[0]
            sim2 = df.loc[df['class2'] == a1_c2, axiom2['class2']].values[0]
            rowlist.append([i, j, (sim1+sim2)/2])
    axiomsimilaritymatrix = pd.DataFrame(rowlist, columns=["axiom1", "axiom2", "overallsim"])
    return axiomsimilaritymatrix

# preparing to split work load to multiple threads
def splitload(split):
    size = len(allrelations)
    portion = size//split
    startend = []
    l = 0
    for x in range(1,split):
        startend.append((l,l+portion))
        l += portion
    startend.append((startend[len(startend)-1][1], size))
    print("split completed")
    return(startend)



#pivot the table into a matrix and output that to a csv, prepare it for mulearn

def pivotIntofinalmatrix(kernelmatrix):
    tic = time.perf_counter()

    kernelmatrix = kernelmatrix.pivot_table(columns='axiom1', index='axiom2', values='overallsim',  fill_value=0).reset_index()
    kernelmatrix.drop(columns = ["axiom2"], inplace = True)
    rawmatrix = kernelmatrix.to_numpy()
    rawmatrix = rawmatrix + rawmatrix.T - np.diag(np.diag(rawmatrix))
    #added <> and axiom type to axioms to comply with rdf miner fromat
    colnames = axiom_type+"(<"+ allrelations["left"] + "> <" + allrelations["right"] + ">)"
    kernelmatrix = pd.DataFrame(data=rawmatrix, columns = colnames)
    kernelmatrix.insert(0,'possibility',allrelations['label'])
    print("kernelmatrix shape is :") 
    print(kernelmatrix.shape)
    kernelmatrix.to_csv( path + 'kernelmatrix.csv', sep=',', index=False)
    print('file kernelmatrix built and saved')
    toc = time.perf_counter()
    print(f"it took {toc - tic:0.4f} seconds")
    return kernelmatrix



# WARNING CHANGE THE SLEEP TIMER AFTER LAUNCHING CORESE IF YOU ARE HAVING AN ERROR WHEN THE OWL FILE IS LARGE, 50 SECONDS IS GOOD FOR 200 MB FILES, 10 IS GOOD FOR 8 MB
#calls for multiprocessing to build the table of axiom similarity
if __name__ == '__main__':
    
    ticfirst = time.perf_counter()
    #Call all functions for a full run
    
    #end version, every parameter that can be changed should be here
    setParam(P_threadcount = 24, P_split =1500,  P_prefix = '' ,  P_relation = 'owl:disjointWith', P_path = 'fragments/',
             P_corese_path = os.path.normpath("C:\corese-server"),
             P_rdfminer_path = os.path.normpath(r"C:\Users\ballo\OneDrive - Université Nice Sophia Antipolis\corese\RDFMining-main"),
             P_command_line = 'start /w cmd /k java -jar -Dfile.encoding=UTF8 -Xmx20G corese-server-4.1.5.jar -e -lp -debug -pp profile.ttl', 
             P_wds_Corese = 'http://localhost:8080/sparql', P_label_type='c', P_list_of_axioms= None, P_score = 'resultsresults.json', P_dont_score = True)
    
    #uncomment if you already have the files and want to read them, process is fast though so you can just create them again
    #df, allrelations, kernelmatrix = readAllFiles()
    
    #prepare to launch corese server
    corese_server = subprocess.Popen(command_line, shell=True, cwd=corese_path)
    #CHANGE THIS TIMER IN CASE OF ERRORS
    time.sleep(10)
    set_axiom_number = 500
    
    #if statement that chooses the way axioms are genarated and scored
    if list_of_axioms == None and score == None :  #randomly generate atomic axioms (NOT RECOMMENDED FOR LARGE ONTOLOGIES 700+ concepts)
        if label_type == 'c' and dont_score == True:# dont score the random generated axioms
            print('generating list from existing without scoring')
            df = buildClassSim()
            allrelations = buildRelationsGenerated(relation, set_axiom_number)# create a list of axioms and send it to /rdfminer/io/ shared folder on ur machine
        elif label_type == 'c' and dont_score == True and relation == 'owl:disjointWith':
            print('generating list from existing and random combination without scoring but not all combinations')
            
            
            
            allrelations = buildRelationsGenerated(relation, set_axiom_number)
            concepts, concept_string, allrelations = clean_scored_atomic_axioms(label_type, axiom_type, score)
            df = buildClassSim(concept_string)
            
            
            
        else:# score the generated list of axioms
            print('generating list from existing and random combination with scoring')
            allrelations = buildRelationsGenerated(relation)# create a list of axioms and send it to /rdfminer/io/ shared folder on ur machine
            list_of_axioms ='listofaxioms.txt'
            rdfminer = subprocess.run('start /w docker-compose exec rdfminer ./rdfminer/scripts/run.sh -a /rdfminer/io/'+ list_of_axioms +' -dir results', shell=True, cwd=rdfminer_path) #process of scoring with rdf miner
            score = 'resultsresults.json'
            concepts, concept_string, allrelations = clean_scored_atomic_axioms(label_type, axiom_type, score, sample = False)
            df = buildClassSim(concept_string)      
    elif list_of_axioms != None and score == None: # use a list of axioms not scored
        print('using a non scored list of axioms')
        rdfminer = subprocess.run ('start /w docker-compose exec rdfminer ./rdfminer/scripts/run.sh -a /rdfminer/io/'+ list_of_axioms +' -dir results', shell=True, cwd=rdfminer_path) #process of scoring with rdf miner
        score = 'resultsresults.json'
        concepts, concept_string, allrelations = clean_scored_atomic_axioms(label_type, axiom_type, score)
        df = buildClassSim(concept_string)
    else:# use a scored list of axioms
        print('using a scored list of axioms')
        concepts, concept_string, allrelations = clean_scored_atomic_axioms(label_type, axiom_type, score)
        df = buildClassSim(concept_string)



    #split load into multiple processes and list of axioms into chunks
    startend = splitload(split)
    size = len(allrelations)
    p = Pool(threadcount)
    tocfirst = time.perf_counter()
    print(f"it took {tocfirst - ticfirst:0.4f} seconds")
    print()
    
    
    tic = time.perf_counter()
    if axiom_type == 'DisjointClasses':
        print('mirror compare')
        kernelmatrix = pd.concat(parmap.starmap(kernelbuilder.matrixfractionAverageSimdis,startend,size, df, allrelations, pm_pool=p, pm_pbar=True),ignore_index = True)
    #similarity is averag
    else:   
        kernelmatrix = pd.concat(parmap.starmap(kernelbuilder.matrixfractionAverageSim,startend,size, df, allrelations, pm_pool=p, pm_pbar=True),ignore_index = True)
    
    
    p.close()
    p.terminate()
    p.join()
    
    #turn the list into a matrix
    kernelmatrix = pivotIntofinalmatrix(kernelmatrix)
    
    
    toc = time.perf_counter()
    print(f"everything took {toc - tic:0.4f} seconds")

