from __future__ import print_function
import os
import subprocess
import SPARQLWrapper
import json
import sys
import numpy as np
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
import seaborn as sns
from SPARQLWrapper import SPARQLWrapper, JSON, XML
import pandas as pd
from itertools import islice
import time



#function used to calculate the kernel matrix can be used with fractions of the relationship table, should be written in a .py
#file and imported in this module

def sparql_service_to_dataframe(service, query):
    """
    Helper function to convert SPARQL results into a Pandas DataFrame.

    Credit to Ted Lawless https://lawlesst.github.io/notebook/sparql-dataframe.html
    """
    sparql = SPARQLWrapper(service)
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



def matrixfractionold(start, end, size):
    path = 'c:/Users/Ali/Desktop/fragments/'
    df = pd.read_csv(path+'classessim.csv')
    allrelations = pd.read_csv(path+'allrelations.csv')
    axiomsimilaritymatrix = pd.DataFrame({"axiom1": [],"axiom1I": [], "axiom2": [],"axiom2I": [], "a1c1": [],
                                          "a1c2": [],"a2c1": [],"a2c2": [],"sim1": [],"sim2": [],"overallsim": []})
    for i, axiom1 in islice(allrelations.iterrows(),start,end,1):
        for j, axiom2 in islice(allrelations.iterrows(),i, size, 1):
                sim1 =float(df.loc[((df['class1']== axiom1['class1']) & (df['class2']== axiom2['class1']))
                                      |((df['class1']== axiom2['class1']) & (df['class2']== axiom1['class1']))
                                       , 'similarity'].values[0])
                sim2= float(df.loc[((df['class1']== axiom1['class2']) & (df['class2']== axiom2['class2']))
                                      |((df['class1']== axiom2['class2']) & (df['class2']== axiom1['class2']))
                                           , 'similarity'].values[0])
                axiomsimilaritymatrix = axiomsimilaritymatrix.append(
                                    {"axiom1": i,"axiom1I": axiom1['I'], "axiom2": j,"axiom2I": axiom2['I'], "a1c1": axiom1['class1'],
                                    "a1c2": axiom1['class2'],"a2c1": axiom2['class1'],"a2c2": axiom2['class2'],
                                    "sim1": sim1,"sim2": sim2,"overallsim":min(sim1,sim2)}
                                    ,ignore_index=True)
    axiomsimilaritymatrix.to_csv( path + str(start) + '.csv', sep=',', index=False)
    return axiomsimilaritymatrix


def matrixfractionVerbose(start, end, size):
    path = 'c:/Users/Ali/Desktop/fragments/'
    df = pd.read_csv(path + 'classessim.csv')
    allrelations = pd.read_csv(path + 'allrelations.csv')
    rowlist = []
    for i in range(start, end):
        axiom1 = allrelations.iloc[i]
        for j in range(i, size):
            axiom2 = allrelations.iloc[j]
            sim1 = df.loc[df['class2'] == axiom1['class1'], axiom2['class1']].values[0]
            sim2 = df.loc[df['class2'] == axiom1['class2'], axiom2['class2']].values[0]
            rowlist.append([i, axiom1['I'], j, axiom2['I'], axiom1['class1'],
                         axiom1['class2'], axiom2['class1'], axiom2['class2'], sim1, sim2, min(sim1, sim2)])
    axiomsimilaritymatrix = pd.DataFrame(rowlist, columns=["axiom1", "axiom1I", "axiom2", "axiom2I", "a1c1",
                                                        "a1c2", "a2c1", "a2c2", "sim1", "sim2", "overallsim"])
    axiomsimilaritymatrix.to_csv(path + str(start) + '.csv', sep=',', index=False)
    return axiomsimilaritymatrix




def matrixfraction(start, end, size, df, allrelations):
    rowlist = []
    for i in range(start, end):
        axiom1 = allrelations.iloc[i]
        a1_c1 =  axiom1['left']
        a1_c2 = axiom1['right']
        for j in range(i, size):
            axiom2 = allrelations.iloc[j]
            sim1 = df.loc[df['class2'] == a1_c1, axiom2['left']].values[0]
            sim2 = df.loc[df['class2'] == a1_c2, axiom2['right']].values[0]
            rowlist.append([i, j, min(sim1, sim2)])
    axiomsimilaritymatrix = pd.DataFrame(rowlist, columns=["axiom1", "axiom2", "overallsim"])
    return axiomsimilaritymatrix






def matrixfractionAverageSim(start, end, size, df, allrelations):#the column name for the first column is class2
    rowlist = []
    for i in range(start, end):
        axiom1 = allrelations.iloc[i]
        a1_c1 =  axiom1['left']
        a1_c2 = axiom1['right']
        for j in range(i, size):
            axiom2 = allrelations.iloc[j]
            sim1 = df.loc[df['class2'] == a1_c1, axiom2['left']].values[0]
            sim2 = df.loc[df['class2'] == a1_c2, axiom2['right']].values[0]
            rowlist.append([i, j, (sim1+sim2)/2])
    axiomsimilaritymatrix = pd.DataFrame(rowlist, columns=["axiom1", "axiom2", "overallsim"])
    return axiomsimilaritymatrix






def matrixfractionAverageSimdis(start, end, size, df, allrelations):#the column name for the first column is class2
    rowlist = []
    for i in range(start, end):
        axiom1 = allrelations.iloc[i]
        a1_l =  axiom1['left']
        a1_r = axiom1['right']
        for j in range(i, size):
            axiom2 = allrelations.iloc[j]
            #because in disjointness left and right dont make a difference, we compare as dis(A B) dis(B A) and dis(B A) dis(B A)
            sim1 = df.loc[df['class2'] == a1_l, axiom2['left']].values[0]
            sim2 = df.loc[df['class2'] == a1_r, axiom2['right']].values[0]
            
            sim3 = df.loc[df['class2'] == a1_l, axiom2['right']].values[0]
            sim4 = df.loc[df['class2'] == a1_r, axiom2['left']].values[0]
            if (sim1+sim2)/2 > (sim3+sim4)/2:
                sim = (sim1+sim2)/2
            else:
                sim = (sim3+sim4)/2
            rowlist.append([i, j, sim])
    axiomsimilaritymatrix = pd.DataFrame(rowlist, columns=["axiom1", "axiom2", "overallsim"])
    return axiomsimilaritymatrix















def matrixfractionl(start, end, size):
    wds_Corese = 'http://localhost:8080/sparql'
    df = pd.read_csv('classessim.csv')
    allrelations = pd.read_csv('allrelations.csv')
    rowlist = []

    for i in range(start, end):
        axiom1 = allrelations.iloc[i]
        for j in range(i, size):
            axiom2 = allrelations.iloc[j]
            sim1 = df.loc[df['class2'] == axiom1['class1'], axiom2['class1']].values[0]
            sim2 = df.loc[df['class2'] == axiom1['class2'], axiom2['class2']].values[0]
            rowlist.append([i, axiom1['I'], j, axiom2['I'], axiom1['class1'],
                         axiom1['class2'], axiom2['class1'], axiom2['class2'], sim1, sim2, min(sim1, sim2)])

    #path = 'c:/Users/Ali/OneDrive/Desktop/corese/fragments/'

    #axiomsimilaritymatrix.to_csv(path + str(start) + '.csv', sep=',', index=False)
    return rowlist









def matrixfractionq(start, end, size):
    wds_Corese = 'http://localhost:8080/sparql'
    df = pd.read_csv('classessim.csv')
    allrelations = pd.read_csv('allrelations.csv')
    axiomsimilaritymatrix = pd.DataFrame({"axiom1": [],"axiom1I": [], "axiom2": [],"axiom2I": [], "a1c1": [],
                                          "a1c2": [],"a2c1": [],"a2c2": [],"sim1": [],"sim2": [],"overallsim": []})
    for i in range(start, end):
        axiom1 = allrelations.iloc[i]
        for j in range(i, size):
                axiom2 = allrelations.iloc[j]
                query1 = '''

                        select (kg:similarity(?class1, ?class2) as ?similarity)  where {
                        ?class1 a owl:Class
                        ?class2 a owl:Class
                        filter regex(?class1, ''' + '\'' + axiom1['class1'] + '\'' +''')
                        filter regex(?class2, ''' + '\'' + axiom2['class1'] + '\'' +''')
                        }
                        ORDER BY DESC (?similarity)

                        '''
                query2 = '''

                        select (kg:similarity(?class1, ?class2) as ?similarity)  where {
                        ?class1 a owl:Class
                        ?class2 a owl:Class
                        filter regex(?class1, ''' + '\'' + axiom1['class2'] + '\'' +''')
                        filter regex(?class2, ''' + '\'' + axiom2['class2'] + '\'' +''')
                        }
                        ORDER BY DESC (?similarity)

                        '''
                pd.set_option('display.max_colwidth',None)  # if your Pandas version is < 1.0 then use -1 as second parameter, None otherwise
                pd.set_option('display.precision', 5)
                pd.set_option('display.max_rows', 9999999999)

                sim1 = float(sparql_service_to_dataframe(wds_Corese, query1).iloc[0][0])
                sim2 = float(sparql_service_to_dataframe(wds_Corese, query2).iloc[0][0])

                #sim1 =float(df.loc[((df['class1']== axiom1['class1']) & (df['class2']== axiom2['class1']))
                #                      |((df['class1']== axiom2['class1']) & (df['class2']== axiom1['class1'])), 'similarity'].values[0])
                #sim2= float(df.loc[((df['class1']== axiom1['class2']) & (df['class2']== axiom2['class2']))
                #                     |((df['class1']== axiom2['class2']) & (df['class2']== axiom1['class2'])), 'similarity'].values[0])
                axiomsimilaritymatrix = axiomsimilaritymatrix.append(
                                    {"axiom1": i,"axiom1I": axiom1['I'], "axiom2": j,"axiom2I": axiom2['I'], "a1c1": axiom1['class1'],
                                    "a1c2": axiom1['class2'],"a2c1": axiom2['class1'],"a2c2": axiom2['class2'],
                                    "sim1": sim1,"sim2": sim2,"overallsim":min(sim1,sim2)}
                                    ,ignore_index=True)

    path = 'c:/Users/Ali/OneDrive/Desktop/corese/fragments/'
    axiomsimilaritymatrix.to_csv( path + str(start) + '.csv', sep=',', index=False)
    return axiomsimilaritymatrix