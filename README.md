# axiom-acceptability
The files needed to replicate the experiments in this publication

1. available are all the main code files needed to perform a full evaluation.to do so you need to install corese server.
	a. download and install the latest corese server from inria 
	b. edit the profile.ttl file and replace with the name of ontology .owl file you want to run the evaluation on.
	c. edit the main.py file and put the proper paths of corese server and outputs. please check all the paths in the file.

2. if you already have a dataset (an Axiom similarity matrix) you can run the multiclass pridictor py file and replace the pre-filled models or classifiers to compare results.
	a. if you do not have a matrix available, a dbpedia disjoint axiom similarity matrix is provided, all axioms are real. which will provide a real world case study.
	b. in addition 2 files named train.csv and test.csv are manual separations of train and test sets of the same dataset if needed to perform in a separate manualy written code to check the results.

3. for the single reason of testing neural networks a file called keratest.py is also included, this was used to test multiple settings of a NN to perform our classification.
