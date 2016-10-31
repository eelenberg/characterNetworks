# characterNetworks Readme

## novelPreprocessing.py

Preprocesses graph data from a character network.

Input: csv file of weighted edges labelled with character names
Output: 2 text files labelled with nonnegative integers, one with a column for weights and one without

## generateGraphsMain.py

Generates graphs according to Preferential Attachment, Erdos Renyi, Chung Lu, or Configuration Model.

Input: dict of parameters - graph name, type of model, number of nodes, degree, number of graphs
Writes graphs to file as a edge lists with the format [graphName]_[graphType]_[number].txt

Example parameters:

	params = {'graph': 'thestand','type':'PA','n': 48,'d': int(2*351/48),'numGen': 100}
	generateGraphs(params)
	params = {'graph': 'thestand','type':'CNFG','n': 39,'dList': degreeVec,'numGen': 100}
	generateGraphs(params)

## novelAnalysis.py

Main script to analyze character networks. Calls the function generateGraphs to make 100 random graphs from each model. Separate flags to generate graphs, perform ML classification, and include eigenvalue histograms as features in addition to graph profile features. Options for SVM-L1, SVM-L2, AdaBoost, Decision Tree, and Random Forest classifiers. Decision Tree option also saves an example tree as a pdf. 

	python novelAnalysis.py
