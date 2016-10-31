import numpy as np
import pandas as pd
from itertools import combinations
import io
from snap import GenPrefAttach,SaveEdgeList,TRnd
from sklearn import svm, base, cross_validation, feature_selection, linear_model
from sklearn.metrics import roc_curve,auc,classification_report,f1_score,accuracy_score,roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import subprocess
from generateGraphsMain import *
from os import listdir
#to visualize
from sklearn.externals.six import StringIO
import pydot



def doSVMcvPrediction(Xin,yin,Xtest,ytest,modType):
    #input: training/test data and labels, model type
    #supported models: SVM (l1 and l2), AdaBoost, decision tree, and random forest
    #train with 5-fold cross validation, then test once using test (holdout) data
    #once the best estimator is chosen here, train on the entire dataset (in + test) outside this function
    #output: training accuracy, gereralization accuracy, feature weights/importances, classifier, 
    #  classification report, training f1-score and generalization f1-score
    nfolds = 5
    cv = cross_validation.StratifiedKFold(yin,nfolds,shuffle=True)
    #l1 penalty enforces sparsity in weights, only available for linear SVM classifier
    if modType in ('SVM-L2','svm-l2'):
        clasf = svm.LinearSVC(loss='squared_hinge', penalty='l2', tol=.001, dual=False, class_weight='balanced')
        cvclasf = GridSearchCV(clasf, param_grid = {
            'C' : [0.05, 0.1, 0.5, 1, 5, 10, 500, 1000]
            }, verbose=0,refit=True,
            cv=cv,
            # scoring='roc_auc',
            scoring='f1_weighted',
        n_jobs=4)

    elif modType in ('SVM-L1','svm-l1'):
        clasf = svm.LinearSVC(loss='squared_hinge', penalty='l1', tol=.001, dual=False, class_weight='balanced')
        cvclasf = GridSearchCV(clasf, param_grid = {
            'C' : [0.05, 0.1, 0.5, 1, 5, 10, 500, 1000]
            }, verbose=0,refit=True,
            cv=cv,
            scoring='f1_weighted',
        n_jobs=4)
    
    #decision tree classifiers
    elif modType in ('ada','adaboost','adaboost-tree'):
        clasf = AdaBoostClassifier()
        cvclasf = GridSearchCV(clasf, param_grid = {
            'n_estimators' : [5,10,25,50,100],
            'learning_rate' : [0.1,0.3,0.5]
            }, verbose=0,refit=True,
            cv=cv,
            scoring='f1_weighted',
        n_jobs=4)

    elif modType in ('dtree','decision-tree'):
        clasf = DecisionTreeClassifier()
        cvclasf = GridSearchCV(clasf, param_grid = {
            'splitter' : ['best'],
            'criterion' : ['entropy','gini'],
            'max_features' : [0.2,'sqrt',1.],
            'max_depth' : [2,4], 
            'class_weight' : ['balanced'], 
            }, verbose=0,refit=True,
            cv=cv,
            scoring='f1_weighted',
        n_jobs=4)

    elif modType in ('rf','random-forest'):
        clasf = RandomForestClassifier()
        cvclasf = GridSearchCV(clasf, param_grid = {
            'n_estimators' : [5,10,25,50,100],
            'criterion' : ['entropy','gini'],
            'max_features' : [0.2,'sqrt',1.],
            'max_depth' : [2,4], 
            'class_weight' : ['balanced'], 
            }, verbose=0,refit=True,
            cv=cv,
            scoring='f1_weighted',
        n_jobs=4)
        
    #TODO: add linear regression, logistic regression, etc. 

    cvclasf.fit(Xin,yin)
    bclasf = cvclasf.best_estimator_
    print "%s %d-fold CV params: %s" % (modType,nfolds,cvclasf.best_params_)
    
    if modType in ('ada','adaboost-tree','dtree','decision-tree','rf','random-forest'):
        w = bclasf.feature_importances_
    elif modType in ('SVM-L1','svm-l1','SVM-L2','svm-l2'):
        w = bclasf.coef_
    
    bclasf.fit(Xin,yin)
    y_train_pred = bclasf.predict(Xin)
    acTrain = accuracy_score(yin,y_train_pred)
    f1Train = f1_score(yin,y_train_pred,average="weighted")
    
    y_pred = bclasf.predict(Xtest)
    report = classification_report(ytest, y_pred)
    acGeneral = accuracy_score(ytest, y_pred)
    f1Gen = f1_score(ytest,y_pred,average="weighted")

    return(acTrain,np.squeeze(w),bclasf,report,(acTrain,acGeneral),(f1Train,f1Gen))

def initializeDirectory(origGraph):
    #compute graph profile features using GraphLab PowerGraph
    print "initializing directory and taking features of original graph..."
    out = subprocess.check_output(["mkdir", "graphs"])
    hdr = "#graph\tsample_prob_keep\tn3_3\tn3_2\tn3_1\tn3_0\tn4_0\tn4_1\tn4_2\tn4_3\tn4_4\tn4_5\tn4_6\tn4_7\tn4_8\tn4_9\tn4_10\truntime\n"
    with open('counts_4_profilesLocal.txt', 'w') as fpt:
        fpt.write(hdr)
    pcommand = '/Users/ethan/graphlab-master-2/release/apps/4-profiles/4profile'
    out = subprocess.check_output([pcommand, "--format", "tsv", "--graph", origGraph])
    hdr2 = "#graph\tevbin0\tevbin1\tevbin2\tevbin3\tevbin4\n"
    with open('counts_eval_bins.txt', 'w') as fpt:
        fpt.write(hdr2)
    generateEigenvalueBins(origGraph,"counts_eval_bins.txt")
    return 0

def generateEigenvalueBins(gname,outDir,nbins=5):
    #get normalized laplacian
    hbins = np.histogram(np.array([0,2]),bins=nbins)[1]
    E = np.loadtxt(gname,delimiter='\t')
    #map everything to number of unique vertices
    un = np.unique(np.vstack((E[:,0],E[:,1])))
    n = len(un)
    A = np.zeros((n,n))
    for e in np.arange(E.shape[0]):
        tmp0 = np.argwhere(un==E[e,0])
        tmp1 = np.argwhere(un==E[e,1])
        A[tmp0,tmp1] = 1
        A[tmp1,tmp0] = 1
    D = np.diag(np.sum(A,1))
    Di = np.linalg.inv(np.sqrt(D))
    L = np.eye(n) - Di.dot(A).dot(Di)
    teigs = np.linalg.eigvalsh(L)
    #take histogram
    neig = len(teigs.flatten()) + nbins
    nh = np.histogram(teigs,bins=hbins)[0]    
    ep = (nh + 1.)/neig #add smoothing and normalize
    #append to file
    with open(outDir, "a") as myfile:
        myfile.write(gname + "\t" +  "\t".join([str(e) for e in ep]) + "\n")
    return 0

def writeTree(treeModel,namesList,filename):
    #utility function that plots a decision tree and saves to file
    dot_data = StringIO()
    export_graphviz(treeModel,out_file=dot_data,feature_names=namesList)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(filename) 

if __name__ == '__main__':
    
    genData = 1 #flag to generate random graphs
    classify = 0 #flag to classify
    useSpectral = 0 #flag to include eigenvalue histograms

    if genData:

        print "generating random graphs..."
        # #generate graphs, these parameters should be easy to automate
        # n,E,filename,outname = 27,123,'./twilightEdgesIDsWeights.txt','graphs/twilight'
        # n,E,filename,outname = 27,1031,'./twilightEdgesIDsWeights.txt','graphs/twilight'
        n,E,filename,outname = 39,280,'./thestandEdgesIDsWeights.txt','graphs/thestand'
        # n,E,filename,outname = 39,6539,'./thestandEdgesIDsWeights.txt','graphs/thestand'
        # n,E,filename,outname = 62,575,'./gobletEdgesIDsWeights.txt','graphs/goblet'
        # n,E,filename,outname = 62,9464,'./gobletEdgesIDsWeights.txt','graphs/goblet'
        
        #automate initializing directory
        #this will throw an error if there is already a folder named graphs
        initializeDirectory(filename)

        A = np.loadtxt(filename,delimiter='\t')
        degreeVec = getDegreeList(A)

        params = {'graph': outname,'type':'CNFG','n': n,'dList': degreeVec,'numGen': 100}
        generateGraphs(params)
        params = {'graph': outname,'type':'CL','n': n,'dList': degreeVec,'numGen': 100}
        generateGraphs(params)
        
        # params = {'graph': outname,'type':'PA','n': n,'d': int(2*E/n),'numGen': 50}
        params = {'graph': outname,'type':'PA','n': n,'d': int(2*E/n),'numGen': 100}
        generateGraphs(params)
        params = {'graph': outname,'type':'GNP','n': n,'d': int(2*E/n),'numGen': 100}
        generateGraphs(params)
        # #generate 100 graphs each from 3 classes in < 3 seconds
        #thresholded PA model
       

        graph_dir = '/Users/ethan/Documents/novels/graphs/'
        #feature design 1
        #analyze global 4-profiles
        #separate by generative model?
        # /Users/ethan/graphlab-master-2/release/apps/4-profiles/4profile --format tsv --graph /Users/ethan/Documents/novels/twilightEdgesIDs.txt
        print "computing global subgraph counts..."
        pcommand = '/Users/ethan/graphlab-master-2/release/apps/4-profiles/4profile'
        for gname in listdir(graph_dir):
            if gname.endswith('.txt') and "_wt" not in gname:
            # if "_CL_" in gname:
                # out = subprocess.check_output([pcommand, "--format", "tsv", "--graph", gname, "--per_vertex", gname])
                out = subprocess.check_output([pcommand, "--format", "tsv", "--graph", graph_dir + gname])
                if useSpectral:
                    generateEigenvalueBins(graph_dir+gname,"counts_eval_bins.txt")
        out = subprocess.check_output(["mv", "counts_4_profilesLocal.txt", "graphs"])
        out = subprocess.check_output(["mv", "counts_eval_bins.txt", "graphs"])
        #global 4 profiles for 100 graphs each from 3 classes in ~1 minute

        # additional feature design could include distributions of local 4-profiles throughout graph
        # or pagerank or centrality measures
        
    if classify:
        #build classifiers
        #as a baseline, split data into train and test 
        #this will verify the classifier can differentiate between graph families
        
        np.random.seed(423322) #for repeatability during writeup
        graphFolder = 'graphsGoblet/'
        # graphFolder = 'graphsTwilight/'
        # graphFolder = 'graphsTheStand/'
        print "Reading data from folder %s" % graphFolder

        #read data from 4-profile output file, read labels and split into train and test
        featInds = np.arange(2,17)
        D = pd.read_csv(graphFolder + 'counts_4_profilesLocal.txt',delimiter='\t')
        X = np.array(D.ix[1:,featInds])
        # print X[0,:]
        #add eigenvalue histogram to X
        if useSpectral:
            #assume its in the exact same order
            featInds2 = np.arange(1,6) #5 bins
            D2 = pd.read_csv(graphFolder + 'counts_eval_bins.txt',delimiter='\t')
            X = np.hstack((X,np.array(D2.ix[1:,featInds2])))
            # X = np.array(D2.ix[1:,featInds2])
        # print X[0,:]
        y = np.zeros(X.shape[0])
        # D.ix[D['#graph'].str.contains('CL'),2:17]
        RGfamilies = ['CL','GNP','PA','CNFG']
        # RGfamilies = ['CL','GNP','PA','PAmult']
        for i,s in enumerate(RGfamilies):
        #     print i
            y[np.array(D['#graph'].ix[1:].str.contains(s))] = i
        # print y
        holdfrac = 0.5
        Xtrain,Xtest,ytrain,ytest = cross_validation.train_test_split(X,y,test_size=holdfrac,stratify=y)

        #preprocess
        scaler1=StandardScaler()
        Xtrain = scaler1.fit_transform(Xtrain.astype(np.double))
        Xtest = scaler1.transform(Xtest.astype(np.double))
        
        # modelType = 'SVM-L2'
        # modelType = 'SVM-L1'
        # modelType = 'adaboost-tree'
        modelType = 'decision-tree'
        # modelType = 'random-forest'
        score,optWeights,clasf,rep,accs,f1s = doSVMcvPrediction(Xtrain, ytrain, Xtest, ytest, modelType)
        #classifier works perfectly
        # print clasf.get_params
        print "Checking distinctness of random graph families..."
        print rep
        # print accs[0],accs[1],f1s[0],f1s[1]
        # print np.column_stack((optWeights.T,D.columns[featInds]))
        
        #see which random graph model the novel gets classified as
        x = np.array(D.ix[0,featInds]).reshape(1,-1)
        if useSpectral:
            x = np.hstack((x,np.array(D2.ix[0,featInds2]).reshape(1,-1)))
            # x = np.array(D2.ix[0,featInds2]).reshape(1,-1)
        scaler=StandardScaler()
        X = scaler.fit_transform(X.astype(np.double))
        x = scaler.transform(x.astype(np.double)) 
        clasf.fit(X,y)
        novelRG = clasf.predict(x)
        if useSpectral:
            Fcol = D.columns[featInds].append(D2.columns[featInds2])
            # Fcol = D2.columns[featInds2]
        else:
            Fcol = D.columns[featInds]
        # print Fcol
        if modelType in ('SVM-L1','SVM-L2'):
            #TODO: print statements when useSpectral
            print "SVM feature weights:"
            print np.column_stack((clasf.coef_.T,Fcol))
            sc = x.dot(clasf.coef_.T) + clasf.intercept_
            print "scores:" 
            # print np.vstack((RGfamilies,(sc-np.min(sc))/np.sum(sc-np.min(sc))))
            print np.vstack((RGfamilies,sc))
        elif modelType in ('adaboost-tree'):
            print "Adaboost feature weights:"
            print np.column_stack((clasf.feature_importances_.T,Fcol))
            # print "prediction probabilities:" 
            # print np.vstack((RGfamilies,clasf.predict_proba(x)))
            print "decision function:"
            print np.vstack((RGfamilies,clasf.decision_function(x)))
            #also the actual tree?
        elif modelType in ('decision-tree'):
            print "Decision tree feature weights:"
            print np.column_stack((clasf.feature_importances_.T,Fcol))
            print "prediction probabilities:" 
            print np.vstack((RGfamilies,clasf.predict_proba(x)))
            #also the actual tree
            Fcol2 = [s.replace('n3','H').replace('n4','F') for s in Fcol]
            writeTree(clasf,Fcol2,graphFolder[:-1]+'_dTree.pdf')
        elif modelType in ('random-forest'):
            print "Random Forest feature weights:"
            print np.column_stack((clasf.feature_importances_.T,Fcol))
            print "prediction probabilities:" 
            print np.vstack((RGfamilies,clasf.predict_proba(x)))
        #print the prediction (and confidence score?)
        print "Fiction novel classified as: %s" % RGfamilies[int(novelRG)]


