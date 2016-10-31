import numpy as np
import pandas as pd
from itertools import combinations
import io
from snap import GenPrefAttach,SaveEdgeList,TRnd
import subprocess

def getDegreeList(A):
    # n = np.unique(np.vstack((A[:,0],A[:,1]))).shape[0]
    n = int(np.max(np.vstack((A[:,0],A[:,1]))) + 1)
    degreeVec = np.zeros(n,dtype=int)
    for e in range(A.shape[0]):
        degreeVec[int(A[e,0])] += 1
        degreeVec[int(A[e,1])] += 1
    return degreeVec


def makeWeightedEdgelist(A,outname):
    #still remove self loops, as they make no sense in this context
    Atmp = np.array([row for row in A if row[0] != row[1]])
    inds = np.lexsort((Atmp[:,1],Atmp[:,0]))
    Asort = Atmp[inds,:]
    #get number of unique entries by taking diff
    Adiff1 = np.vstack((np.array([1,1]),np.diff(Asort,axis=0)))
    Adiff = np.any(Adiff1!=0,axis=1)
    #find where the diffs are equal to 1 a and diff that to get counts of unique
    outUnique = Asort[Adiff==1]
    outCounts = np.diff(np.hstack((np.where(Adiff==1)[0],Adiff.shape[0])))
    out = np.column_stack((outUnique,outCounts))
    if outname:
        np.savetxt(outname,out,fmt=('%d','%d','%d'),delimiter='\t',comments='')
    return out


def removeDuplicateEdges(X):
    #remove duplicates and self loops (and also sort)
    # xtmp = np.vstack({tuple(row) for row in X})
    xtmp = np.vstack({tuple(row) for row in X if row[0] != row[1]})
    inds = np.lexsort((xtmp[:,1],xtmp[:,0]))
    out = xtmp[inds,:]
    return out
        

def myPA(nodes,m,seed=4639):
    np.random.seed(seed)
    edgeList = []
    degreeVec = np.zeros(nodes)
    #initialize first step
    degreeVec[0:2] = np.array([1, 1])
    edgeList.append((0,1))
    for n in np.arange(2,nodes):
        #connect to existing vertices according to preferential attachment model
        # weighting of distribution is degreeVec[:n]
        probs = np.double(degreeVec[:n])
        neighbors = np.random.choice(np.arange(n),m,replace=True,p=probs/np.sum(probs))
        # print neighbors
        degreeVec[n] = m
        for dit in np.arange(m):
            #if edge included, increment both degrees and append edge to the list
            degreeVec[neighbors[dit]] += 1
            edgeList.append((neighbors[dit],n))
        # print degreeVec
        # print "avg degree: " + str(np.sum(degreeVec)/n)
    return np.asarray(edgeList)


def generateGraphs(params):
    graphname = params['graph']
    n = int(params['n'])
    numit = int(params['numGen'])
    graphType = params['type']
        
    if graphType == 'GNP':
        deg = int(params['d'])
        #every node has average degree deg, total number of edges is deg*n/2, divide by total possible edges 2/(n*(n-1))
        p = float(deg)/(n-1)
        # print "degree is " + str(p)
        np.random.seed(4639)
        #generate all randomness at once
        pairs = np.array([t for t in combinations(np.arange(n),2)])
        ps = np.random.rand(pairs.shape[0],numit) <= p
        for it in np.arange(numit):
            #keep the edges that are sampled
            pairsKeep = pairs[ps[:,it]==1]
            outname = graphname + '_' + graphType + '_' + str(it) + '.txt'
            np.savetxt(outname,pairsKeep,fmt=('%d','%d'),delimiter='\t',comments='')

    elif graphType == 'PA':
        deg = int(params['d'])
        for it in np.arange(numit):
            #is this degree right? or scale by 2
            #solve directly: 2/n + 2m = deg = 2|E|/n
            # x = myPA(n, int(deg-2./n), seed=it*4639+5011)
            x = myPA(n, int(deg/2.-1./n), seed=it*4639+5011)
            # x = myPA(n, int(deg/2.), seed=it*4639+5011)
            tmpname = graphname + '_' + graphType + '_' + str(it) + '_dup.txt'
            outname = graphname + '_' + graphType + '_' + str(it) + '.txt'
            # outname = graphname + '_' + graphType + 'mult_' + str(it) + '.txt'
            # makeWeightedEdgelist(x,tmpname)
            # np.savetxt(tmpname,x,fmt=('%d','%d'),delimiter='\t',comments='')
            xfinal = removeDuplicateEdges(x)
            np.savetxt(outname,xfinal,fmt=('%d','%d'),delimiter='\t',comments='')
            #make a weighted graph, keep track of weights for direct comparison with twilightEdgesIDsWeights.txt
            
    #keep the top edges that correspond to target |E| in original graph
    elif graphType == 'Pthresh':
        deg = int(params['d'])
        # Etarget = deg*n/2
        for it in np.arange(numit):
            #is this degree right? or scale by 2
            #solve directly: 2/n + 2m = deg = 2|E|/n
            x = myPA(n, int(deg/2.-1./n), seed=it*4639+5011)
            tmpname = graphname + '_' + graphType + '_' + str(it) + '_dup.txt'
            outname = graphname + '_' + graphType + '_' + str(it) + '.txt'
            xweighted = makeWeightedEdgelist(x,tmpname)
            #take the Etarget edges with largest weight
            Etarget = min(np.floor(deg*n/2.),xweighted.shape[0])
            eind = np.argsort(xweighted[:,2])[::-1] #sort by weight
            xtop = removeDuplicateEdges(xweighted[eind[:Etarget],:2])
            np.savetxt(outname,xfinal,fmt=('%d','%d'),delimiter='\t',comments='')
            

    elif graphType == 'PAsnap':
        deg = int(params['d'])
        Trnd1 = TRnd()
        for it in np.arange(numit):
            #generate graph
            Trnd1.PutSeed(it*4639+5011)
            x = GenPrefAttach(n,deg,Trnd1)
            #save output
            outname = graphname + '_' + graphType + '_' + str(it) + '.txt'
            SaveEdgeList(x,outname)
            #remove the top 3 lines, sed -i '' -e 1,3d tmp.txt
            emp = ''
            out = subprocess.call(["sed", "-i", emp, "-e", "1,3d", outname])
            
    elif graphType == 'CL':
        #get degree sequence from input
        w = params['dList']
        wnorm = float(np.sum(w))
        nc2 = n*(n-1)/2
        pairs = np.zeros((nc2,2))
        pairComp = np.zeros(nc2)
        for e,(i,j) in enumerate(combinations(np.arange(n),2)):
            #array comparison
            pairComp[e] = w[i]*w[j]/wnorm
            pairs[e,0] = i
            pairs[e,1] = j
        rands = np.random.rand(nc2,numit)
        for it in np.arange(numit):
                pairsKeep = pairs[rands[:,it] < pairComp]
                outname = graphname + '_' + graphType + '_' + str(it) + '.txt'
                np.savetxt(outname,pairsKeep,fmt=('%d','%d'),delimiter='\t',comments='')

    elif graphType == 'CNFG':
        w = params['dList']
        wnorm = np.sum(w)
        elist = np.zeros(wnorm)
        st = 0
        for i,wi in enumerate(w):
            elist[st:(st+wi)] = i
            st += wi
        for it in np.arange(numit):
            plist = np.random.permutation(elist)
            x = plist.reshape(-1,2)
            #if column 1 is greater than column 0 then swap that column
            xswap = x[:,0] > x[:,1]
            x[xswap,0:2] = np.column_stack((x[xswap,1],x[xswap,0]))
            tmpname = graphname + '_' + graphType + '_' + str(it) + '_wt.txt'
            outname = graphname + '_' + graphType + '_' + str(it) + '.txt'
            #sort correctly and remove self loops, duplicates
            xweighted = makeWeightedEdgelist(x,tmpname)
            np.savetxt(outname,xweighted[:,:2],fmt=('%d','%d'),delimiter='\t',comments='')
            

if __name__ == '__main__':
    #example parameters
    #123 undirected, but 1031 total weight if including multiedges
    # params = {'graph': 'twilight','type':'PA','n': 27,'d': int(2*1031/27),'numGen': 3}
    # params = {'graph': 'twilight','type':'PA','n': 27,'d': int(2*123/27),'numGen': 3}
    # 575 undirected, but 9464 total weight if including multiedges
    params = {'graph': 'goblet','type':'PA','n': 62,'d': int(2*575/62),'numGen': 3}
    generateGraphs(params)

    
