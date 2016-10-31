import numpy as np
import pandas as pd
import io

#this script preprocesses graph data from a character network
#input: csv file of weighted edges labelled with character names
#output: 2 text files labelled with nonnegative integers, 
#    one with a column for weights and one without

def main():
    filename = './thestandEdgesNames.csv'
    savename = './thestandEdgesIDs.txt'
    savename2 = './thestandEdgesIDsWeights.txt'
    # filename = './twilightEdgesNames.csv'
    # savename = './twilightEdgesIDs.txt'
    # savename2 = './twilightEdgesIDsWeights.txt'
    # filename = './gobletEdgesNames.csv'
    # savename = './gobletEdgesIDs.txt'
    # savename2 = './gobletEdgesIDsWeights.txt'
    E = pd.read_csv(filename)
    E1 = E['Source']
    E2 = E['Target']
    namesText = np.unique(np.vstack((E1,E2)))
    namesInds = [i for i in range(len(namesText))]
    # print namesText,namesInds
    E1 = E1.replace(namesText,namesInds)
    E2 = E2.replace(namesText,namesInds)
    #write to file
    out = np.column_stack((E1,E2))
    # labelNames = 'Source,Target'
    np.savetxt(savename,out,fmt=('%d','%d'),delimiter='\t',comments='')
    #save weights too
    np.savetxt(savename2,np.column_stack((out,E['weight'])),fmt=('%d','%d','%d'),delimiter='\t',comments='')

    print "n: %d" % len(namesText)
    # print "E: %d" % E['weight'].sum()
    print "E: %d" % E.shape[0]

if __name__ == '__main__':
    main()
