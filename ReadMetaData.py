import os
import pickle


global_name = "mvaTrees6-300-OutputSuppression-v15-pow4-times01Lambda-inverse-p05Only"
nentries=-1

metaData='data/'+global_name+'/metaData_'+str(nentries)+'.pkl'
if os.path.exists(metaData):
    # Get the meta data containing the keys (input feature anmes)
    print("Obtaining input features from metaData_{}.pkl".format(global_name))
    metaDataFile = open(metaData, 'rb')
    metaDataDict = pickle.load(metaDataFile)
    metaDataFile.close()
    
    for idx,(key,pair) in enumerate(metaDataDict.items()):
        print("key: {},    Pair:  {}".format(key,pair))
