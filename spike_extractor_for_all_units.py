import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import numpy as np
import pandas as pd

data_directory = 'raw_data/'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()
pathToSave = '/home/amirali/Desktop/Thesis/Codes/unit_data/Data/'

for sessIdx in range(len(sessions.index.values)):
    print("Session: ", str(sessions.index.values[sessIdx]))
    currSessNum = sessions.index.values[sessIdx]
    currSession = cache.get_session_data(currSessNum)

    if not 'natural_scenes' in currSession.stimulus_names:
        continue
    
    stimTable    = currSession.get_stimulus_table()
    imageTable   = stimTable[stimTable.stimulus_name == 'natural_scenes']
    imageIndices = imageTable.frame.unique()
    nonImageIdx  = np.where(imageIndices == -1.0)
    imageIndices = np.sort(np.delete(imageIndices, nonImageIdx))
    
    startWindow  = np.arange(-0.2, 0.78, 0.01)
    endWindow    = np.arange(-0.18, 0.8, 0.01)
    
    probes            = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
    currSessionProbes = list(set(list(currSession.units.ecephys_structure_acronym.unique())) & set(probes))
    
    for imageIdx in imageIndices:
        print("Image Stim: ", imageIdx)
        imagePath = os.path.join(pathToSave, str(int(imageIdx)))
        sessionPath = os.path.join(imagePath, str(sessions.index.values[sessIdx]))
        
        if not os.path.exists(imagePath):
            os.makedirs(imagePath)
        
        if not os.path.exists(sessionPath):
            os.makedirs(sessionPath)
            
        thisImageIndices    = np.where(imageTable.frame == imageIdx)[0]
        numImageIterations  = thisImageIndices.shape[0]
        imageStartTime      = imageTable.iloc[thisImageIndices].start_time.values
        for probe in currSessionProbes:
            print("Probe: ", probe)
            probePath = os.path.join(sessionPath, probe)
            if not os.path.exists(probePath):
                os.makedirs(probePath)
                
            if len([name for name in os.listdir(probePath) if os.path.isfile(os.path.join(probePath, name))]) > 0:
                continue
                
            probeRows = np.where(currSession.units.ecephys_structure_acronym == probe)
            probeIDs  = currSession.units.ecephys_structure_acronym.iloc[probeRows].index
            imageAvgSpikeValues = np.zeros((numImageIterations, probeIDs.shape[0], startWindow.shape[0]))
            for id in range(len(probeIDs)):
                for timeIdx in range(len(startWindow)):
                    tInit = startWindow[timeIdx]
                    tEnd  = endWindow[timeIdx]
                    for startTime in range(len(imageStartTime)):
                        imageAvgSpikeValues[startTime, id, timeIdx] = np.where((currSession.spike_times[probeIDs[id]] <= imageStartTime[startTime] 
                                                                    + tEnd) & (currSession.spike_times[probeIDs[id]] >= imageStartTime[startTime] 
                                                                    + tInit))[0].shape[0]
            imageAvgSpikeValues = np.mean(imageAvgSpikeValues, axis=0)
            with open(probePath + '/spikes.npy', 'wb') as f:
                np.save(f, imageAvgSpikeValues)