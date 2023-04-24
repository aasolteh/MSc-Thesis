import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import numpy as np
import pandas as pd

def spike_extractor(session, stim_indices, stim_table, start_win, end_win, probe_units):
    stim_mean_spike_vals = spike_average_calculator(start_win, end_win, stim_indices, stim_table, probe_units, session)
    return(stim_mean_spike_vals)

def indice_pic_calculator(dataframe):
    indice_list = []
    for i in range(dataframe.shape[0]):
        file_name = dataframe.iloc[i].split('.')
        indice_list.append(float(file_name[0].split('_')[-1]))
    return indice_list

def indice_stim_calculator(stim_table, idx):
    stim_images   = stim_table[stim_table.stimulus_name == 'natural_scenes']
    object_indice = stim_images[stim_images.frame == idx].index.values
    return object_indice

def spike_average_calculator(start_win, end_win, indices, stim_table, units, session):
    mean_spike_values   = np.zeros((units.shape[0], start_win.shape[0]))
    for win_idx in range(len(start_win)):
        startValues = stim_table.iloc[indices].start_time.values
        for unit in range(len(units)):
            unitSpikes = np.zeros(startValues.shape[0])
            for start_idx in range(len(startValues)):
                unitSpikes[start_idx] = np.where((session.spike_times[units[unit]] > startValues[start_idx] + start_win[win_idx]) & 
                                    (session.spike_times[units[unit]] < startValues[start_idx] + end_win[win_idx]))[0].shape[0]
            mean_spike_values[unit, win_idx] = np.mean(unitSpikes)
    return mean_spike_values

data_directory = 'raw_data/'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()

for sessNum in range(len(sessions.index.values)):
    print(sessNum)
    thisSessionNum = sessions.index.values[sessNum]
    thisSession    = cache.get_session_data(thisSessionNum)
    stimTable      = thisSession.get_stimulus_table()
    isImage        = np.where(stimTable.stimulus_name.unique() == "natural_scenes")[0].shape[0]
    if not isImage:
        continue

    ## Creating Relative Paths
    sessionPath = '/home/amirali/Desktop/Thesis/Codes/unit_data/Data/HiLoContrast/' + str(thisSessionNum)
    if not os.path.exists(sessionPath):
        os.makedirs(sessionPath)
    sessionDataPath = '/home/amirali/Desktop/Thesis/Codes/unit_data/Data/HiLoContrast/' + str(thisSessionNum) + '/'
    if not os.path.exists(sessionDataPath + 'high_data'):
        os.makedirs(sessionDataPath + 'high_data')
    if not os.path.exists(sessionDataPath + 'low_data'):
        os.makedirs(sessionDataPath + 'low_data')
    highDataPath = sessionDataPath + 'high_data/'
    lowDataPath   = sessionDataPath + 'low_data/'
    
    ## Extracting Probes
    probes           = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
    session_probes   = list(thisSession.units.ecephys_structure_acronym.unique())
    available_probes = list(set(probes) & set(session_probes))

    ## Extracting Stimuli
    wholeDataFrame = pd.read_csv('labels.csv')
    lowDataFrame   = wholeDataFrame[wholeDataFrame.LContrast == 1].name
    highDataFrame  = wholeDataFrame[wholeDataFrame.HContrast == 1].name
    
    lowPicIndices  = indice_pic_calculator(lowDataFrame)
    highPicIndices = indice_pic_calculator(highDataFrame)

    for probe in available_probes:
        probeUnits       = thisSession.units[(thisSession.units.ecephys_structure_acronym == probe)].index.values
        startWin         = np.arange(-0.2, 0.75, 0.01, dtype=float)
        endWin           = np.arange(-0.15, 0.80, 0.01, dtype=float)
        stimSpikesInTime = np.zeros((len(lowPicIndices), len(probeUnits), len(startWin)))
        for idx_low in range(len(lowPicIndices)):
            lowStimIndices  = indice_stim_calculator(stimTable, lowPicIndices[idx_low])
            stimSpikesInTime[idx_low, :, :] = spike_extractor(session=thisSession, stim_indices=lowStimIndices, stim_table=stimTable,
                                                                start_win=startWin, end_win=endWin, probe_units=probeUnits)
        with open(lowDataPath + 'lowMeanSpikeValues_' + probe + '.npy', 'wb') as f:
                np.save(f, stimSpikesInTime)
        stimSpikesInTime = np.zeros((len(highPicIndices), len(probeUnits), len(startWin)))
        for idx_high in range(len(highPicIndices)):
            highStimIndices  = indice_stim_calculator(stimTable, highPicIndices[idx_high])
            stimSpikesInTime[idx_high, :, :] = spike_extractor(session=thisSession, stim_indices=highStimIndices,  stim_table=stimTable,
                                                                start_win=startWin, end_win=endWin, probe_units=probeUnits)
        with open(highDataPath + 'highMeanSpikeValues_' + probe + '.npy', 'wb') as f:
                np.save(f, stimSpikesInTime)