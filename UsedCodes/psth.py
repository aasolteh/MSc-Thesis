import os
from core import get_spike_sessions
from plotFuncs import PSTHPlotter

sessions, cache = get_spike_sessions()

selecetedStimuli = ['gabors', 'flashes', 'drifting_gratings', 'static_gratings', 'natural_scenes']
stimStopTime     = [0.4, 0.4, 2.4, 0.4, 0.4]
for sessIdx in range(len(sessions)):
    currSession = cache.get_session_data(sessions.index.values[sessIdx])
    os.chdir("/home/amirali/Desktop/Thesis/Codes/results/PSTH")
    os.mkdir(str(sessions.index.values[sessIdx]))
    os.chdir(str(sessions.index.values[sessIdx]))
    sessUnits   = list(currSession.units["ecephys_structure_acronym"].unique())
    PSTHPlotter(selecetedStimuli, stimStopTime, sessUnits, currSession)