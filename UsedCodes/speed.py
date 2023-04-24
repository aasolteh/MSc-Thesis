import os
from core import get_spike_sessions
from plotFuncs import SPEEDPlotter

sessions, cache = get_spike_sessions()
os.mkdir("/home/amirali/Desktop/Thesis/Codes/results/SPEED")

for sessIdx in range(len(sessions)):
    print(sessIdx)
    currSession = cache.get_session_data(sessions.index.values[sessIdx])
    os.chdir("/home/amirali/Desktop/Thesis/Codes/results/SPEED")
    os.mkdir(str(sessions.index.values[sessIdx]))
    os.chdir(str(sessions.index.values[sessIdx]))
    SPEEDPlotter(currSession)