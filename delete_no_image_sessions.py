import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import numpy as np
import shutil

data_directory = 'raw_data/'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
sessions = cache.get_session_table()

for sessNum in range(len(sessions.index.values)):
    print(sessNum)
    thisSessionNum = sessions.index.values[sessNum]
    thisSession    = cache.get_session_data(thisSessionNum)
    stimTable        = thisSession.get_stimulus_table()
    isImage = np.where(stimTable.stimulus_name.unique() == "natural_scenes")[0].shape[0]
    
    if isImage == 0:
        shutil.rmtree('/home/amirali/Desktop/Thesis/Codes/unit_data/Data/' + str(thisSessionNum))