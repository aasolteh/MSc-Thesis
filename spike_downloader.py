import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# this path determines where downloaded data will be stored
mainPath = os.getcwd()
manifest_path = os.path.join(mainPath + '/raw_data', "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()

for _ in range(len(sessions.index.values)):
    cache.get_session_data(sessions.index.values[_],
                            isi_violations_maximum=np.inf,
                            amplitude_cutoff_maximum=np.inf,
                            presence_ratio_minimum=-np.inf)