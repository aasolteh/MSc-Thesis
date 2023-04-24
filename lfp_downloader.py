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

probes = ['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF']
for _ in range(len(sessions.index.values)):
    session = cache.get_session_data(sessions.index.values[_])
    for probe in session.probes.index.values:
        session.get_lfp(probe)