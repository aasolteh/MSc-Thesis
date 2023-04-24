from core import get_spike_sessions

sessions, cache, data_directory = get_spike_sessions()
numPics = 118
for pic in range(-1, numPics):
    cache.get_natural_scene_template(pic)