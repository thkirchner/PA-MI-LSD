from ippai.simulate import SegmentationClasses
from ippai.utils import TissueSettingsGenerator
from ippai.utils import CHROMOPHORE_LIBRARY
from ippai.utils import SPECTRAL_LIBRARY
from ippai.utils import Chromophore
from ippai.utils import Tags
from ippai.io_handling import load_hdf5

import pandas as pd
import numpy as np

path = "./SETS/MI-LSD/MI-LSD_SET03_"
WAVELENGTHS = np.arange(680, 981, 20)
NUM_VOLUMES = 1000

max_size = 3000000
label_rCu = np.zeros([max_size])
vol_id = np.zeros([max_size])
spectra = np.zeros([4, 16, max_size])
i = 0

for idx_vol in range(NUM_VOLUMES):
    idx_vol = idx_vol + 4000
    volume = load_hdf5(path + str(idx_vol) + "_ill_0/ippai_output.hdf5")
    seg_ = np.asarray(
        volume['simulations']['original_data']['simulation_properties'][str(WAVELENGTHS[0])]['seg'])
    rCu_ = np.asarray(
        volume['simulations']['original_data']['simulation_properties'][str(WAVELENGTHS[0])]['oxy'])
    segmented_rCu = np.ma.masked_where(seg_ > 3, rCu_).compressed()
    desc = np.zeros([4, WAVELENGTHS.shape[0], segmented_rCu.shape[0]])
    
    for idx_ill in range(4):
        volume = load_hdf5(path + str(idx_vol) + "_ill_"+ str(idx_ill) + "/ippai_output.hdf5")
        for idx_WL, WL in enumerate(WAVELENGTHS):
            initial_pressure = np.asarray(
                volume['simulations']['original_data']['optical_forward_model_output'][str(WL)]['initial_pressure'])
            desc[idx_ill,idx_WL,:] = np.ma.masked_where(seg_ > 3, initial_pressure).compressed()
    for idx_pixel in range(segmented_rCu.shape[0]):
        vol_id[i] = idx_vol
        label_rCu[i] = segmented_rCu[idx_pixel]
        spectra[:,:,i] = desc[:,:,idx_pixel]
        i += 1
    print(idx_vol, "/", NUM_VOLUMES)
print(vol_id[:i].shape)
print(label_rCu[:i].shape)
print(spectra[:,:,:i].shape)

np.save("./MI-LSD_SET03_TEST_array_label_rCu.npy", label_rCu[:i])
np.save("./MI-LSD_SET03_TEST_array_vol_id.npy", vol_id[:i])
np.save("./MI-LSD_SET03_TEST_array_spectra.npy", spectra[:,:,:i])
