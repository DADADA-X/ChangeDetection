import rasterio
import numpy as np
import matplotlib.pyplot as plt

import utils

with rasterio.open('/data/xyj/MSD/results/nlcd_only_baseline/submission/1950_predictions.tif') as f:
    data_nlcd_class = f.read(1)
    input_profile = f.profile.copy()

output_profile = input_profile.copy()
output_profile["driver"] = "GTiff"

data_nlcd_idx = utils.NLCD_CLASS_TO_IDX_MAP[data_nlcd_class].astype(np.uint8)


