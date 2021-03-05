import numpy as np

# Other settings
## Number of classes
HR_NCLASSES = 5 # Water, Tree Canopy, Low Vegetation, Imprevious surfaces, Nodata
LR_NCLASSES = 17

NLCD_CLASSES = [0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90,
                95]  # 16 classes + 1 nodata class ("0"). Note that "12" is "Perennial Ice/Snow" and is not present in Maryland.

GAMMA = (39.0 / 40.0)
ETA = (1.0 / 40.0)

NAIP_2013_MEANS = np.array([117.00, 130.75, 122.50, 159.30])
NAIP_2013_STDS = np.array([38.16, 36.68, 24.30, 66.22])
NAIP_2017_MEANS = np.array([72.84, 86.83, 76.78, 130.82])
NAIP_2017_STDS = np.array([41.78, 34.66, 28.76, 58.95])

NLCD_CLASS_COLORMAP = {  # Copied from the emebedded color table in the NLCD data files
    0: (0, 0, 0, 255),
    11: (70, 107, 159, 255),
    12: (209, 222, 248, 255),
    21: (222, 197, 197, 255),
    22: (217, 146, 130, 255),
    23: (235, 0, 0, 255),
    24: (171, 0, 0, 255),
    31: (179, 172, 159, 255),
    41: (104, 171, 95, 255),
    42: (28, 95, 44, 255),
    43: (181, 197, 143, 255),
    52: (204, 184, 121, 255),
    71: (223, 223, 194, 255),
    81: (220, 217, 57, 255),
    82: (171, 108, 40, 255),
    90: (184, 217, 235, 255),
    95: (108, 159, 184, 255)
}

NLCD_IDX_TO_REDUCED_LC_MAP = np.array([
    4,  # 0 No data 0
    0,  # 1 Open Water
    4,  # 2 Ice/Snow
    2,  # 3 Developed Open Space
    3,  # 4 Developed Low Intensity
    3,  # 5 Developed Medium Intensity
    3,  # 6 Developed High Intensity
    3,  # 7 Barren Land
    1,  # 8 Deciduous Forest
    1,  # 9 Evergreen Forest
    1,  # 10 Mixed Forest
    1,  # 11 Shrub/Scrub
    2,  # 12 Grassland/Herbaceous
    2,  # 13 Pasture/Hay
    2,  # 14 Cultivated Crops
    1,  # 15 Woody Wetlands
    1,  # 16 Emergent Herbaceious Wetlands
])

NLCD_IDX_TO_REDUCED_LC_ACCUMULATOR = np.array([
    [0, 0, 0, 0, 1],  # 0 No data 0
    [0.98, 0.02, 0, 0, 0],  # 1 Open Water
    [0, 0, 0, 0, 1],  # 2 Ice/Snow
    [0, 0.39, 0.49, 0.12, 0],  # 3 Developed Open Space
    [0, 0.31, 0.34, 0.35, 0],  # 4 Developed Low Intensity
    [0.01, 0.13, 0.22, 0.64, 0],  # 5 Developed Medium Intensity
    [0, 0.03, 0.07, 0.90, 0],  # 6 Developed High Intensity
    [0.05, 0.13, 0.43, 0.40, 0],  # 7 Barren Land
    [0, 0.93, 0.05, 0, 0],  # 8 Deciduous Forest
    [0, 0.95, 0.04, 0, 0],  # 9 Evergreen Forest
    [0, 0.92, 0.07, 0, 0],  # 10 Mixed Forest
    [0, 0.58, 0.38, 0.04, 0],  # 11 Shrub/Scrub
    [0.01, 0.23, 0.54, 0.22, 0],  # 12 Grassland/Herbaceous
    [0, 0.12, 0.83, 0.03, 0],  # 13 Pasture/Hay
    [0, 0.05, 0.92, 0.01, 0],  # 14 Cultivated Crops
    [0, 0.94, 0.05, 0, 0],  # 15 Woody Wetlands
    [0.08, 0.86, 0.05, 0, 0],  # 16 Emergent Herbaceious Wetlands
])

## data augmentation probability
TRAIN_AUG_PROB = 0.5
VALID_AUG_PROB = 0

## Training implement details
SAVE_PERIOD = 5
NUM_CHIPS_PER_TILE = 10
TRAIN_CHIP_SIZE = 512

## Inference implement details
INF_CHIP_SIZE = 512
INF_PADDING = 256


## Change detection
NUM_CHIPS_PER_TILE_DET = 10
TRAIN_CHIP_SIZE_DET = 256