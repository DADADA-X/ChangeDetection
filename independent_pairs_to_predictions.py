import os
import time
import datetime
import argparse
import rasterio
import numpy as np
from pathlib import Path

import config
import utils

parser = argparse.ArgumentParser(description='Helper script for combining DFC2021 prediction into submission format')
parser.add_argument('-i', '--input_dir', type=str, required=True, help='The path to a directory containing the output of the `inference.py` script.')
parser.add_argument('-o', '--output_dir', type=str, required=True, help='The path to output the consolidated predictions, should be different than `--input_dir`.')
# parser.add_argument('--soft_assignment', action="store_true", help='Flag for combining predictions using soft assignment. You can only use this if you ran the `inference.py` script with the `--save_soft` flag.')
args = parser.parse_args()


def main():
    print("Starting to combine predictions at %s" % (str(datetime.datetime.now())))

    #-------------------
    # Setup
    #-------------------
    assert os.path.exists(args.input_dir) and len(os.listdir(args.input_dir)) > 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    #-------------------
    # Run for each pair of predictions that we find in `--input_dir`
    #-------------------
    idxs_2013 = [
        fn.split("_")[0]
        for fn in os.listdir(args.input_dir)
        if fn.endswith("predictions-2013.tif")
    ]

    idxs_2017 = [
        fn.split("_")[0]
        for fn in os.listdir(args.input_dir)
        if fn.endswith("predictions-2017.tif")
    ]

    assert len(idxs_2013) > 0, "No matching files found in '%s'" % (args.input_dir)
    assert set(idxs_2013) == set(idxs_2017), "Missing some predictions"

    for i, idx in enumerate(idxs_2013):
        tic = time.time()

        print("(%d/%d) Processing tile %s" % (i+1, len(idxs_2013), idx), end=" ... ")

        # if args.soft_assignment:
        #     fn_2013 = os.path.join(args.input_dir, "%s_predictions-soft-2013.tif" % (idx))
        #     fn_2017 = os.path.join(args.input_dir, "%s_predictions-soft-2017.tif" % (idx))
        # else:
        #     fn_2013 = os.path.join(args.input_dir, "%s_predictions-2013.tif" % (idx))
        #     fn_2017 = os.path.join(args.input_dir, "%s_predictions-2017.tif" % (idx))
        fn_2013 = os.path.join(args.input_dir, "%s_predictions-2013.tif" % (idx))
        fn_2017 = os.path.join(args.input_dir, "%s_predictions-2017.tif" % (idx))
        output_fn = os.path.join(args.output_dir, "%s_predictions.tif" % (idx))

        assert os.path.exists(fn_2013) and os.path.exists(fn_2017)

        ## Load the independent predictions for both years
        with rasterio.open(fn_2013) as f:
            # if args.soft_assignment:
            #     t1 = np.rollaxis(f.read(), 0, 3)
            # else:
            #     t1 = f.read(1)
            t1_reduced = f.read(1)
            input_profile = f.profile.copy() # save the metadata for writing output
            
        with rasterio.open(fn_2017) as f:
            # if args.soft_assignment:
            #     t2 = np.rollaxis(f.read(), 0, 3)
            # else:
            #     t2 = f.read(1)
            t2_reduced = f.read(1)

        ## Convert to reduced land cover predictions
        # if args.soft_assignment:
        #     t1_reduced = (t1 @ config.NLCD_IDX_TO_REDUCED_LC_ACCUMULATOR).argmax(axis=2)
        #     t2_reduced = (t2 @ config.NLCD_IDX_TO_REDUCED_LC_ACCUMULATOR).argmax(axis=2)
        # else:
        #     t1_reduced = config.NLCD_IDX_TO_REDUCED_LC_MAP[t1]
        #     t2_reduced = config.NLCD_IDX_TO_REDUCED_LC_MAP[t2]

        ## Convert the two layers of predictions into the format expected by codalab
        predictions = (t1_reduced * 4) + t2_reduced
        predictions[predictions==5] = 0
        predictions[predictions==10] = 0
        predictions[predictions==15] = 0
        predictions = predictions.astype(np.uint8)

        ## Write output as GeoTIFF
        input_profile["count"] = 1
        with rasterio.open(output_fn, "w", **input_profile) as f:
            f.write(predictions, 1)

        print("finished in %0.4f seconds" % (time.time() - tic))

if __name__ == "__main__":
    main()