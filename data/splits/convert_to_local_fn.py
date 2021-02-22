import argparse
import pandas as pd
import numpy as np
from pathlib import Path

splits_p = Path('/home/xyj/code/dfc2021-msd-baseline/data/splits')
csv_fns = [str(f) for f in splits_p.glob("*.csv")]
local_csv_fns = [str(f) for f in splits_p.glob("*_local.csv")]
url_csv_fns = [f for f in csv_fns if f not in local_csv_fns]

for input_fn in url_csv_fns:
    input_dataframe = pd.read_csv(input_fn)
    image_fns = input_dataframe["image_fn"].values
    label_fns = input_dataframe["label_fn"].values
    groups = input_dataframe["group"].values

    image_fns_local = []
    label_fns_local = []

    for image_fn in image_fns:
        image_fns_local.append(image_fn.replace('https://dfc2021.blob.core.windows.net', '/data/xyj'))

    for label_fn in label_fns:
        label_fns_local.append(label_fn.replace('https://dfc2021.blob.core.windows.net', '/data/xyj'))

    image_fns_local = np.array(image_fns_local, dtype=object)
    label_fns_local = np.array(label_fns_local, dtype=object)

    output_dataframe = pd.DataFrame({"image_fn": image_fns_local, "label_fn": label_fns_local, "group": groups})

    output_fn = input_fn.split('.')[0] + '_local.csv'
    output_dataframe.to_csv(output_fn, index=False)
