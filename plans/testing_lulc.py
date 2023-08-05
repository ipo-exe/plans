import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datasets as ds
from plans.project import Project
import glob, copy
import warnings

warnings.filterwarnings('ignore')

'''
stage_file = "../samples/table_series_stage_GS01txt"
rc_file = "../samples/table_rating_curve_GS01.txt"
rating = ds.RatingCurve(
    name="PCJ_RC1",
    date_start="2000-01-01",
    date_end="2020-01-01",
)


rating.load(
    table_file=rc_file,
    hobs_field="H_obs",
    qobs_field="F_obs",
    units_h="cm"
)

print(rating.data.head())
#rating.view()

rating.fit(n_grid=10)
#rating.update_model_data(h0=29.699, a=0.004877, b=2.06950)

rating.view_model(transform=True)

'''

proj = Project(root="C:/plans", name="Demo")

zip_url = "https://zenodo.org/record/8217681/files/default_samples.zip?download=1"

proj.download_datasets(zip_url=zip_url)
#zip_file = "C:/output/mysamples.zip"
#proj.extract_datasets(zip_file=zip_file)


