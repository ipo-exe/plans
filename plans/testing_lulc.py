import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datasets as ds
import glob, copy
import warnings

warnings.filterwarnings('ignore')
stage_file = r"C:\Users\Ipo\My Drive\myProjects\121_paper_plans3br\inputs\datasets\potiribu\baked\potiribu_series_daily_stage_#75186000.txt"
sfile = r"C:\Users\Ipo\My Drive\myProjects\121_paper_plans3br\inputs\datasets\potiribu\potiribu_rating_curve_#75186000.txt"
#sfile = r"C:\Users\Ipo\My Drive\myProjects\121_paper_plans3br\inputs\datasets\pcj\pcj_rating_curve_#62663800.txt"
#sfile = r"C:\Users\Ipo\My Drive\myProjects\121_paper_plans3br\inputs\datasets\parsul\parsul_rating_curve_#58287000.txt"

rating = ds.RatingCurve(
    name="PCJ_RC1",
    date_start="2000-01-01",
    date_end="2020-01-01",
)

print(type(rating))

rating.load(
    table_file=sfile,
    hobs_field="H_obs",
    qobs_field="F_obs",
    units_h="cm"
)

print(rating.data.head())
#rating.view()

rating.fit(n_grid=10)


rating.update_model_data(h0=29.699, a=2.06950, b=0.004877)

print(rating.data_model.head().to_string())
df = rating.data_model

plt.scatter(df["Hobs"], df["Qobs"])
plt.plot(df["Hobs"], df["Qobs_Mean"])
plt.show()

from plans.analyst import Univar

uni = Univar(data=df["e"].values)
uni.view()

uni = Univar(data=df["eT"].values)
uni.view()
