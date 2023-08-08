import matplotlib.pyplot as plt

from plans.project import Project
from plans import datasets as ds
import warnings
import os

warnings.filterwarnings('ignore')

pro = Project(name="Demo", root="C:/plans")


rc = ds.RatingCurve(name="RC01", date_start="2000-01-01", date_end="2020-01-01")

rc.load(
    table_file="{}/hydro/table_rating_curve_GS01.txt".format(pro.path_ds),
    hobs_field="H_obs",
    qobs_field="F_obs"
)

rc.fit()

print(rc.data.head().to_string())

obs = rc.get_bands(talk=True)

df = obs["Statistics"]

print(df.head(20).to_string())

plt.fill_between(x=df["H"], y1=df["Q_p05"], y2=df["Q_p95"], alpha=0.6, color="tab:grey")
plt.scatter(rc.data["Hobs"], rc.data["Qobs"])
plt.show()