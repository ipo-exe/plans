from plans.project import Project
from plans import datasets as ds
import warnings
import os

warnings.filterwarnings('ignore')

#pro = Project(name="Demo", root="C:/plans")

s = "C:\gis\Godoi_et_al\K-factor_EPIC\K-factor_EPIC.tif"
print(os.path.dirname(os.path.abspath(s)))
print(os.path.basename(os.path.abspath(s)).split(".")[0])