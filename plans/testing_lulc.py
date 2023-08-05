from plans.project import Project
from plans import datasets as ds
import warnings

warnings.filterwarnings('ignore')

pro = Project(name="Demo", root="C:/plans")

series_lulc = ds.LULCSeries(name="LULC Series")
# the folder method needs also the table of LULC attributes
series_lulc.load_folder(
    table_file="{}/lulc/table_lulc.txt".format(pro.path_ds),
    folder="{}/lulc".format(pro.path_ds),
    name_pattern="map_lulc_*",
    talk=True
)

print(series_lulc.catalog.to_string())

map_lulc_change = series_lulc.get_lulcc(
    date_start="1985-01-01",
    date_end="2000-01-01",
    by_lulc_id=39
)

map_lulc_change.view()