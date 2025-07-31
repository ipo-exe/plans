import os
import matplotlib.pyplot as plt
from plans.hydro import Global
from testing.hydro_utils import copy_inputs, delete_inputs

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":

    # instantiate
    m = Global(name="Global", alias="GLB000")
    m.boot(bootfile="./data/bootfile_Global.csv")
    copy_inputs(dst_folder=m.folder_data)
    m.load()

    m.params["g0"]["value"] = 10
    m.params["s0"]["value"] = 0

    # set up simulation data
    m.setup()
    print(m.data_tah)
    print(m.data_guh["global"].sum())

    m.solve()

    print(m.data.info())

    m.view(mode="river")
    m.export(
        folder="./data/Global/outputs",
        filename=m.name,
        views=True,
        mode="river"
    )
    df1 = m.data[[m.field_datetime, "q_global"]].copy()
    m.params["k_q"]["value"] = 100
    m.setup()
    m.solve()
    m.view(mode="river")
    df2 = m.data[[m.field_datetime, "q_global"]].copy()


    plt.plot(df1[m.field_datetime], df1["q_global"], "b")
    plt.plot(df2[m.field_datetime], df2["q_global"], "r")
    plt.show()

    delete_inputs(dst_folder=m.folder_data)
    print("\n>>> OK. PASSING.\n")



