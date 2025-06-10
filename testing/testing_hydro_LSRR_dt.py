import os
import matplotlib.pyplot as plt
from plans.hydro import LSRR

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    m = LSRR()
    m.boot("./data/bootfile_LSRR_0.csv")
    m.load()
    # check dt
    print(m.params["dt"])

    m.run()
    plt.plot(m.data["DateTime"], m.data["Q"] / m.params["dt"]["value"], ".", c="b")

    # change value and units of dt
    m.params["dt"]["value"] = 60
    m.params["dt"]["units"] = "min"
    print(m.params["dt"])

    # allow for dt change
    m.update_dt_flag = True
    # call updating method (it will convert to k-units)
    m.update_dt()
    print(m.params["dt"])
    # check results
    m.run()
    plt.plot(m.data["DateTime"], m.data["Q"] / m.params["dt"]["value"], ".", c="r")
    plt.ylabel("mm/d")
    plt.show()