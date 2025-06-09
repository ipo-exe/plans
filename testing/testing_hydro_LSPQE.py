import pprint

import pandas as pd

from plans.hydro import LSPQE


if __name__ == "__main__":
    m = LSPQE(name="LSPQE", alias="LSPEQ01")
    m.boot(bootfile="./data/bootfile_LSPQE.csv")
    m.load()
    # with epot
    m.shutdown_epot = False
    m.run()
    m.evaluate()
    m.export(
        folder="./data/LSPQE/outputs",
        filename="LSPQE-epot",
        views=True
    )
    # no epot
    m.shutdown_epot = True
    m.run()
    m.evaluate()
    m.export(
        folder="./data/LSPQE/outputs",
        filename="LSPQE-noepot",
        views=True
    )


