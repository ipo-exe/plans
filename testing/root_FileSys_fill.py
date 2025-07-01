from plans.root import FileSys

if __name__ == "__main__":

    fs = FileSys(folder_base="C:/plans", name="mytest")

    FileSys.make_dir(fs.folder_main)

    dc = {
        "inputs": {
            "topo":
                {
                    "_intermediate": {0:0}
                },
            "lulc":
                {
                    "bsl": {
                            "_intermediate": {0:0},
                    },
                    "bau": {
                        "_intermediate": {0:0},
                    }
                },
            "clim":
                {
                    "bsl": {
                        "_intermediate": {0:0},
                    },
                    "bau": {
                        "_intermediate": {0:0},
                    }
                },
            "basins":
                {
                    "_intermediate": {0:0},
                },
        },
        "outputs": {0:0}
    }

    FileSys.fill(dict_struct=dc, folder=fs.folder_main, handle_files=False)
