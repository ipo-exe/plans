import pandas as pd

df = pd.read_csv("../plans/iofiles.csv", sep=",")
df_input = df.query("io == 'input'")
print(df_input.to_string())

lst_input = list()
for i in range(len(df_input)):
    s_format = df_input["format"].values[i]
    lst_input.append("``{}``\n".format(df_input["file"].values[i]))
    lst_input.append("-" * 60)
    s_line = "\n[{}] {}.\n\n".format(
        s_format,
        df_input["description"].values[i])
    lst_input.append(s_line)
    if s_format == "raster map":
        lst_input.append("")
        lst_input.append(" - Data type: {}\n".format(df_input["data type"].values[i]))
        lst_input.append(" - No-data value: {}\n".format(df_input["no-data value"].values[i]))
    lst_input.append("\n\n")

f = open("./input_catalog.rst", "w")
f.writelines(lst_input)
f.close()