import os
import pprint
from plans.parsers import flare

# ensure here is the current dir for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if __name__ == "__main__":
    print("hello")

    sn = "N00002p031231"
    n = flare.decode_number(encoded_number=sn)
    print(n)
    print(type(n))

    ex = "w20p0e22s13P3s12"
    e = flare.decode_extent(ex)
    pprint.pprint(e)

    n = -34.343242
    ne = flare.encode_number(n, decimals=7, len_min=5)
    print(ne)

    e2 = flare.encode_extent(e, len_min=3, human=True)
    print(e2.upper())
    e = flare.decode_extent(e2)
    pprint.pprint(e)

    ts = "rs2014f2015"#"20140302t124804p143zw0300"
    ts_dc = flare.decode_epoch(ts)
    pprint.pprint(ts_dc)
    ts = flare.encode_epoch(ts_dc)
    print(ts)
    #ts_iso = flare.encode_iso8601(datetime_dc=ts_dc)
    #print(ts_iso)