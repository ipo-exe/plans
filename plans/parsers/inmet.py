"""
Parse data from ``INMET`` files (Brazillian Meteorological Office)

Description:
    The ``inmet`` module provides data parsers for the ``INMET`` ``.csv`` data files.

License:
    This software is released under the GNU General Public License v3.0 (GPL-3.0).
    For details, see: https://www.gnu.org/licenses/gpl-3.0.html

Author:
    Iporã Possantti

Contact:
    possantti@gmail.com


Overview
--------

todo overview
Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Pellentesque habitant morbi tristique senectus
et netus et malesuada fames ac turpis egestas.

>>> from plans.parsers import inmet

Class aptent taciti sociosqu ad litora torquent per
conubia nostra, per inceptos himenaeos. Nulla facilisi. Mauris eget nisl
eu eros euismod sodales. Cras pulvinar tincidunt enim nec semper.

Example
-------

todo examples
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nulla mollis tincidunt erat eget iaculis. Mauris gravida ex quam,
in porttitor lacus lobortis vitae. In a lacinia nisl.

.. code-block:: python

    import numpy as np
    from plans import analyst

    # get data to a vector
    data_vector = np.random.rand(1000)

    # instantiate the Univar object
    uni = analyst.Univar(data=data_vector, name="my_data")

    # view data
    uni.view()

Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl. Mauris gravida ex quam, in porttitor lacus lobortis vitae.
In a lacinia nisl.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

from plans.datasets import get_random_colors

# todo docstrings
class _StationINMET_:

    # todo doctring
    def __init__(self, name="MyStationINMET"):
        # Object attributes
        self.name = name
        self.dtfield = "DateTime"
        self.prefix = "inmet"

        self.gls = {
            'CH (NUVENS ALTAS)(codigo)': 'CH',
            'PRECIPITACAO TOTAL, HORARIO(mm)': 'P',
            'VISIBILIDADE, HORARIA(codigo)': 'Vis'
        }

        # data
        d1, d2 = self._metadata_util()
        self.station_metadata = d1
        self.station_metadata_df = None
        self.glossary = self.get_glossary()
        self.data = None
        self.start_data = None
        self.end_data = None

        # reading
        self.reading_metadata_splitter = ": "
        self.reading_delimiter = ";"
        self.reading_decimals = "."
        self.reading_skiprows = len(self.station_metadata) + 1
        self.reading_dtfields = ["Data Medicao", "Hora Medicao"]
        self.reading_encoding = "utf-8"
        self.reading_dtformat = '%Y-%m-%d %H:%M:%S'
        self.reading_nan = ['null', -9999]

    # todo doctring
    def get_metadata(self):
        dict_meta = {
            "Name": self.name,
            "Start": self.start_data,
            "End": self.end_data
        }
        dict_meta.update(self.station_metadata)
        return dict_meta

    # todo doctring
    def update(self):
        if self.data is not None:
            self.start_data = self.data[self.dtfield].min()
            self.end_data = self.data[self.dtfield].max()
        return None

    # todo doctring
    def _metadata_util(self):
        d = {
            "NameStation": [None, "Nome"],
            "CodeStation": [None, "CodigoEstacao"],
            "Y": [None, "Latitude"],
            "X": [None, "Longitude"],
            "Z": [None, "Altitude"],
            "Status": [None, "Situacao"],
            "StartStation": [None, "DataInicial"],
            "EndStation": [None, "DataFinal"],
            "TimeScale": [None, "PeriodicidadedaMedicao"]
        }
        d1 = {}
        d2 = {}
        for k in d:
            d1[k] = d[k][0]
            d2[d[k][1]] = k

        return d1, d2

    # todo doctring
    def _standardize_metadata_fields(self, metadata):
        dict_1, dict_convert = self._metadata_util()
        new_metadata = {}
        for k in dict_convert:
            new_metadata[dict_convert[k]] = metadata[k]
        return new_metadata

    # todo doctring
    def _get_columns_gls(self):
        # compute the opposite
        inverse_mapping = {v: k for k, v in self.gls.items()}
        return self.gls, inverse_mapping

    # todo doctring
    def get_glossary(self):
        a, b = self._get_columns_gls()
        df = pd.DataFrame(
            {
                "Field": [a[k] for k in a],
                "SourceDescription": [k for k in a]
            }
        )
        return df

    # todo docstring
    def fix_text(self, text, remove_spaces=False):
        text = text.upper()
        # Iterate through each line and replace problematic characters
        text = text.replace('?', "A")
        text = text.replace('Ç', "C")
        text = text.replace('Á', "A")
        text = text.replace('Ã', "A")
        text = text.replace('Í', "I")
        text = text.replace('É', "E")
        text = text.replace('(YYYY-MM-DD)', '')
        if remove_spaces:
            text = text.replace(" ", "")
        return text

    # todo doctring
    def read_metadata(self, file_path):
        # Read station_metadata
        with open(file_path, 'r') as file:
            # Read lines until a blank line is encountered, indicating the end of station_metadata
            metadata = {}
            for line in file:
                if line.strip() == '' or len(line.strip()) > 50:
                    break
                key, value = line.strip().split(self.reading_metadata_splitter)
                value = value.replace(",", ".") # fix decimals
                key = self.fix_text(text=key, remove_spaces=True)
                metadata[key] = value
        # standardize fields
        metadata = self._standardize_metadata_fields(metadata=metadata)
        return metadata

    # todo doctring
    def load_metadata(self, file_path):
        self.station_metadata = self.read_metadata(file_path)
        self.station_metadata_df = pd.DataFrame(
            {
                "Field": [k for k in self.station_metadata],
                "Value": [self.station_metadata[k] for k in self.station_metadata]
            }
        )
        return None

    # todo doctring
    def read_data(self, file_path):

        # load station_metadata first
        self.load_metadata(file_path=file_path)

        # Read data using pandas, specifying the data type of the 'Hora Medicao' column as a string
        df = pd.read_csv(
            file_path,
            skiprows=self.reading_skiprows,
            delimiter=self.reading_delimiter,
            dtype={e: str for e in self.reading_dtfields},
            decimal=self.reading_decimals,
            encoding=self.reading_encoding,
            na_values=self.reading_nan
        )
        # Combine 'Data Medicao' and 'Hora Medicao' columns into a single datetime field
        if len(self.reading_dtfields) > 1:
            df[self.dtfield] = pd.to_datetime(df[self.reading_dtfields[0]].str.split().str[0] + ' ' + df[self.reading_dtfields[1]].str.split().str[0], format=self.reading_dtformat)
        else:
            df[self.dtfield] = pd.to_datetime(df[self.reading_dtfields[0]] + " 12:00:00", format=self.reading_dtformat)

        # Drop the original 'Data Medicao' and 'Hora Medicao' columns
        df.drop(self.reading_dtfields, axis=1, inplace=True)

        # Reorganize columns with 'DateTime' at the beginning
        columns = [self.dtfield] + [column for column in df if column != self.dtfield]

        # Apply the column reorganization
        df = df[columns]

        # Check and eliminate the last column if it doesn't have a name
        if "Unnamed" in columns[-1]:
            df = df.iloc[:, :-1]

        fix_names = {}
        for c in df.columns[1:]:
            fix_names[c] = self.fix_text(c)
        # Rename columns in the DataFrame
        df.rename(columns=fix_names, inplace=True)

        # get new colum names
        simplified_names, inverse_mapping = self._get_columns_gls()

        # Rename columns in the DataFrame
        df.rename(columns=simplified_names, inplace=True)

        # Clear void rows
        df = df.dropna(subset=df.columns[1:], how='all')

        return df

    # todo doctring
    def load_data(self, file_path):
        self.data = self.read_data(file_path=file_path)
        self.update()
        return None

    # todo doctring
    def export_data(self, folder):
        filename = "{}_{}.csv".format(self.prefix, self.station_metadata["CodeStation"])
        file_path = "{}/{}".format(folder, filename)
        self.data.to_csv(file_path, sep=";", index=False)
        return file_path

    # todo doctring
    def export_metadata(self, folder):
        filename = "{}_{}_info.csv".format(self.prefix, self.station_metadata["CodeStation"])
        file_path = "{}/{}".format(folder, filename)
        self.station_metadata_df.to_csv(file_path, sep=";", index=False)
        return file_path

    # todo doctring
    def export_glossary(self, folder):
        filename = "{}_{}_glossary.csv".format(self.prefix, self.station_metadata["CodeStation"])
        file_path = "{}/{}".format(folder, filename)
        df = self.get_glossary()
        df.to_csv(file_path, sep=";", index=False)
        return file_path

    # todo docstring
    def export(self, folder):
        f1 = self.export_data(folder=folder)
        f2 = self.export_metadata(folder=folder)
        f3 = self.export_glossary(folder=folder)
        return {
            "Data": f1,
            "Metadata": f2,
            "Glossary": f3
        }

# todo docstrings
class _Conventional_(_StationINMET_):

    # todo docstring
    def __init__(self, name="MyConvetional"):
        super().__init__(name=name)
        # overwrite super() attributes:
        self.prefix = "inmet_conv"

        self.gls =  {
            'CH (NUVENS ALTAS)(codigo)': 'CH',
            'VISIBILIDADE, HORARIA(codigo)': 'Vis'
        }
        self.glossary = self.get_glossary()

        # reading
        self.reading_metadata_splitter = ": "
        self.reading_delimiter = ";"
        self.reading_decimals = "."
        self.reading_skiprows = len(self.station_metadata) + 1
        self.reading_dtfields = ["DATA MEDICAO", "HORA MEDICAO"]

    # todo docstring
    def _metadata_util(self):
        d = {
            "NameStation": [None, "NOME"],
            "CodeStation": [None, "CODIGOESTACAO"],
            "Y": [None, "LATITUDE"],
            "X": [None, "LONGITUDE"],
            "Z": [None, "ALTITUDE"],
            "Status": [None, "SITUACAO"],
            "StartStation": [None, "DATAINICIAL"],
            "EndStation": [None, "DATAFINAL"],
            "TimeScale": [None, "PERIODICIDADEDAMEDICAO"]
        }
        d1 = {}
        d2 = {}
        for k in d:
            d1[k] = d[k][0]
            d2[d[k][1]] = k

        return d1, d2

# todo docstring
class ConventionalHourly(_Conventional_):

    # todo docstring
    def __init__(self, name="MyConvetionalHourly"):
        super().__init__(name=name)
        # overwrite super() attributes:
        self.prefix = "inmet_convh"
        self.gls = {
            'CH (NUVENS ALTAS)(CODIGO)': 'CH',
            'CL (NUVENS BAIXAS)(CODIGO)': 'CL',
            'CM (NUVENS MEDIAS)(CODIGO)': 'CM',
            'NEBULOSIDADE, HORARIA(DECIMOS)': 'Neb',
            'PRECIPITACAO TOTAL, HORARIO(MM)': 'P',
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(MB)': 'PALoc',
            'PRESSAO ATMOSFERICA AO NIVEL DO MAR, HORARIA(MB)': 'PASea',
            'TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)': 'TempDB',
            'TEMPERATURA DO AR - BULBO UMIDO, HORARIA(°C)': 'TempWB',
            'TEMPERATURA DO PONTO DE ORVALHO(°C)': 'TempDP',
            'UMIDADE RELATIVA DO AR, HORARIA(%)': 'RM',
            'VENTO, DIRECAO HORARIA(CODIGO)': 'WD',
            'VENTO, VELOCIDADE HORARIA(M/S)': 'WS',
            'VISIBILIDADE, HORARIA(CODIGO)': 'Vis'
        }

        # reading
        self.reading_dtfields = ["Data Medicao", "Hora Medicao"]
        self.reading_dtformat = '%Y-%m-%d %H%M'

# todo docstring
class ConventionalDaily(_Conventional_):
    """Parser object for ``INMET`` Conventional Stations - Daily measurements.

    **Input format**

    The input ``csv`` file is expected to present the following structure:

    .. code-block:: text

        Nome: PELOTAS
        Codigo Estacao: 83985
        Latitude: -31.78333333
        Longitude: -52.41666666
        Altitude: 13
        Situacao: Operante
        Data Inicial: 1925-12-31
        Data Final: 2023-01-01
        Periodicidade da Medicao: Diaria

        Data Medicao;EVAPORACAO DO PICHE, DIARIA(mm);INSOLACAO TOTAL, DIARIO(h);PRECIPITACAO TOTAL, DIARIO(mm);TEMPERATURA MAXIMA, DIARIA(°C);TEMPERATURA MEDIA COMPENSADA, DIARIA(°C);TEMPERATURA MINIMA, DIARIA(°C);UMIDADE RELATIVA DO AR, MEDIA DIARIA(%);UMIDADE RELATIVA DO AR, MINIMA DIARIA(%);VENTO, VELOCIDADE MEDIA DIARIA(m/s);
        1961-01-01;4.5;null;null;null;null;18.6;74;70;2.666667;
        1961-01-02;4.9;null;0;null;null;17.4;63.5;53;3.333333;
        1961-01-03;5.4;null;0;null;null;16.4;67.75;56;2;
        1961-01-04;3.3;null;5.8;null;null;17.8;73.25;69;2;
        1961-01-05;5.7;null;0;null;null;17.8;68.5;52;5.333333;
        1961-01-06;5.8;null;5.4;null;null;17.2;64.25;60;3.333333;
        1961-01-07;3.4;null;0;null;null;9.4;77.5;62;2.666667;

    **Output format - data**

    The output ``csv`` file for data is designed to
    be parsed in the following format:

    .. code-block:: text

        DateTime;EvP;IsT;P;Temp_max;Temp_mean;Temp_min;RM_mean;RM_min;WS_mean
        1961-01-01 12:00:00;4.5;;;;;18.6;74.0;70.0;2.666667
        1961-01-02 12:00:00;4.9;;0.0;;;17.4;63.5;53.0;3.333333
        1961-01-03 12:00:00;5.4;;0.0;;;16.4;67.75;56.0;2.0
        1961-01-04 12:00:00;3.3;;5.8;;;17.8;73.25;69.0;2.0
        1961-01-05 12:00:00;5.7;;0.0;;;17.8;68.5;52.0;5.333333
        1961-01-06 12:00:00;5.8;;5.4;;;17.2;64.25;60.0;3.333333

    **Output format - metadata**

    The output ``csv`` info table for metadata is designed to
    be parsed in the following format:

    .. code-block:: text

        Field;Value
        NameStation;PELOTAS
        CodeStation;83985
        Y;-31.78333333
        X;-52.41666666
        Z;13
        Status;Operante
        StartStation;1925-12-31
        EndStation;2023-01-01
        TimeScale;Diaria

    **Output format - glossary**

    The output ``csv`` info table for ` glossary is designed to
    be parsed in the following format:

    .. code-block:: text

        Field;SourceDescription
        IsT;INSOLACAO TOTAL, DIARIO(h)
        EvP;EVAPORACAO DO PICHE, DIARIA(mm)
        P;PRECIPITACAO TOTAL, DIARIO(mm)
        Temp_max;TEMPERATURA MAXIMA, DIARIA(°C)
        Temp_mean;TEMPERATURA MEDIA COMPENSADA, DIARIA(°C)
        Temp_min;TEMPERATURA MINIMA, DIARIA(°C)
        RM_mean;UMIDADE RELATIVA DO AR, MEDIA DIARIA(%)
        RM_min;UMIDADE RELATIVA DO AR, MINIMA DIARIA(%)
        WS_mean;VENTO, VELOCIDADE MEDIA DIARIA(m/s)


    **Examples:**

    First import the module:

    >>> from plans.parsers import inmet

    Instantiate the parser and load data:

    >>> auto = inmet.ConventionalDaily()
    >>> auto.load_data(file_path="path/to/file.csv")

    To export only data:

    >>> auto.export_data(folder="path/to/folder")

    To export only metadata:

    >>> auto.export_metadata(folder="path/to/folder")

    To export only data glossary:

    >>> auto.export_glossary(folder="path/to/folder")

    To export all files at once (data, metadata and glossary files):

    >>> auto.export(folder="path/to/folder")

    """

    # todo docstring
    def __init__(self, name="MyConvetionalDaily"):
        super().__init__(name=name)
        # overwrite super() attributes:
        self.prefix = "inmet_convd"
        self.gls = {
            'INSOLACAO TOTAL, DIARIO(H)': "IsT",
            'EVAPORACAO DO PICHE, DIARIA(MM)': "EvP",
            'PRECIPITACAO TOTAL, DIARIO(MM)': "P",
            'TEMPERATURA MAXIMA, DIARIA(°C)': "Temp_max",
            'TEMPERATURA MEDIA COMPENSADA, DIARIA(°C)': "Temp_mean",
            'TEMPERATURA MINIMA, DIARIA(°C)': "Temp_min",
            'UMIDADE RELATIVA DO AR, MEDIA DIARIA(%)': "RM_mean",
            'UMIDADE RELATIVA DO AR, MINIMA DIARIA(%)': "RM_min",
            'VENTO, VELOCIDADE MEDIA DIARIA(M/S)':"WS_mean",
        }

        # reading
        self.reading_dtfields = ["Data Medicao"]
        self.reading_dtformat = '%Y-%m-%d %H:%M:%S'

# todo docstring
class _Automatic_(_StationINMET_):

    # todo docstring
    def __init__(self, name="MyAutomatic"):
        super().__init__(name=name)
        # Object attributes
        self.prefix = "inmet_auto"
        # reading specs
        self.reading_delimiter = ";"
        self.reading_skiprows = len(self.station_metadata)

# todo docstring
class AutomaticByYear(_Automatic_):
    """Parser object for ``INMET`` Automatic Stations provided by Year.
    The The input ``csv`` file is expected to present the following structure:

    .. code-block:: text

        REGIAO:;CO
        UF:;DF
        ESTACAO:;BRASILIA
        CODIGO (WMO):;A001
        LATITUDE:;-15,789343
        LONGITUDE:;-47,925756
        ALTITUDE:;1160,96
        DATA DE FUNDACAO:;07/05/00
        Data;Hora UTC;PRECIPITAÇÃO TOTAL, HORÁRIO (mm);PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB);PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB);PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB);RADIACAO GLOBAL (Kj/m²);TEMPERATURA DO AR - BULBO SECO, HORARIA (°C);TEMPERATURA DO PONTO DE ORVALHO (°C);TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C);TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C);TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C);TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C);UMIDADE REL. MAX. NA HORA ANT. (AUT) (%);UMIDADE REL. MIN. NA HORA ANT. (AUT) (%);UMIDADE RELATIVA DO AR, HORARIA (%);VENTO, DIREÇÃO HORARIA (gr) (° (gr));VENTO, RAJADA MAXIMA (m/s);VENTO, VELOCIDADE HORARIA (m/s);
        2022/01/01;0000 UTC;;;;;;;;;;;;;;;;;;
        2022/01/01;0100 UTC;0;935,4;935,4;934,9;;21,9;19,8;22,7;21,9;19,9;19,2;88;80;88;4;2,9;1,2;
        2022/01/01;0200 UTC;;;;;;;;;;;;;;;;;;
        2022/01/01;0300 UTC;0;935,1;935,7;935,1;;21,7;20,2;22,2;21,6;20,2;19,9;91;88;91;53;4;1,2;
        2022/01/01;0400 UTC;1,6;934,6;935,1;934,6;;21,3;20,1;21,7;21,2;20,2;20,1;93;91;93;32;5;3,1;
        2022/01/01;0500 UTC;0;934;934,6;934;;20,6;19,7;21,3;20,6;20,2;19,6;94;93;94;31;5,6;2,3;
        2022/01/01;0600 UTC;0;933,8;934;933,8;;20,5;19,5;20,7;20,3;19,7;19,4;95;94;95;56;3,4;1,2;
        2022/01/01;0700 UTC;0;933,7;933,9;933,7;;20,4;19,5;20,5;20,3;19,6;19,4;95;94;95;40;2,8;1,6;
        2022/01/01;0800 UTC;0;933,8;933,8;933,7;;20,6;19,7;20,6;20,4;19,7;19,5;95;95;95;32;4;2,3;
        2022/01/01;0900 UTC;3,6;934,3;934,3;933,8;;20,7;19,8;20,7;20,6;19,8;19,7;95;95;95;31;4,4;2,1;


    The output ``csv`` file is designed to be parsed in the following format:

    .. code-block:: text

        REGIAO:;CO
        UF:;DF


    **Examples:**

    >>> auto = AutomaticByYear()
    >>> auto.load_data(file_path="path/to/file.csv")

    """

    # todo docstring
    def __init__(self, name="MyAutomaticYer"):
        super().__init__(name=name)

        self.prefix = "inmet_autoy"

        self.gls = {
            "PRECIPITACAO TOTAL, HORARIO (MM)": "P",
            "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (MB)": "PALoc",
            "PRESSAO ATMOSFERICA MAX.NA HORA ANT. (AUT) (MB)": "PALoc_premax",
            "PRESSAO ATMOSFERICA MIN. NA HORA ANT. (AUT) (MB)": "PALoc_premin",
            "RADIACAO GLOBAL (KJ/M²)": "Rad",
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "TempDB",
            "TEMPERATURA DO PONTO DE ORVALHO (°C)": "TempDP",
            "TEMPERATURA MAXIMA NA HORA ANT. (AUT) (°C)": "TempDB_premax",
            "TEMPERATURA MINIMA NA HORA ANT. (AUT) (°C)": "TempDB_premin",
            "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)": "TempDP_premax",
            "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)": "TempDP_premin",
            "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)": "RM_premax",
            "UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)": "RM_premin",
            "UMIDADE RELATIVA DO AR, HORARIA (%)": "RM",
            "VENTO, DIRECAO HORARIA (GR) (° (GR))": "WD",
            "VENTO, RAJADA MAXIMA (M/S)": "WG",
            "VENTO, VELOCIDADE HORARIA (M/S)": "WS",
        }

        # data
        d1, d2 = self._metadata_util()
        self.station_metadata = d1
        self.glossary = self.get_glossary()

        # reading
        self.reading_metadata_splitter = ":;"
        self.reading_dtfields = ["DATA (YYYY-MM-DD)", "HORA (UTC)"]
        self.reading_dtformat = '%Y-%m-%d %H:%M'
        self.reading_dtfields = ["Data", "Hora UTC"]
        self.reading_dtformat = '%Y/%m/%d %H%M'
        self.reading_encoding = "latin-1"
        self.reading_decimals = ","

    # todo docstring
    def _metadata_util(self):
        d = {
            "NameStation": [None, "ESTACAO"],
            "CodeStation": [None, "CODIGO(WMO)"],
            "Y": [None, "LATITUDE"],
            "X": [None, "LONGITUDE"],
            "Z": [None, "ALTITUDE"],
            "StartStation": [None, "DATADEFUNDACAO"],
            "Region": [None, "REGIAO"],
            "State": [None, "UF"]
        }
        d1 = {}
        d2 = {}
        for k in d:
            d1[k] = d[k][0]
            d2[d[k][1]] = k

        return d1, d2

# todo docstring
class Automatic(_Automatic_):
    """Parser object for ``INMET`` Automatic Stations provided by date range.
    The input ``csv`` file is expected to present the following structure:

    .. code-block:: text

        Nome: RIO GRANDE
        Codigo Estacao: A802
        Latitude: -32.07888888
        Longitude: -52.16777777
        Altitude: 4.92
        Situacao: Operante
        Data Inicial: 2001-11-15
        Data Final: 2023-01-01
        Periodicidade da Medicao: Horaria

        Data Medicao;Hora Medicao;PRECIPITACAO TOTAL, HORARIO(mm);PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(mB);PRESSAO ATMOSFERICA REDUZIDA NIVEL DO MAR, AUT(mB);PRESSAO ATMOSFERICA MAX.NA HORA ANT. (AUT)(mB);PRESSAO ATMOSFERICA MIN. NA HORA ANT. (AUT)(mB);RADIACAO GLOBAL(Kj/m²);TEMPERATURA DA CPU DA ESTACAO(°C);TEMPERATURA DO AR - BULBO SECO, HORARIA(°C);TEMPERATURA DO PONTO DE ORVALHO(°C);TEMPERATURA MAXIMA NA HORA ANT. (AUT)(°C);TEMPERATURA MINIMA NA HORA ANT. (AUT)(°C);TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT)(°C);TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT)(°C);TENSAO DA BATERIA DA ESTACAO(V);UMIDADE REL. MAX. NA HORA ANT. (AUT)(%);UMIDADE REL. MIN. NA HORA ANT. (AUT)(%);UMIDADE RELATIVA DO AR, HORARIA(%);VENTO, DIRECAO HORARIA (gr)(° (gr));VENTO, RAJADA MAXIMA(m/s);VENTO, VELOCIDADE HORARIA(m/s);
        2001-11-17;1500;null;null;null;null;null;null;null;null;null;null;null;null;null;null;null;null;null;null;null;null;
        2001-11-17;1600;null;null;null;null;null;null;null;null;null;null;null;null;null;null;null;null;null;null;null;null;
        2001-11-17;1700;0;1017.3;1018.059963;1018;1017.3;3808;null;20.7;9;20.9;20.2;9.8;7.2;null;49;42;47;80;6.2;3.2;
        2001-11-17;1800;0;1016.4;1017.160067;1017.2;1016.4;3398;null;20.4;9.9;21.1;20.3;10.6;8.7;null;53;46;51;80;6.4;3.5;
        2001-11-17;1900;0;1015.6;1016.358952;1016.4;1015.6;2780;null;20.6;11.2;21.3;20.3;11.6;9.7;null;57;49;55;72;6.9;3.6;
        2001-11-17;2000;0;1015.2;1015.959429;1015.6;1015.2;1989;null;20.3;11.8;20.7;20.2;12.3;10.7;null;59;54;58;71;7.1;4.3;
        2001-11-17;2100;0;1015.6;1016.361805;1015.6;1015.3;1111;null;19.5;11.9;20.3;19.5;12.4;11.3;null;63;56;61;74;7.7;4.5;
        2001-11-17;2200;0;1015.4;1016.164268;1015.5;1015.4;268;null;18.5;11.4;19.4;18.5;12.5;11.4;null;65;62;63;70;8.3;5;
        2001-11-17;2300;0;1015.8;1016.566146;1015.8;1015.4;0;null;17.9;10.9;18.5;17.9;11.8;10.8;null;65;62;64;57;8.3;4.2;
        2001-11-18;0000;0;1015.9;1016.666221;1015.9;1015.8;-4;null;17.9;10.3;18.1;17.8;10.9;10.2;null;63;61;61;56;8.6;5.5;


    The output ``csv`` file is designed to be parsed in the following format:

    .. code-block:: text

        REGIAO:;CO
        UF:;DF


    **Examples:**

    >>> auto = AutomaticByYear()
    >>> auto.load_data(file_path="path/to/file.csv")
    >>> auto.data.to_csv("path/to/output.csv")

    """
    def __init__(self, name="MyAutomatic"):
        super().__init__(name=name)
        self.gls = {
            "PRECIPITACAO TOTAL, HORARIO(MM)": "P",
            "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA(MB)": "PALoc",
            "PRESSAO ATMOSFERICA REDUZIDA NIVEL DO MAR, AUT(MB)": "PASea",
            "PRESSAO ATMOSFERICA MAX.NA HORA ANT. (AUT)(MB)": "PALoc_premax",
            "PRESSAO ATMOSFERICA MIN. NA HORA ANT. (AUT)(MB)": "PALoc_premin",
            "RADIACAO GLOBAL(KJ/M²)": "Rad",
            "TEMPERATURA DA CPU DA ESTACAO(°C)": "TempCPU",
            "TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)": "TempDB",
            "TEMPERATURA DO PONTO DE ORVALHO(°C)": "TempDP",
            "TEMPERATURA MAXIMA NA HORA ANT. (AUT)(°C)": "TempDB_premax",
            "TEMPERATURA MINIMA NA HORA ANT. (AUT)(°C)": "TempDB_premin",
            "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT)(°C)": "TempDP_premax",
            "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT)(°C)": "TempDP_premin",
            "TENSAO DA BATERIA DA ESTACAO(V)": "Volt",
            "UMIDADE REL. MAX. NA HORA ANT. (AUT)(%)": "RM_premin",
            "UMIDADE REL. MIN. NA HORA ANT. (AUT)(%)": "RM_premax",
            "UMIDADE RELATIVA DO AR, HORARIA(%)": "RM",
            "VENTO, DIRECAO HORARIA (GR)(° (GR))": "WD",
            "VENTO, RAJADA MAXIMA(M/S)": "WG",
            "VENTO, VELOCIDADE HORARIA(M/S)": "WS",
        }
        # data
        d1, d2 = self._metadata_util()
        self.station_metadata = d1
        self.glossary = self.get_glossary()

        # reading
        self.reading_metadata_splitter = ": "
        self.reading_dtformat = '%Y-%m-%d %H%M'

    def _metadata_util(self):
        d = {
            "NameStation": [None, "NOME"],
            "CodeStation": [None, "CODIGOESTACAO"],
            "Y": [None, "LATITUDE"],
            "X": [None, "Longitude".upper()],
            "Z": [None, "Altitude".upper()],
            "Status": [None, "Situacao".upper()],
            "StartStation": [None, "DataInicial".upper()],
            "EndStation": [None, "DataFinal".upper()],
            "TimeScale": [None, "PeriodicidadedaMedicao".upper()]
        }
        d1 = {}
        d2 = {}
        for k in d:
            d1[k] = d[k][0]
            d2[d[k][1]] = k
        return d1, d2


if __name__ == "__main__":
    print("Hello world!")

    f = "C:/data/"


