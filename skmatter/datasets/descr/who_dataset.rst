.. _who:

who_data
#########

`who_dataset.csv` is a compilation of multiple publically-available datasets
through data.worldbank.org. Specifically, the following versioned datasets are used:

- NY.GDP.PCAP.CD (v2_4770383) [1]_
- SE.XPD.TOTL.GD.ZS (v2_4773094) [2]_
- SH.DYN.AIDS.ZS (v2_4770518) [3]_
- SH.IMM.IDPT (v2_4770682) [4]_
- SH.IMM.MEAS (v2_4774112) [5]_
- SH.TBS.INCD (v2_4770775) [6]_
- SH.XPD.CHEX.GD.ZS (v2_4771258) [7]_
- SN.ITK.DEFC.ZS (v2_4771336) [8]_
- SP.DYN.LE00.IN (v2_4770556) [9]_
- SP.POP.TOTL (v2_4770385) [10]_

where the corresponding file names are `API_{dataset}_DS2_excel_en_{version}.xls`.

This dataset, intended only for demonstration, contains 2020 country-year pairings and
the corresponding values above.

Data Set Characteristics
------------------------

    :Number of Instances: 2020

    :Number of Features: 10

References
----------

   .. [1] https://data.worldbank.org/indicator/NY.GDP.PCAP.CD
   .. [2] https://data.worldbank.org/indicator/SE.XPD.TOTL.GD.ZS
   .. [3] https://data.worldbank.org/indicator/SH.DYN.AIDS.ZS
   .. [4] https://data.worldbank.org/indicator/SH.IMM.IDPT
   .. [5] https://data.worldbank.org/indicator/SH.IMM.MEAS
   .. [6] https://data.worldbank.org/indicator/SH.TBS.INCD
   .. [7] https://data.worldbank.org/indicator/SH.XPD.CHEX.GD.ZS
   .. [8] https://data.worldbank.org/indicator/SN.ITK.DEFC.ZS
   .. [9] https://data.worldbank.org/indicator/SP.DYN.LE00.IN
   .. [10] https://data.worldbank.org/indicator/SP.POP.TOTL
   

Reference Code
--------------

and compiled through the following script, where the datasets have been placed in a folder named `who_data`:

.. code-block:: python

    import os
    import pandas as pd
    import numpy as np

    files = os.listdir('who_data/')
    indicators = [f[4:f[4:].index('_')+4] for f in files]
    indicator_codes = {}
    data_dict = {}
    entries = []

    for file in files:
        data = pd.read_excel(
            "who_data/" + file,
            header=3,
            sheet_name="Data",
            index_col=0,
        )
    
        indicator = data["Indicator Code"].values[0]
        indicator_codes[indicator] = data["Indicator Name"].values[0]

        for index in data.index:
            for year in range(1900, 2022):
                if str(year) in data.loc[index] and not np.isnan(
                    data.loc[index].loc[str(year)]
                ):
                    if (index, year) not in data_dict:
                        data_dict[(index, year)] = np.nan * np.ones(len(indicators))
                    data_dict[(index, year)][indicators.index(indicator)] = data.loc[index].loc[str(year)]

    with open('who_data.csv','w') as outf:
        outf.write('Country,Year,'+','.join(indicators)+'\n')
        for key, data in data_dict.items():
            if np.count_nonzero(~np.isnan(np.array(data, dtype=float))) == len(indicators):
                outf.write('{},{},{}\n'.format(key[0].replace(',',' '), key[1], ','.join([str(d) for d in data])))
