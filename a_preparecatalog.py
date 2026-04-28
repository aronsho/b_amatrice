# ========= IMPORTS =========
from seismostats.utils import binning_test
import pandas as pd

# ======== SPECIFY PARAMETERS =========
delta_m = 0.01
training_share = 0.6
max_depth = 50.0
max_latitude = 43.1
min_latitude = 42.5
max_longitude = 13.4
min_longitude = 12.9

location = 'data/catalogs/Amatrice_CAT5.v20210325'


# ======== GET DATA ===================
cat_raw = pd.read_csv(location,
                      sep=r"\s+",
                      header=None,
                      skiprows=22,)


# ========= PREPARE DATA ===================

column_names = [
    "year", "month", "day", "hour", "minute", "second",
    "latitude", "longitude", "depth",
    "EH1", "EH2", "AZ", "EZ",
    "ML", "Mw",
    "ID"
]

cat_raw.columns = column_names

cat_all = cat_raw.copy()
cat_all['time'] = pd.to_datetime(
    cat_all[['year', 'month', 'day', 'hour', 'minute', 'second']])
cat_all['magnitude'] = cat_all['ML']
# delete the time columns
cat_all.drop(columns=['year', 'month', 'day', 'hour',
             'minute', 'second'], inplace=True)

# test binning
print(binning_test(cat_all.magnitude.values, delta_x=0.01))
cat_all.delta_m = delta_m

# remove nonan
cat_all = cat_all[~cat_all['magnitude'].isna()]

# remove events deeper than 50 km
cat_all = cat_all[cat_all['depth'] <= max_depth]
cat_all = cat_all[cat_all.latitude <= max_latitude]
cat_all = cat_all[cat_all.latitude >= min_latitude]
cat_all = cat_all[cat_all.longitude >= min_longitude]
cat_all = cat_all[cat_all.longitude <= max_longitude]


# ======== SPLIT DATA ===================
# sort df_amatrice in time
cat_all = cat_all.sort_values(by='time')

cat_train = cat_all.copy()
cat_train = cat_train.iloc[:int(training_share*len(cat_train))]
cat_test = cat_all.copy()
cat_test = cat_test.iloc[int(training_share*len(cat_test)):]

# ======== SAVE DATA ======================
cat_train.to_csv('data/training/Amatrice_CAT5_train.csv', index=False)
cat_test.to_csv('data/testing/Amatrice_CAT5_test.csv', index=False)
