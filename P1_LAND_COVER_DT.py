## Project 1 - Land Classification using Decision Trees


import rasterio
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd

# For validation / diagnostics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1 = Coastal Aerosol
# 2 = Blue
# 3 = Green
# 4 = Red
# 5 = NIR
# 6 = SWIR1
# 7 = SWIR2
# 8 = Cirrus

# Label classes
# 1 = clouds
# 2 = Water
# 3 = Vegetation
# 4 = bare earth
# 5 = glacier / ice

#xds = xr.open_dataset(r"C:\UoT\3247_ers\ESRI_W1_Raster\DisplayRaster\DisplayRaster\Data\Montana\G_2014.tif")
#xds.rio.write_crs(4326, inplace=True)

source_img = r"C:\UoT\3247_ers\ESRI_W1_Raster\DisplayRaster\DisplayRaster\Data\Montana\G_2014.tif"

# Load Bands
with rasterio.open(source_img) as ca:
    ca_band = ca.read(1)

with rasterio.open(source_img) as blue:
    blue_band = blue.read(2)
    img_profile = blue.profile
    
with rasterio.open(source_img) as green:
    green_band = green.read(3)
    
with rasterio.open(source_img) as red:
    red_band = red.read(4)
    
with rasterio.open(source_img) as nir:
    nir_band = nir.read(5)
    
with rasterio.open(source_img) as swir1:
    swir1_band = swir1.read(6)
    
with rasterio.open(source_img) as swir2:
    swir2_band = swir2.read(7)

with rasterio.open(source_img) as cirrus:
    cirrus_band = cirrus.read(8)    

    
# Stack bands
X = np.dstack((ca_band,blue_band, green_band, red_band, nir_band, swir1_band, swir2_band, cirrus_band)).reshape(-1, 8)

# Import shapefile
gdf = gpd.read_file(r"C:\UoT\3247_ers_AGP\p1_labels.shp")

# Make sure the crs are the same
if gdf.crs != img_profile["crs"]:
    print('crs dont match!')
    gdf = gdf.to_crs(img_profile["crs"])



# Here I extract the value at these locations for each band
with rasterio.open(source_img) as img:
    # Here I need to extract the COORDINATES for my labels. That's it - just the coordinates
    coords = [(x,y) for x,y in zip(gdf.geometry.x, gdf.geometry.y)]
    # Sample all bands at once
    samples = list(img.sample(coords))

X_train = np.array(samples)  # shape: (n_samples, n_bands) = 87. i.e. there should be same number of rows as y_train
y_train = gdf["class"].values # shape: (n) = 87


# DECISION TREE _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#clf = DecisionTreeClassifier(max_depth=5)
#clf.fit(X_train, y_train)

# RANDOM FOREST _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
#clf.fit(X_train, y_train)

# XGBOOST _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
# Get the unique classes in your labels
original_classes = np.unique(y_train)

# Create a mapping: e.g., {1: 0, 2: 1, ..., 7: 5}
class_mapping = {label: idx for idx, label in enumerate(original_classes)}

# Apply the mapping to y_train
y_train_cleaned = np.array([class_mapping[y] for y in y_train])

clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
clf.fit(X_train, y_train_cleaned)


# _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

# Predict
preds = clf.predict(X)
preds_full = preds.astype(np.int16)  # Use a type that supports negative values
classified_image = preds_full.reshape(red_band.shape) # 

# Update raster profile safely
img_profile.update(
    dtype=rasterio.int16,     # supports negative nodata
    count=1,
    nodata=-9,
    compress='lzw'
)


# Plot
plt.imshow(classified_image, cmap='terrain')
plt.title("Land Cover Classification")
plt.colorbar()
plt.show()

# Export back to .tif

img_profile.update(
    dtype=rasterio.int16,    # or uint16, depending on number of classes
    count=1,
    compress='lzw'           # optional: compress the output
)

# Write the classified raster
with rasterio.open(r"C:\UoT\3247_ers\ESRI_W1_Raster\DisplayRaster\DisplayRaster\Data\Montana\G_2014_classified_xg_v6.tif", "w", **img_profile) as dst:
    dst.write(classified_image.astype(rasterio.int16), 1)  # write to band 1

# _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

# DIAGNOSE THE MODEL

X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

clf = RandomForestClassifier().fit(X_tr, y_tr)
y_pred = clf.predict(X_te)

print(confusion_matrix(y_te, y_pred))
print(classification_report(y_te, y_pred))

test_accuracy = clf.score(X_te, y_te)
print(test_accuracy)