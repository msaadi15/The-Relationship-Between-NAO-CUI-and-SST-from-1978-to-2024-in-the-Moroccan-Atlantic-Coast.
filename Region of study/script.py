import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point

# --- 1. Chemin du fichier GeoJSON ---
geojson_path = r"C:\Users\moham\OneDrive\Documents\fog\Morocco_ADM0_simplified.simplified.geojson"

# --- 2. Coordonnées des villes ---
cities = {
    'Dakhla': (23.71, -15.93),
    'Boujdour': (26.13, -14.48),
    'Laayoune': (27.15, -13.20),
    'Tan-Tan': (28.43, -11.10),
    'Sidi Ifni': (29.38, -10.18),
    'Agadir': (30.42, -9.58),
    'Essaouira': (31.51, -9.77),
    'Safi': (32.30, -9.24),
    'Casablanca': (33.57, -7.59),
    'Kenitra': (34.25, -6.58),
    'Tangier': (35.78, -5.81),
    'Lisbon': (38.72, -9.14),
}

# --- 3. Chargement du GeoJSON Maroc ---
gdf_maroc = gpd.read_file(geojson_path)

# --- 4. Création des points des villes ---
villes_list = []
geometry_villes = []

for name, (lat, lon) in cities.items():
    villes_list.append({'Ville': name})
    geometry_villes.append(Point(lon, lat))

gdf_villes = gpd.GeoDataFrame(
    villes_list,
    geometry=geometry_villes,
    crs="EPSG:4326"
)

# --- 5. Création de la figure Cartopy ---
fig = plt.figure(figsize=(12, 18))
ax = plt.axes(projection=ccrs.PlateCarree())

# --- 6. Ajout des coastlines (autres pays) ---
ax.add_feature(
    cfeature.COASTLINE,
    linewidth=1.0,
    edgecolor='gray',
    zorder=1
)

# --- 7. Ajout des frontières (optionnel mais propre) ---
ax.add_feature(
    cfeature.BORDERS,
    linestyle=':',
    linewidth=0.6,
    edgecolor='gray',
    zorder=1
)

# --- 8. Plot du Maroc ---
gdf_maroc.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    edgecolor='darkblue',
    facecolor='lightyellow',
    linewidth=1.2,
    zorder=2
)

# --- 9. Plot des villes ---
gdf_maroc_villes = gdf_villes[gdf_villes['Ville'] != 'Lisbon']
gdf_lisbon = gdf_villes[gdf_villes['Ville'] == 'Lisbon']

ax.scatter(
    gdf_maroc_villes.geometry.x,
    gdf_maroc_villes.geometry.y,
    color='red',
    s=60,
    transform=ccrs.PlateCarree(),
    zorder=3
)

ax.scatter(
    gdf_lisbon.geometry.x,
    gdf_lisbon.geometry.y,
    color='red',
    s=80,
    transform=ccrs.PlateCarree(),
    zorder=3
)

# --- 10. Étiquettes des villes ---
for x, y, label in zip(
    gdf_villes.geometry.x,
    gdf_villes.geometry.y,
    gdf_villes['Ville']
):
    ax.text(
        x + 0.2,
        y,
        label,
        fontsize=10,
        ha='left',
        va='center',
        transform=ccrs.PlateCarree(),
        zorder=4
    )

# --- 11. Étendue de la carte ---
ax.set_extent([-28, 0, 20, 41], crs=ccrs.PlateCarree())

# --- 12. Grille géographique ---
gl = ax.gridlines(draw_labels=True, linestyle=':', alpha=0.7)
gl.top_labels = False
gl.right_labels = False

# --- 13. Affichage ---
plt.show()
