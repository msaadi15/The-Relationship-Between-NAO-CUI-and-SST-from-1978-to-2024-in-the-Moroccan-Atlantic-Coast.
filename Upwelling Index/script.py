import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

def calculate_upwelling_zones_from_your_data(file_path, start_year=1978, end_year=2024):
    """Calculate upwelling intensity zones using ACTUAL NetCDF data"""
    print(f"=== CALCULATING UPWELLING ZONES {start_year}-{end_year} ===")
    print("Using ACTUAL NetCDF wind data for calculations...")
    
    dataset = nc.Dataset(file_path, 'r')
    
    # Read data
    time_var = dataset.variables['valid_time']
    lat_var = dataset.variables['latitude'][:]
    lon_var = dataset.variables['longitude'][:]
    u10_var = dataset.variables['u10']
    v10_var = dataset.variables['v10']
    
    # Time conversion
    times = nc.num2date(time_var[:], time_var.units)
    regular_times = []
    for t in times:
        if hasattr(t, 'year'):
            regular_times.append(datetime(t.year, t.month, t.day))
        else:
            regular_times.append(t)
    
    regular_times = np.array(regular_times)
    
    # Longitude conversion
    lon_var = np.where(lon_var > 180, lon_var - 360, lon_var)
    
    # Time filter
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    time_mask = (regular_times >= start_date) & (regular_times <= end_date)
    time_indices = np.where(time_mask)[0]
    
    print(f"Processing {len(time_indices)} time steps ({start_year}-{end_year})")
    
    # Study region (Moroccan Atlantic Coast)
    lat_min, lat_max = 20, 42
    lon_min, lon_max = -30, -5
    
    lat_idx = np.where((lat_var >= lat_min) & (lat_var <= lat_max))[0]
    lon_idx = np.where((lon_var >= lon_min) & (lon_var <= lon_max))[0]
    
    lats_region = lat_var[lat_idx]
    lons_region = lon_var[lon_idx]
    
    # Create land mask using SST if available
    print("Creating ocean mask...")
    land_mask_region = None
    if 'sst' in dataset.variables:
        print("  Using SST data to create accurate land mask...")
        try:
            sst_data = dataset.variables['sst'][0, :, :]
            if np.nanmax(sst_data) > 200:
                sst_data = sst_data - 273.15
            
            land_mask = (sst_data < -10) | (sst_data > 40) | np.isnan(sst_data)
            land_mask_region = land_mask[lat_idx[:, None], lon_idx]
            print(f"  Land pixels identified: {np.sum(land_mask_region)}")
        except:
            print("  Warning: Could not use SST for land mask")
    
    if land_mask_region is None:
        print("  Using wind data for land mask...")
        u10_sample = u10_var[0, :, :]
        land_mask = np.isnan(u10_sample) | (np.abs(u10_sample) < 0.01)
        land_mask_region = land_mask[lat_idx[:, None], lon_idx]
        print(f"  Land pixels identified: {np.sum(land_mask_region)}")
    
    # Calculate monthly climatology of upwelling index
    print("Calculating monthly upwelling climatology...")
    
    # Initialize arrays for monthly means
    monthly_upwelling = np.zeros((12, len(lat_idx), len(lon_idx)))
    monthly_counts = np.zeros(12)
    
    # Physical constants
    rho_air = 1.22         # kg/m³
    rho_water = 1025.0     # kg/m³
    C_d = 1.3e-3           # drag coefficient
    omega = 7.2921e-5      # Earth's rotation rate (rad/s)
    
    # Process in chunks to save memory
    chunk_size = 12
    
    for chunk_start in range(0, len(time_indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(time_indices))
        chunk_indices = time_indices[chunk_start:chunk_end]
        
        u10_chunk = u10_var[chunk_indices, :, :]
        v10_chunk = v10_var[chunk_indices, :, :]
        
        for idx, time_idx in enumerate(chunk_indices):
            month = regular_times[time_idx].month - 1
            
            u10_region = u10_chunk[idx, lat_idx[:, None], lon_idx]
            v10_region = v10_chunk[idx, lat_idx[:, None], lon_idx]
            
            # Calculate upwelling index
            Vmag = np.sqrt(u10_region**2 + v10_region**2)
            
            f = 2 * omega * np.sin(np.deg2rad(lats_region))[:, None]
            f = np.where(np.abs(f) < 1e-10, np.nan, f)
            
            My = (rho_air * C_d * Vmag * v10_region) / (rho_water * f)
            upwelling_index = -My * 100.0
            
            upwelling_index[land_mask_region] = np.nan
            
            monthly_upwelling[month] += upwelling_index
            monthly_counts[month] += 1
    
    # Calculate monthly climatology
    for month in range(12):
        if monthly_counts[month] > 0:
            monthly_upwelling[month] /= monthly_counts[month]
    
    # Calculate annual mean upwelling
    annual_upwelling = np.nanmean(monthly_upwelling, axis=0)
    
    # Apply light Gaussian smoothing
    sigma = 0.3
    annual_upwelling_smooth = annual_upwelling.copy()
    ocean_mask = ~np.isnan(annual_upwelling)
    
    if np.any(ocean_mask):
        temp_array_filled = np.where(ocean_mask, annual_upwelling, 0)
        smoothed = gaussian_filter(temp_array_filled, sigma=sigma)
        count = gaussian_filter(ocean_mask.astype(float), sigma=sigma)
        annual_upwelling_smooth = np.where(count > 0.01, smoothed / count, np.nan)
        annual_upwelling_smooth[land_mask_region] = np.nan
    
    dataset.close()
    
    print(f"Calculated upwelling range: {np.nanmin(annual_upwelling_smooth):.2f} to {np.nanmax(annual_upwelling_smooth):.2f} m³/s/100m")
    
    return annual_upwelling_smooth, lats_region, lons_region, monthly_upwelling, land_mask_region

def analyze_actual_upwelling_data(annual_upwelling):
    """Analyze ACTUAL computed upwelling data to determine thresholds"""
    print("=== ANALYZING ACTUAL COMPUTED UPWELLING DATA FOR THRESHOLDS ===")
    
    # Remove NaN values (land, invalid data)
    valid_upwelling = annual_upwelling[~np.isnan(annual_upwelling)]
    
    if len(valid_upwelling) > 0:
        print(f"Analyzing {len(valid_upwelling)} valid ocean pixels...")
        
        # Calculate percentiles from ACTUAL data
        p10 = np.percentile(valid_upwelling, 10)
        p25 = np.percentile(valid_upwelling, 25)
        p50 = np.percentile(valid_upwelling, 50)
        p75 = np.percentile(valid_upwelling, 75)
        p90 = np.percentile(valid_upwelling, 90)
        
        print(f"Actual upwelling data statistics:")
        print(f"  Min: {np.min(valid_upwelling):.2f} m³/s/100m")
        print(f"  10th percentile: {p10:.2f} m³/s/100m")
        print(f"  25th percentile: {p25:.2f} m³/s/100m")
        print(f"  50th percentile (median): {p50:.2f} m³/s/100m")
        print(f"  75th percentile: {p75:.2f} m³/s/100m")
        print(f"  90th percentile: {p90:.2f} m³/s/100m")
        print(f"  Max: {np.max(valid_upwelling):.2f} m³/s/100m")
        print(f"  Mean: {np.mean(valid_upwelling):.2f} m³/s/100m")
        print(f"  Std Dev: {np.std(valid_upwelling):.2f} m³/s/100m")
        
        # Determine thresholds based on percentiles of ACTUAL data
        # Use more natural breaks: 25%, 50%, 75% percentiles
        weak_threshold = p25      # Bottom 25% = Weak
        moderate_threshold = p50  # 25-50% = Moderate  
        strong_threshold = p75    # 50-75% = Strong
        # >75% = Very Strong
        
        print(f"\nDynamic classification thresholds from ACTUAL NetCDF data:")
        print(f"  Very Strong Upwelling: > {strong_threshold:.1f} m³/s/100m (top 25%)")
        print(f"  Strong Upwelling: {moderate_threshold:.1f} to {strong_threshold:.1f} m³/s/100m (25-50%)")
        print(f"  Moderate Upwelling: {weak_threshold:.1f} to {moderate_threshold:.1f} m³/s/100m (25-50%)")
        print(f"  Weak Upwelling: < {weak_threshold:.1f} m³/s/100m (bottom 25%)")
        
        return weak_threshold, moderate_threshold, strong_threshold
    else:
        print("ERROR: No valid upwelling data computed from NetCDF!")
        print("Using fallback thresholds (20, 60, 90)")
        return 20.0, 60.0, 90.0

def classify_upwelling_zones_based_on_actual_data(upwelling_data, land_mask, thresholds):
    """Classify upwelling zones using thresholds from ACTUAL data"""
    print("=== CLASSIFYING UPWELLING ZONES USING ACTUAL DATA THRESHOLDS ===")
    
    weak_threshold, moderate_threshold, strong_threshold = thresholds
    
    print(f"Classification thresholds from ACTUAL NetCDF data:")
    print(f"  Very Strong Upwelling: > {strong_threshold:.1f} m³/s/100m")
    print(f"  Strong Upwelling: {moderate_threshold:.1f} to {strong_threshold:.1f} m³/s/100m")
    print(f"  Moderate Upwelling: {weak_threshold:.1f} to {moderate_threshold:.1f} m³/s/100m")
    print(f"  Weak Upwelling: < {weak_threshold:.1f} m³/s/100m")
    
    # Create classification map
    zone_map = np.full_like(upwelling_data, np.nan)
    
    # Apply land mask first
    zone_map[land_mask] = np.nan
    
    # Classify zones
    ocean_mask = ~np.isnan(upwelling_data) & ~land_mask
    
    if np.any(ocean_mask):
        ocean_data = upwelling_data[ocean_mask]
        
        # Classify based on ACTUAL thresholds
        weak_mask = ocean_data <= weak_threshold
        moderate_mask = (ocean_data > weak_threshold) & (ocean_data <= moderate_threshold)
        strong_mask = (ocean_data > moderate_threshold) & (ocean_data <= strong_threshold)
        very_strong_mask = ocean_data > strong_threshold
        
        # Assign values to zone_map
        zone_map.flat[np.flatnonzero(ocean_mask)[weak_mask]] = 1      # Weak
        zone_map.flat[np.flatnonzero(ocean_mask)[moderate_mask]] = 2   # Moderate
        zone_map.flat[np.flatnonzero(ocean_mask)[strong_mask]] = 3     # Strong
        zone_map.flat[np.flatnonzero(ocean_mask)[very_strong_mask]] = 4 # Very Strong
        
        print(f"Zone distribution:")
        print(f"  Weak: {np.sum(weak_mask)} pixels ({np.sum(weak_mask)/len(ocean_data)*100:.1f}%)")
        print(f"  Moderate: {np.sum(moderate_mask)} pixels ({np.sum(moderate_mask)/len(ocean_data)*100:.1f}%)")
        print(f"  Strong: {np.sum(strong_mask)} pixels ({np.sum(strong_mask)/len(ocean_data)*100:.1f}%)")
        print(f"  Very Strong: {np.sum(very_strong_mask)} pixels ({np.sum(very_strong_mask)/len(ocean_data)*100:.1f}%)")
    
    return zone_map

def plot_upwelling_zones_corrected(zone_map, lats, lons, thresholds, start_year=1978, end_year=2024):
    """Create map of upwelling intensity zones with ACTUAL data colors"""
    print("=== CREATING UPWELLING ZONE MAP WITH ACTUAL DATA ===")
    
    # Define region boundaries
    lat_min, lat_max = 20, 42
    lon_min, lon_max = -30, -5
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Use Cartopy features
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, color='white', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=2)
    
    # Color scheme: Blue (weak) to Red (strong)
    colors = ['#2166ac', '#67a9cf', '#fddbc7', '#ef8a62', '#b2182b']
    zone_labels = ['Weak', 'Moderate', 'Strong', 'Very Strong']
    weak_thresh, mod_thresh, strong_thresh = thresholds
    zone_descriptions = [
        f'Weak Upwelling (<{weak_thresh:.0f} m³/s/100m)',
        f'Moderate Upwelling ({weak_thresh:.0f}-{mod_thresh:.0f} m³/s/100m)', 
        f'Strong Upwelling ({mod_thresh:.0f}-{strong_thresh:.0f} m³/s/100m)',
        f'Very Strong Upwelling (>{strong_thresh:.0f} m³/s/100m)'
    ]
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Plot zones
    zone_masked = np.ma.masked_where(np.isnan(zone_map), zone_map)
    
    # Use RdBu_r colormap: Red=strong (4), Blue=weak (1)
    levels = [0.5, 1.5, 2.5, 3.5, 4.5]
    cmap = plt.cm.RdBu_r
    
    # Plot filled contours
    contour = ax.contourf(lon_grid, lat_grid, zone_masked, 
                         levels=levels,
                         cmap=cmap,
                         transform=ccrs.PlateCarree(),
                         alpha=0.8, zorder=1, extend='both')
    
    # Add contour lines
    cs = ax.contour(lon_grid, lat_grid, zone_masked, levels=levels,
                    colors='black', linewidths=0.8, 
                    transform=ccrs.PlateCarree(), zorder=2)
    
    # Add contour labels
    plt.clabel(cs, inline=True, fontsize=9, fmt=lambda x: zone_labels[int(x-1)] if 1 <= x <= 4 else '')
    
    # Add major coastal cities
    cities = {
        'Dakhla': (23.71, -15.93), 'Boujdour': (26.13, -14.48), 
        'Laayoune': (27.15, -13.20), 'Tan-Tan': (28.43, -11.10),
        'Sidi Ifni': (29.38, -10.18), 'Agadir': (30.42, -9.58),
        'Essaouira': (31.51, -9.77), 'Safi': (32.30, -9.24),
        'Casablanca': (33.57, -7.59), 'Kenitra': (34.25, -6.58),
        'Tangier': (35.78, -5.81), 'Lisbon': (38.72, -9.14)
    }
    
    for city, (lat, lon) in cities.items():
        ax.plot(lon, lat, 'ko', markersize=4, transform=ccrs.PlateCarree(), zorder=3)
        ax.text(lon + 0.2, lat + 0.1, city, fontsize=8, transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8), zorder=3)
    
    # Grid lines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xlocator = plt.FixedLocator([-30, -25, -20, -15, -10, -5])
    gl.ylocator = plt.FixedLocator([20, 25, 30, 35, 40, 42])
    
    # Create custom legend
    from matplotlib.patches import Patch
    
    legend_colors = ['#2166ac', '#67a9cf', '#fddbc7', '#ef8a62']
    legend_elements = []
    for i, (color, label) in enumerate(zip(legend_colors, zone_descriptions)):
        legend_elements.append(Patch(facecolor=color, edgecolor='black', 
                                    label=label, alpha=0.8))
    
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, 
              framealpha=0.9, fancybox=True)
    
    # Title
    title = f'Upwelling Intensity Zones - Moroccan Atlantic Coast\n'
    title += f'Annual Mean ({start_year}-{end_year})\n'
    title += f'Ekman Transport per 100m Coastline (m³/s/100m)\n'
    title += f'Thresholds from ACTUAL NetCDF data'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_path = "C:/Users/moham/OneDrive/Documents/Climate_Project/"
    filename = f'upwelling_intensity_zones_{start_year}_{end_year}_actual.png'
    plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Map saved as: {filename}")
    
    return fig

def plot_upwelling_magnitude_map_corrected(upwelling_data, lats, lons, land_mask, thresholds, start_year=1978, end_year=2024):
    """Create a continuous color map of upwelling magnitude"""
    print("=== CREATING UPWELLING MAGNITUDE MAP FROM ACTUAL DATA ===")
    
    # Define region boundaries
    lat_min, lat_max = 20, 42
    lon_min, lon_max = -30, -5
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Use Cartopy features
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, color='white', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=2)
    
    # Apply land mask
    upwelling_masked = upwelling_data.copy()
    upwelling_masked[land_mask] = np.nan
    
    # Plot upwelling magnitude
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Get ocean data
    ocean_data = upwelling_masked[~np.isnan(upwelling_masked)]
    
    if len(ocean_data) > 0:
        # Determine data range
        vmin = np.percentile(ocean_data, 5)
        vmax = np.percentile(ocean_data, 95)
        
        # Use RdBu_r colormap
        cmap = plt.cm.RdBu_r
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        
        # Mask NaN values
        upwelling_plot = np.ma.masked_where(np.isnan(upwelling_masked), upwelling_masked)
        
        # Plot
        cf = ax.pcolormesh(lon_grid, lat_grid, upwelling_plot, 
                          cmap=cmap, norm=norm,
                          transform=ccrs.PlateCarree(), alpha=0.9, zorder=1, shading='auto')
        
        # Add threshold lines
        weak_thresh, mod_thresh, strong_thresh = thresholds
        
        # Plot threshold contours
        threshold_levels = [weak_thresh, mod_thresh, strong_thresh]
        threshold_labels = ['Weak/Moderate', 'Moderate/Strong', 'Strong/Very Strong']
        
        cs = ax.contour(lon_grid, lat_grid, upwelling_plot, levels=threshold_levels,
                       colors=['blue', 'green', 'red'], linewidths=1.5, 
                       transform=ccrs.PlateCarree(), zorder=2)
        
        # Label the threshold contours
        fmt = {}
        for level, label in zip(threshold_levels, threshold_labels):
            fmt[level] = f'{label}\n({level:.0f} m³/s/100m)'
        
        plt.clabel(cs, inline=True, fontsize=8, fmt=fmt, colors='black')
        
        # Add coastline highlighting
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='darkgray', zorder=3)
    
    # Add cities
    cities = {
        'Dakhla': (23.71, -15.93), 'Agadir': (30.42, -9.58),
        'Essaouira': (31.51, -9.77), 'Safi': (32.30, -9.24),
        'Casablanca': (33.57, -7.59), 'Tangier': (35.78, -5.81)
    }
    
    for city, (lat, lon) in cities.items():
        ax.plot(lon, lat, 'ko', markersize=4, transform=ccrs.PlateCarree(), zorder=4)
        ax.text(lon + 0.2, lat + 0.1, city, fontsize=8, transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8), zorder=4)
    
    # Grid
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    
    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
    cbar.set_label('Ekman Transport (m³/s per 100m coastline)\nBlue=Weak, Red=Strong', 
                  fontsize=11, fontweight='bold')
    
    # Add threshold lines to colorbar
    cbar.ax.axhline(y=weak_thresh, color='blue', linewidth=2)
    cbar.ax.axhline(y=mod_thresh, color='green', linewidth=2)
    cbar.ax.axhline(y=strong_thresh, color='red', linewidth=2)
    
    # Calculate statistics
    if len(ocean_data) > 0:
        mean_upwelling = np.mean(ocean_data)
        std_upwelling = np.std(ocean_data)
        
        # Title
        ax.set_title(f'Upwelling Magnitude - Moroccan Atlantic Coast\n'
                    f'Annual Mean ({start_year}-{end_year})\n'
                    f'Mean: {mean_upwelling:.1f} ± {std_upwelling:.1f} m³/s/100m\n'
                    f'Thresholds computed from ACTUAL NetCDF data',
                    fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_path = "C:/Users/moham/OneDrive/Documents/Climate_Project/"
    filename = f'upwelling_magnitude_{start_year}_{end_year}_actual.png'
    plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Magnitude map saved as: {filename}")
    
    return fig

def analyze_upwelling_zones_corrected(file_path, start_year=1978, end_year=2024):
    """Complete upwelling zone analysis using ACTUAL NetCDF data"""
    print("="*80)
    print(f"UPWELLING ZONE ANALYSIS {start_year}-{end_year}")
    print("Using ACTUAL NetCDF data for ALL calculations")
    print("="*80)
    
    # Calculate upwelling data FROM NetCDF
    annual_upwelling, lats, lons, monthly_upwelling, land_mask = calculate_upwelling_zones_from_your_data(
        file_path, start_year, end_year
    )
    
    # Analyze ACTUAL data to get thresholds
    thresholds = analyze_actual_upwelling_data(annual_upwelling)
    
    # Classify zones using ACTUAL thresholds
    zone_map = classify_upwelling_zones_based_on_actual_data(annual_upwelling, land_mask, thresholds)
    
    # Create maps
    fig1 = plot_upwelling_zones_corrected(zone_map, lats, lons, thresholds, start_year, end_year)
    fig2 = plot_upwelling_magnitude_map_corrected(annual_upwelling, lats, lons, land_mask, thresholds, start_year, end_year)
    
    # Calculate statistics
    print("\n=== FINAL UPWELLING ZONE STATISTICS ===")
    
    ocean_mask = ~np.isnan(zone_map)
    zone_map_ocean = zone_map[ocean_mask]
    upwelling_ocean = annual_upwelling[ocean_mask]
    
    if len(zone_map_ocean) > 0:
        zone_labels = ['Weak', 'Moderate', 'Strong', 'Very Strong']
        
        for zone_level, label in enumerate(zone_labels, 1):
            count = np.sum(zone_map_ocean == zone_level)
            percentage = (count / len(zone_map_ocean)) * 100
            print(f"  {label} Upwelling: {count} pixels ({percentage:.1f}%)")
        
        # Find strongest upwelling locations
        strong_indices = np.where(zone_map == 4)
        if len(strong_indices[0]) > 0:
            idx = np.argmax(annual_upwelling[strong_indices])
            max_lat_idx = strong_indices[0][idx]
            max_lon_idx = strong_indices[1][idx]
            max_lat = lats[max_lat_idx]
            max_lon = lons[max_lon_idx]
            max_value = annual_upwelling[max_lat_idx, max_lon_idx]
            print(f"\n  Maximum upwelling: {max_value:.1f} m³/s/100m at {max_lat:.1f}°N, {max_lon:.1f}°W")
    
    # Save detailed statistics
    output_path = "C:/Users/moham/OneDrive/Documents/Climate_Project/"
    
    # Save raw data for verification
    np.save(os.path.join(output_path, f'upwelling_data_{start_year}_{end_year}.npy'), annual_upwelling)
    np.save(os.path.join(output_path, f'zone_map_{start_year}_{end_year}.npy'), zone_map)
    
    print(f"\nRaw data saved as:")
    print(f"  upwelling_data_{start_year}_{end_year}.npy")
    print(f"  zone_map_{start_year}_{end_year}.npy")
    
    return {
        'annual_upwelling': annual_upwelling,
        'zone_map': zone_map,
        'lats': lats,
        'lons': lons,
        'land_mask': land_mask,
        'thresholds': thresholds
    }

# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":
    # Set file path
    file_path = "C:/Users/moham/OneDrive/Documents/Climate_Project/dadefda24611707dd32599a670df250b.nc"
    
    print("Starting upwelling zone analysis using ACTUAL NetCDF data...")
    print(f"Data file: {file_path}")
    print(f"Output directory: C:/Users/moham/OneDrive/Documents/Climate_Project/")
    print("\nThis version:")
    print("  1. Calculates upwelling from ACTUAL NetCDF wind data")
    print("  2. Computes thresholds from ACTUAL calculated upwelling values")
    print("  3. Classifies zones based on ACTUAL data percentiles")
    print("  4. No fake sample data - everything from NetCDF file")
    
    try:
        # Run analysis
        results = analyze_upwelling_zones_corrected(file_path, start_year=1978, end_year=2024)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("Generated files (ALL from ACTUAL NetCDF data):")
        print(f"  1. upwelling_intensity_zones_1978_2024_actual.png - Zone classification map")
        print(f"  2. upwelling_magnitude_1978_2024_actual.png - Continuous magnitude map")
        print(f"  3. upwelling_data_1978_2024.npy - Raw upwelling data (for verification)")
        print(f"  4. zone_map_1978_2024.npy - Raw zone classification (for verification)")
        print("\nAll calculations and thresholds derived from ACTUAL NetCDF data.")
        
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("An error occurred during analysis.")
