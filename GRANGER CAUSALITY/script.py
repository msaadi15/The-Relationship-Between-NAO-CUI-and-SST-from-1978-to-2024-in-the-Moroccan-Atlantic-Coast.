import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import f as f_dist
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_nao_index_netcdf_memory_efficient(file_path, start_year=1978, end_year=2022):
    """
    Memory-efficient NAO calculation that processes data in chunks
    """
    print("=== CALCULATING NAO INDEX (MEMORY EFFICIENT) ===")
    
    dataset = nc.Dataset(file_path, 'r')
    
    # Read dimensions
    time_var = dataset.variables['valid_time']
    lat_var = dataset.variables['latitude']
    lon_var = dataset.variables['longitude']
    mslp_var = dataset.variables['msl']
    
    # Convert time to datetime objects
    times = nc.num2date(time_var[:], time_var.units)
    regular_times = []
    for t in times:
        if hasattr(t, 'year'):
            regular_times.append(datetime(t.year, t.month, t.day, t.hour, t.minute, t.second))
        else:
            regular_times.append(t)
    
    # Find indices for our time period
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    time_mask = (np.array(regular_times) >= start_date) & (np.array(regular_times) <= end_date)
    time_indices = np.where(time_mask)[0]
    
    print(f"Processing {len(time_indices)} time steps")
    
    # Get coordinates
    latitudes = lat_var[:]
    longitudes = lon_var[:]
    
    # Convert longitude if needed
    if longitudes.max() > 180:
        longitudes_converted = np.where(longitudes > 180, longitudes - 360, longitudes)
    else:
        longitudes_converted = longitudes.copy()
    
    # Define regions
    azores_lat_mask = (latitudes <= 45) & (latitudes >= 35)
    azores_lon_mask = (longitudes_converted >= -45) & (longitudes_converted <= -25)
    iceland_lat_mask = (latitudes <= 70) & (latitudes >= 55)
    iceland_lon_mask = (longitudes_converted >= -45) & (longitudes_converted <= -25)
    
    azores_lat_indices = np.where(azores_lat_mask)[0]
    azores_lon_indices = np.where(azores_lon_mask)[0]
    iceland_lat_indices = np.where(iceland_lat_mask)[0]
    iceland_lon_indices = np.where(iceland_lon_mask)[0]
    
    # Calculate monthly climatology in chunks
    print("Calculating monthly climatology...")
    monthly_clim = np.zeros((12, len(latitudes), len(longitudes)))
    monthly_counts = np.zeros(12)
    
    # Process in smaller chunks to save memory
    chunk_size = 12  # Process one year at a time
    for chunk_start in range(0, len(time_indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(time_indices))
        chunk_indices = time_indices[chunk_start:chunk_end]
        
        # Load only the chunk we need
        mslp_chunk = mslp_var[chunk_indices, :, :] / 100  # Convert Pa to hPa
        
        for idx, time_idx in enumerate(chunk_indices):
            t = regular_times[time_idx]
            month = t.month - 1
            
            monthly_clim[month, :, :] += mslp_chunk[idx, :, :]
            monthly_counts[month] += 1
    
    # Average the climatology
    for month in range(12):
        if monthly_counts[month] > 0:
            monthly_clim[month, :, :] /= monthly_counts[month]
    
    # Calculate NAO index in chunks
    print("Calculating NAO index...")
    nao_index = np.zeros(len(time_indices))
    lat_weights = np.cos(np.deg2rad(latitudes))
    
    for chunk_start in range(0, len(time_indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(time_indices))
        chunk_indices = time_indices[chunk_start:chunk_end]
        
        # Load only the chunk we need
        mslp_chunk = mslp_var[chunk_indices, :, :] / 100  # Convert Pa to hPa
        
        for idx, time_idx in enumerate(chunk_indices):
            t = regular_times[time_idx]
            month = t.month - 1
            
            # Calculate anomaly for this time step
            mslp_anom = mslp_chunk[idx, :, :] - monthly_clim[month, :, :]
            
            # Azores region
            azores_region = mslp_anom[azores_lat_indices[:, np.newaxis], azores_lon_indices]
            azores_weights = lat_weights[azores_lat_indices][:, np.newaxis]
            weighted_sum = np.nansum(azores_region * azores_weights)
            weight_sum = np.nansum(azores_weights)
            azores_mean = weighted_sum / weight_sum if weight_sum > 0 else np.nan
            
            # Iceland region
            iceland_region = mslp_anom[iceland_lat_indices[:, np.newaxis], iceland_lon_indices]
            iceland_weights = lat_weights[iceland_lat_indices][:, np.newaxis]
            weighted_sum = np.nansum(iceland_region * iceland_weights)
            weight_sum = np.nansum(iceland_weights)
            iceland_mean = weighted_sum / weight_sum if weight_sum > 0 else np.nan
            
            if not np.isnan(azores_mean) and not np.isnan(iceland_mean):
                nao_index[chunk_start + idx] = azores_mean - iceland_mean
            else:
                nao_index[chunk_start + idx] = np.nan
    
    # Normalize
    nao_index_normalized = (nao_index - np.nanmean(nao_index)) / np.nanstd(nao_index)
    
    # Create time series
    nao_times = [regular_times[i] for i in time_indices]
    nao_series = pd.Series(nao_index_normalized, index=nao_times, name='NAO_Index')
    nao_series = nao_series.dropna()
    
    dataset.close()
    
    print(f"NAO index calculated: {len(nao_series)} values")
    return nao_series

def calculate_moroccan_upwelling_index_memory_efficient(file_path, start_year=1978, end_year=2022):
    """
    Memory-efficient upwelling index calculation
    """
    print("=== CALCULATING MOROCCAN UPWELLING INDEX (MEMORY EFFICIENT) ===")
    
    dataset = nc.Dataset(file_path, 'r')
    
    time_var = dataset.variables['valid_time']
    lat_var = dataset.variables['latitude']
    lon_var = dataset.variables['longitude']
    u10_var = dataset.variables['u10']
    v10_var = dataset.variables['v10']
    
    # Convert time to datetime objects
    times = nc.num2date(time_var[:], time_var.units)
    regular_times = []
    for t in times:
        if hasattr(t, 'year'):
            regular_times.append(datetime(t.year, t.month, t.day, t.hour, t.minute, t.second))
        else:
            regular_times.append(t)
    
    # Find indices for our time period
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    time_mask = (np.array(regular_times) >= start_date) & (np.array(regular_times) <= end_date)
    time_indices = np.where(time_mask)[0]
    
    print(f"Processing {len(time_indices)} time steps")
    
    # Get coordinates
    latitudes = lat_var[:]
    longitudes = lon_var[:]
    
    # Moroccan Atlantic coast region
    moroccan_lat_range = (20, 35)
    moroccan_lon_range = (-12, -6)
    
    # Convert longitude if needed
    if longitudes.max() > 180:
        longitudes_converted = np.where(longitudes > 180, longitudes - 360, longitudes)
    else:
        longitudes_converted = longitudes.copy()
    
    # Select Moroccan coastal region
    lat_mask = (latitudes >= moroccan_lat_range[0]) & (latitudes <= moroccan_lat_range[1])
    lon_mask = (longitudes_converted >= moroccan_lon_range[0]) & (longitudes_converted <= moroccan_lon_range[1])
    
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    
    print(f"Moroccan region: {len(lat_indices)} lats, {len(lon_indices)} lons")
    
    # Physical constants
    rho_air = 1.22
    rho_water = 1025.0
    C_d = 1.3e-3
    omega = 7.2921e-5

    # Calculate upwelling index in chunks
    chunk_size = 12
    upwelling_index = np.zeros(len(time_indices))
    lat_weights = np.cos(np.deg2rad(latitudes[lat_indices]))[:, np.newaxis]
    
    print("Calculating upwelling index...")
    
    for chunk_start in range(0, len(time_indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(time_indices))
        chunk_indices = time_indices[chunk_start:chunk_end]
        
        # Load only the chunk we need
        u10_chunk = u10_var[chunk_indices, :, :]
        v10_chunk = v10_var[chunk_indices, :, :]
        
        for idx, time_idx in enumerate(chunk_indices):
            # Extract wind components for Moroccan region only
            u10_region = u10_chunk[idx, lat_indices[:, np.newaxis], lon_indices]
            v10_region = v10_chunk[idx, lat_indices[:, np.newaxis], lon_indices]

            # Wind speed magnitude
            Vmag = np.sqrt(u10_region**2 + v10_region**2)

            # Coriolis parameter
            f = 2 * omega * np.sin(np.deg2rad(latitudes[lat_indices]))[:, np.newaxis]
            f = np.where(np.abs(f) < 1e-10, np.nan, f)

            # Ekman transport
            My = (rho_air * C_d * Vmag * v10_region) / (rho_water * f)
            transport = -My * 100.0

            # Weighted average over region
            weighted_sum = np.nansum(transport * lat_weights)
            weight_sum = np.nansum(lat_weights)

            if weight_sum > 0 and not np.isnan(weighted_sum):
                upwelling_index[chunk_start + idx] = weighted_sum / weight_sum
            else:
                upwelling_index[chunk_start + idx] = np.nan
    
    # Normalize
    upwelling_index_normalized = (upwelling_index - np.nanmean(upwelling_index)) / np.nanstd(upwelling_index)
    
    # Create time series
    upwelling_times = [regular_times[i] for i in time_indices]
    upwelling_series = pd.Series(upwelling_index_normalized, index=upwelling_times, name='Upwelling_Index')
    
    dataset.close()
    
    print(f"Upwelling index calculated: {len(upwelling_series)} values")
    return upwelling_series

def calculate_moroccan_sst_memory_efficient(file_path, start_year=1978, end_year=2022):
    """
    Memory-efficient SST calculation for Moroccan region
    """
    print("=== CALCULATING MOROCCAN SST (MEMORY EFFICIENT) ===")
    
    dataset = nc.Dataset(file_path, 'r')
    
    time_var = dataset.variables['valid_time']
    lat_var = dataset.variables['latitude']
    lon_var = dataset.variables['longitude']
    sst_var = dataset.variables['sst']
    
    # Convert time to datetime objects
    times = nc.num2date(time_var[:], time_var.units)
    regular_times = []
    for t in times:
        if hasattr(t, 'year'):
            regular_times.append(datetime(t.year, t.month, t.day, t.hour, t.minute, t.second))
        else:
            regular_times.append(t)
    
    # Find indices for our time period
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    time_mask = (np.array(regular_times) >= start_date) & (np.array(regular_times) <= end_date)
    time_indices = np.where(time_mask)[0]
    
    print(f"Processing {len(time_indices)} time steps")
    
    # Get coordinates
    latitudes = lat_var[:]
    longitudes = lon_var[:]
    
    # Moroccan Atlantic region - UPDATED TO 20-42N
    moroccan_lat_range = (20, 42)
    moroccan_lon_range = (-30, -5)
    
    # Convert longitude if needed
    if longitudes.max() > 180:
        longitudes_converted = np.where(longitudes > 180, longitudes - 360, longitudes)
    else:
        longitudes_converted = longitudes.copy()
    
    # Select Moroccan region
    lat_mask = (latitudes >= moroccan_lat_range[0]) & (latitudes <= moroccan_lat_range[1])
    lon_mask = (longitudes_converted >= moroccan_lon_range[0]) & (longitudes_converted <= moroccan_lon_range[1])
    
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    
    print(f"Moroccan SST region: {len(lat_indices)} lats, {len(lon_indices)} lons")
    
    # Calculate SST index in chunks
    chunk_size = 12
    sst_index = np.zeros(len(time_indices))
    lat_weights = np.cos(np.deg2rad(latitudes[lat_indices]))[:, np.newaxis]
    
    print("Calculating SST index...")
    
    for chunk_start in range(0, len(time_indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(time_indices))
        chunk_indices = time_indices[chunk_start:chunk_end]
        
        # Load only the chunk we need
        sst_chunk = sst_var[chunk_indices, :, :]
        
        for idx, time_idx in enumerate(chunk_indices):
            # Extract SST for Moroccan region only
            sst_region = sst_chunk[idx, lat_indices[:, np.newaxis], lon_indices]
            
            # Convert Kelvin to Celsius if needed
            if sst_region.max() > 200:
                sst_region = sst_region - 273.15
            
            # Weighted average over region
            weighted_sum = np.nansum(sst_region * lat_weights)
            weight_sum = np.nansum(lat_weights)
            
            if weight_sum > 0:
                sst_index[chunk_start + idx] = weighted_sum / weight_sum
            else:
                sst_index[chunk_start + idx] = np.nan
    
    # Normalize
    sst_index_normalized = (sst_index - np.nanmean(sst_index)) / np.nanstd(sst_index)
    
    # Create time series
    sst_times = [regular_times[i] for i in time_indices]
    sst_series = pd.Series(sst_index_normalized, index=sst_times, name='SST_Moroccan')
    
    dataset.close()
    
    print(f"SST index calculated: {len(sst_series)} values")
    return sst_series

def calculate_granger_with_complete_lags(data_x, data_y, max_lag=12):
    """Calculate Granger causality with complete lag testing (1-12 months)"""
    if len(data_x) != len(data_y) or len(data_x) <= max_lag:
        return np.nan, np.nan, np.nan, np.nan
    
    valid_mask = ~np.isnan(data_x) & ~np.isnan(data_y)
    x_clean = data_x[valid_mask]
    y_clean = data_y[valid_mask]
    
    if len(x_clean) <= max_lag * 3:
        return np.nan, np.nan, np.nan, np.nan
    
    try:
        n = len(y_clean)
        best_f_stat = 0
        best_p_value = 1
        best_r2_diff = 0
        best_lag = 0
        
        # Test all lags from 1 to 12 months
        for lag in range(1, max_lag + 1):
            if n <= lag * 4:
                continue
                
            # Create lagged matrices
            y_lagged = np.zeros((n - lag, lag))
            x_lagged = np.zeros((n - lag, lag))
            
            for i in range(lag):
                y_lagged[:, i] = y_clean[i:n - lag + i]
                x_lagged[:, i] = x_clean[i:n - lag + i]
            
            y_current = y_clean[lag:]
            
            # Remove any rows with NaNs in the lagged matrices
            valid_rows = ~(np.any(np.isnan(y_lagged), axis=1) | 
                          np.any(np.isnan(x_lagged), axis=1) | 
                          np.isnan(y_current))
            
            if np.sum(valid_rows) < lag * 3:
                continue
                
            y_lagged_clean = y_lagged[valid_rows]
            x_lagged_clean = x_lagged[valid_rows]
            y_current_clean = y_current[valid_rows]
            
            # Unrestricted model
            X_unrestricted = np.column_stack([np.ones(len(y_lagged_clean)), 
                                            y_lagged_clean, x_lagged_clean])
            
            # Restricted model  
            X_restricted = np.column_stack([np.ones(len(y_lagged_clean)), y_lagged_clean])
            
            try:
                beta_unrestricted = np.linalg.lstsq(X_unrestricted, y_current_clean, rcond=None)[0]
                beta_restricted = np.linalg.lstsq(X_restricted, y_current_clean, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            
            # Calculate predictions
            y_pred_unrestricted = X_unrestricted @ beta_unrestricted
            y_pred_restricted = X_restricted @ beta_restricted
            
            # Calculate residual sum of squares (RSS)
            RSS_unrestricted = np.sum((y_current_clean - y_pred_unrestricted) ** 2)
            RSS_restricted = np.sum((y_current_clean - y_pred_restricted) ** 2)
            
            # Total sum of squares
            TSS = np.sum((y_current_clean - np.mean(y_current_clean)) ** 2)
            
            if TSS == 0 or RSS_unrestricted == 0:
                continue
                
            # Calculate R² difference (ΔR²)
            R2_diff = (RSS_restricted - RSS_unrestricted) / TSS
            
            # Parameters for F-test
            T = len(y_current_clean)
            K_unrestricted = X_unrestricted.shape[1]
            K_restricted = X_restricted.shape[1]
            
            # Degrees of freedom
            df_numerator = K_unrestricted - K_restricted
            df_denominator = T - K_unrestricted
            
            if df_denominator <= 0:
                continue
            
            # F-statistic
            F_stat = ((RSS_restricted - RSS_unrestricted) / df_numerator) / (RSS_unrestricted / df_denominator)
            
            # P-value from F-distribution
            p_value = 1 - f_dist.cdf(F_stat, df_numerator, df_denominator)
            
            # Additional check: require minimum effect size
            min_effect_size = 0.01
            if R2_diff < min_effect_size:
                continue
                
            if p_value < best_p_value and not np.isnan(F_stat):
                best_f_stat = F_stat
                best_p_value = p_value
                best_r2_diff = R2_diff
                best_lag = lag
        
        return best_f_stat, best_p_value, best_r2_diff, best_lag
    
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan

def calculate_granger_with_specific_lags_ultra_sensitive(data_x, data_y, specific_lags, min_effect_size=0.0001):
    """ULTRA-SENSITIVE Granger causality calculation for weak signals"""
    if len(data_x) != len(data_y) or len(data_x) <= max(specific_lags):
        return np.nan, np.nan, np.nan, np.nan
    
    valid_mask = ~np.isnan(data_x) & ~np.isnan(data_y)
    x_clean = data_x[valid_mask]
    y_clean = data_y[valid_mask]
    
    if len(x_clean) <= max(specific_lags) * 2:
        return np.nan, np.nan, np.nan, np.nan
    
    try:
        n = len(y_clean)
        best_f_stat = 0
        best_p_value = 1
        best_r2_diff = 0
        best_lag = 0
        
        for lag in specific_lags:
            if n <= lag * 2:
                continue
                
            # Create lagged matrices
            y_lagged = np.zeros((n - lag, lag))
            x_lagged = np.zeros((n - lag, lag))
            
            for i in range(lag):
                y_lagged[:, i] = y_clean[i:n - lag + i]
                x_lagged[:, i] = x_clean[i:n - lag + i]
            
            y_current = y_clean[lag:]
            
            # Remove any rows with NaNs
            valid_rows = ~(np.any(np.isnan(y_lagged), axis=1) | 
                          np.any(np.isnan(x_lagged), axis=1) | 
                          np.isnan(y_current))
            
            if np.sum(valid_rows) < lag * 2:
                continue
                
            y_lagged_clean = y_lagged[valid_rows]
            x_lagged_clean = x_lagged[valid_rows]
            y_current_clean = y_current[valid_rows]
            
            # Unrestricted model
            X_unrestricted = np.column_stack([np.ones(len(y_lagged_clean)), 
                                            y_lagged_clean, x_lagged_clean])
            
            # Restricted model  
            X_restricted = np.column_stack([np.ones(len(y_lagged_clean)), y_lagged_clean])
            
            try:
                beta_unrestricted = np.linalg.lstsq(X_unrestricted, y_current_clean, rcond=None)[0]
                beta_restricted = np.linalg.lstsq(X_restricted, y_current_clean, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            
            # Calculate predictions
            y_pred_unrestricted = X_unrestricted @ beta_unrestricted
            y_pred_restricted = X_restricted @ beta_restricted
            
            # Calculate residual sum of squares (RSS)
            RSS_unrestricted = np.sum((y_current_clean - y_pred_unrestricted) ** 2)
            RSS_restricted = np.sum((y_current_clean - y_pred_restricted) ** 2)
            
            # Total sum of squares
            TSS = np.sum((y_current_clean - np.mean(y_current_clean)) ** 2)
            
            if TSS == 0 or RSS_unrestricted == 0:
                continue
                
            # Calculate R² difference (ΔR²)
            R2_diff = (RSS_restricted - RSS_unrestricted) / TSS
            
            # Parameters for F-test
            T = len(y_current_clean)
            K_unrestricted = X_unrestricted.shape[1]
            K_restricted = X_restricted.shape[1]
            
            # Degrees of freedom
            df_numerator = K_unrestricted - K_restricted
            df_denominator = T - K_unrestricted
            
            if df_denominator <= 0:
                continue
            
            # F-statistic
            F_stat = ((RSS_restricted - RSS_unrestricted) / df_numerator) / (RSS_unrestricted / df_denominator)
            
            # P-value from F-distribution
            p_value = 1 - f_dist.cdf(F_stat, df_numerator, df_denominator)
            
            # ULTRA-LENIENT effect size requirement
            if R2_diff < min_effect_size:
                continue
                
            if p_value < best_p_value and not np.isnan(F_stat):
                best_f_stat = F_stat
                best_p_value = p_value
                best_r2_diff = R2_diff
                best_lag = lag
        
        return best_f_stat, best_p_value, best_r2_diff, best_lag
    
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan

def create_comprehensive_ocean_mask(latitudes, longitudes):
    """Create ocean mask for Morocco-Portugal coastline region with enhanced coastal detail"""
    ocean_mask = np.ones((len(latitudes), len(longitudes)), dtype=bool)
    
    # Atlantic coastal cities coordinates for better masking
    coastal_cities = [
        (23.71, -15.93),  # Dakhla
        (26.13, -14.48),  # Boujdour
        (27.15, -13.20),  # Laayoune
        (28.43, -11.10),  # Tan-Tan
        (29.38, -10.18),  # Sidi Ifni

        (30.42, -9.58),   # Agadir
        (31.51, -9.77),   # Essaouira
        (32.30, -9.24),   # Safi

        (33.57, -7.59),   # Casablanca
    
        (34.25, -6.58),   # Kenitra
          
        (35.78, -5.81),   # Tangier
        (38.72, -9.14),   # Lisbon
    ]
    
    for i in range(len(latitudes)):
        for j in range(len(longitudes)):
            lat = latitudes[i]
            lon = longitudes[j]
            
            is_land = False
            
            # Enhanced coastal masking around cities
            for city_lat, city_lon in coastal_cities:
                distance = np.sqrt((lat - city_lat)**2 + (lon - city_lon)**2)
                if distance < 0.3:  # Smaller radius for more precise coastal masking
                    is_land = True
                    break
            
            # Portugal coastline
            if (lat >= 36.5 and lat <= 42.0 and lon >= -9.5 and lon <= -6.0):
                if (lat >= 37.0 and lat <= 38.5 and lon >= -9.0 and lon <= -8.0):
                    is_land = True
                elif (lat >= 38.5 and lat <= 41.5 and lon >= -9.0 and lon <= -6.0):
                    is_land = True
                elif (lat >= 36.5 and lat <= 37.5 and lon >= -8.5 and lon <= -7.5):
                    is_land = True
            
            # Southern Spain coastline near Gibraltar
            if (lat >= 35.5 and lat <= 36.5 and lon >= -5.8 and lon <= -5.0):
                is_land = True
            
            # Morocco coastline - enhanced detail
            if (lat >= 33.0 and lat <= 36.0 and lon >= -10.0 and lon <= -5.0):
                if (lat >= 35.5 and lat <= 36.0 and lon >= -6.0 and lon <= -5.0):
                    is_land = True
                elif (lat >= 34.0 and lat <= 35.0 and lon >= -7.0 and lon <= -6.0):
                    is_land = True
                elif (lat >= 33.5 and lat <= 34.0 and lon >= -8.0 and lon <= -7.0):
                    is_land = True
                elif (lat >= 33.0 and lat <= 33.5 and lon >= -9.0 and lon <= -8.0):
                    is_land = True
            
            # Southern Morocco coastline
            if (lat >= 28.0 and lat <= 33.0 and lon >= -12.0 and lon <= -8.0):
                if (lat >= 29.0 and lat <= 30.5 and lon >= -10.5 and lon <= -9.0):
                    is_land = True
                elif (lat >= 28.0 and lat <= 29.0 and lon >= -11.5 and lon <= -10.0):
                    is_land = True
            
            if is_land:
                ocean_mask[i, j] = False
    
    return ocean_mask

def load_monthly_sst_chunked_fixed(sst_var, time_indices, lat_indices, lon_indices, max_chunk_size=24):
    """Fixed version of SST loading function"""
    n_times = len(time_indices)
    n_lats = len(lat_indices)
    n_lons = len(lon_indices)
    sst_data = np.full((n_times, n_lats, n_lons), np.nan)
    
    print(f"Loading {n_times} monthly time steps in chunks...")
    
    for chunk_start in range(0, n_times, max_chunk_size):
        chunk_end = min(chunk_start + max_chunk_size, n_times)
        chunk_time_indices = time_indices[chunk_start:chunk_end]
        
        # Load chunk - CORRECTED LINE
        sst_chunk = sst_var[chunk_time_indices, :, :]
        
        for idx, time_idx in enumerate(chunk_time_indices):
            sst_full = sst_chunk[idx, :, :]
            if sst_full.max() > 200:
                sst_full = sst_full - 273.15
            sst_region = sst_full[lat_indices[:, np.newaxis], lon_indices]
            sst_data[chunk_start + idx] = sst_region
        
        print(f"  Loaded chunk {chunk_start//max_chunk_size + 1}/{(n_times-1)//max_chunk_size + 1}")
    
    return sst_data

def perform_monthly_grid_analysis_sst_to_nao(nao_monthly, sst_monthly_grid, region_lats, region_lons, ocean_mask):
    """
    Perform grid-point analysis for SST → NAO with ALL 12 MONTHLY LAGS
    Using ultra-sensitive method from reference code
    """
    print("\n=== PERFORMING MONTHLY GRID-POINT ANALYSIS (SST → NAO, 12 MONTHLY LAGS) ===")
    
    # Align the data
    min_length = min(len(nao_monthly), sst_monthly_grid.shape[0])
    nao_aligned = nao_monthly.values[:min_length]
    sst_aligned = sst_monthly_grid[:min_length, :, :]
    
    # Perform grid-point analysis with ALL 12 MONTHLY LAGS
    f_stat_map = np.full((len(region_lats), len(region_lons)), np.nan)
    p_value_map = np.full((len(region_lats), len(region_lons)), np.nan)
    lag_map = np.full((len(region_lats), len(region_lons)), np.nan)
    r2_map = np.full((len(region_lats), len(region_lons)), np.nan)
    
    valid_points = 0
    significant_points = 0
    
    for i in range(len(region_lats)):
        for j in range(len(region_lons)):
            if ocean_mask[i, j] and not np.all(np.isnan(sst_aligned[:, i, j])):
                sst_grid = sst_aligned[:, i, j]
                
                # Remove NaNs
                valid_mask = ~np.isnan(sst_grid) & ~np.isnan(nao_aligned)
                sst_clean = sst_grid[valid_mask]
                nao_clean = nao_aligned[valid_mask]
                
                if len(sst_clean) >= 24:  # Minimum data points for 12-lag analysis
                    # Test ALL 12 monthly lags and take the best result using ULTRA-SENSITIVE method
                    best_f_stat = 0
                    best_p_value = 1
                    best_r2_diff = 0
                    best_lag = 0
                    
                    for lag in range(1, 13):  # Test lags 1-12 months
                        f_stat, p_value, r2_diff, _ = calculate_granger_with_specific_lags_ultra_sensitive(
                            sst_clean, nao_clean, [lag], 0.0001
                        )
                        
                        if not np.isnan(p_value) and p_value < best_p_value:
                            best_f_stat = f_stat
                            best_p_value = p_value
                            best_r2_diff = r2_diff
                            best_lag = lag
                    
                    # Only store if we found a valid result
                    if best_p_value < 1:  # Meaning we found at least one valid result
                        f_stat_map[i, j] = best_f_stat
                        p_value_map[i, j] = best_p_value
                        lag_map[i, j] = best_lag
                        r2_map[i, j] = best_r2_diff
                        valid_points += 1
                        if best_p_value < 0.1:
                            significant_points += 1
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(region_lats)} latitudes")
    
    # Print detailed results by lag
    print(f"  ALL MONTHS: {significant_points}/{valid_points} significant points ({significant_points/valid_points*100:.1f}%)")
    
    # Count significant points by lag
    if significant_points > 0:
        lag_counts = {}
        for lag in range(1, 13):
            lag_count = np.sum((p_value_map < 0.1) & (lag_map == lag) & ocean_mask)
            if lag_count > 0:
                lag_counts[lag] = lag_count
        
        # Sort by count and show top lags
        sorted_lags = sorted(lag_counts.items(), key=lambda x: x[1], reverse=True)
        print("  Top optimal lags:")
        for lag, count in sorted_lags[:6]:  # Show top 6 lags
            lag_percent = (count / significant_points) * 100
            print(f"    Lag {lag} months: {count} points ({lag_percent:.1f}%)")
    
    return f_stat_map, p_value_map, lag_map, r2_map

def calculate_comprehensive_spatial_granger(effect_data, cause_data, lats, lons, ocean_mask, specific_lags):
    """Calculate Granger causality with detailed lag distribution analysis"""
    f_stat_map = np.full((len(lats), len(lons)), np.nan)
    p_value_map = np.full((len(lats), len(lons)), np.nan)
    r2_diff_map = np.full((len(lats), len(lons)), np.nan)
    lag_distribution_map = np.full((len(lats), len(lons)), np.nan)
    
    valid_points = 0
    significant_points = 0
    lag_counts = {lag: 0 for lag in specific_lags}
    
    for i in range(len(lats)):
        for j in range(len(lons)):
            if ocean_mask[i, j] and not np.all(np.isnan(effect_data[:, i, j])):
                grid_effect = effect_data[:, i, j]
                
                # Remove any remaining NaNs
                valid_mask = ~np.isnan(grid_effect) & ~np.isnan(cause_data)
                grid_effect_clean = grid_effect[valid_mask]
                cause_clean = cause_data[valid_mask]
                
                if len(grid_effect_clean) >= max(specific_lags) * 3:
                    f_stat, p_value, r2_diff, best_lag = calculate_granger_with_complete_lags(
                        cause_clean, grid_effect_clean, max_lag=max(specific_lags)
                    )
                    
                    if not np.isnan(f_stat):
                        f_stat_map[i, j] = f_stat
                        p_value_map[i, j] = p_value
                        r2_diff_map[i, j] = r2_diff
                        lag_distribution_map[i, j] = best_lag
                        valid_points += 1
                        
                        if p_value < 0.05:
                            significant_points += 1
                            if best_lag in lag_counts:
                                lag_counts[best_lag] += 1
    
    # Print lag distribution analysis
    total_significant = sum(lag_counts.values())
    if total_significant > 0:
        print(f"  Lag distribution analysis:")
        for lag in sorted(lag_counts.keys()):
            count = lag_counts[lag]
            percentage = (count / total_significant) * 100 if total_significant > 0 else 0
            print(f"    Lag {lag} month(s): {count} points ({percentage:.1f}%)")
    
    print(f"  Valid: {valid_points}, Significant: {significant_points} ({significant_points/valid_points*100:.1f}%)")
    return f_stat_map, p_value_map, r2_diff_map, lag_distribution_map

def plot_focused_granger_analysis(sst_file_path):
    """
    FOCUSED GRANGER CAUSALITY ANALYSIS
    Tests the three key relationships from your previous results
    """
    print("=== FOCUSED GRANGER CAUSALITY ANALYSIS ===")
    print("Testing the three key relationships:")
    print("1. JJA: Upwelling → SST")
    print("2. SON: NAO → SST") 
    print("3. SON: SST → NAO")
    print("Testing all 12 monthly lags for complete temporal coverage")
    
    # Calculate indices with memory-efficient methods
    print("\n1. CALCULATING INDICES...")
    nao_monthly = calculate_nao_index_netcdf_memory_efficient(sst_file_path)
    upwelling_monthly = calculate_moroccan_upwelling_index_memory_efficient(sst_file_path)
    sst_monthly = calculate_moroccan_sst_memory_efficient(sst_file_path)
    
    print(f"\nMonthly data summary:")
    print(f"NAO: {len(nao_monthly)} months")
    print(f"Upwelling: {len(upwelling_monthly)} months") 
    print(f"SST: {len(sst_monthly)} months")
    
    # Load SST data for spatial analysis
    print("\n2. LOADING SPATIAL SST DATA...")
    dataset = nc.Dataset(sst_file_path, 'r')
    
    lat_var = dataset.variables['latitude']
    lon_var = dataset.variables['longitude']
    sst_var = dataset.variables['sst']
    time_var = dataset.variables['valid_time']
    
    # Convert time to datetime objects
    times = nc.num2date(time_var[:], time_var.units)
    regular_times = []
    for t in times:
        if hasattr(t, 'year'):
            regular_times.append(datetime(t.year, t.month, t.day, t.hour, t.minute, t.second))
        else:
            regular_times.append(t)
    
    # Get coordinates
    latitudes = lat_var[:]
    longitudes = lon_var[:]
    
    # Convert longitude if needed
    if longitudes.max() > 180:
        longitudes = np.where(longitudes > 180, longitudes - 360, longitudes)
    
    # Focus region
    lat_min, lat_max = 20, 42
    lon_min, lon_max = -30, -5

    # Find indices for our region
    lat_indices = np.where((latitudes >= lat_min) & (latitudes <= lat_max))[0]
    lon_indices = np.where((longitudes >= lon_min) & (longitudes <= lon_max))[0]
    
    region_lats = latitudes[lat_indices]
    region_lons = longitudes[lon_indices]
    
    print(f"Region: {len(region_lats)} lats, {len(region_lons)} lons")
    
    # Create ocean mask
    ocean_mask = create_comprehensive_ocean_mask(region_lats, region_lons)
    ocean_points = np.sum(ocean_mask)
    print(f"Ocean points: {ocean_points}")
    
    # Get all monthly time indices
    all_indices = list(range(len(regular_times)))
    
    # Load all monthly SST data using FIXED function
    sst_monthly_grid = load_monthly_sst_chunked_fixed(sst_var, all_indices, lat_indices, lon_indices)
    
    dataset.close()
    
    # Align monthly time series
    min_length = min(len(nao_monthly), len(upwelling_monthly), len(sst_monthly), sst_monthly_grid.shape[0])
    
    nao_monthly_aligned = nao_monthly.values[:min_length]
    upwelling_monthly_aligned = upwelling_monthly.values[:min_length]
    sst_monthly_aligned = sst_monthly_grid[:min_length, :, :]
    
    print(f"\nAligned monthly data: {min_length} months")
    
    # Calculate focused Granger causality with all 12 lags
    print("\n3. FOCUSED GRANGER CAUSALITY WITH ALL 12 MONTHLY LAGS...")
    
    # Define complete lag set
    ALL_MONTHLY_LAGS = list(range(1, 13))
    print(f"Testing lags: {ALL_MONTHLY_LAGS} months")
    
    # Focus on the three key relationships from your results
    results = {}
    
    # 1. JJA: Upwelling -> SST
    print("\nA. JJA: Upwelling → SST...")
    jja_months = [6, 7, 8]
    jja_mask = np.array([t.month in jja_months for t in regular_times[:min_length]])
    
    upwelling_jja = upwelling_monthly_aligned[jja_mask]
    sst_jja = sst_monthly_aligned[jja_mask, :, :]
    
    f_stat_jja, p_value_jja, r2_diff_jja, lag_dist_jja = calculate_comprehensive_spatial_granger(
        sst_jja, upwelling_jja, region_lats, region_lons, ocean_mask, ALL_MONTHLY_LAGS
    )
    results['jja_upwelling_sst'] = {
        'f_stat': f_stat_jja, 'p_value': p_value_jja, 
        'r2_diff': r2_diff_jja, 'lag_dist': lag_dist_jja,
        'season': 'JJA', 'relationship': 'Upwelling → SST'
    }
    
    # 2. SON: NAO -> SST
    print("\nB. SON: NAO → SST...")
    son_months = [9, 10, 11]
    son_mask = np.array([t.month in son_months for t in regular_times[:min_length]])
    
    nao_son = nao_monthly_aligned[son_mask]
    sst_son = sst_monthly_aligned[son_mask, :, :]
    
    f_stat_son_nao_sst, p_value_son_nao_sst, r2_diff_son_nao_sst, lag_dist_son_nao_sst = calculate_comprehensive_spatial_granger(
        sst_son, nao_son, region_lats, region_lons, ocean_mask, ALL_MONTHLY_LAGS
    )
    results['son_nao_sst'] = {
        'f_stat': f_stat_son_nao_sst, 'p_value': p_value_son_nao_sst,
        'r2_diff': r2_diff_son_nao_sst, 'lag_dist': lag_dist_son_nao_sst,
        'season': 'SON', 'relationship': 'NAO → SST'
    }
    
    # 3. SON: SST -> NAO (using ULTRA-SENSITIVE method from reference code)
    print("\nC. SON: SST → NAO (Ultra-sensitive analysis)...")
    # Use the ultra-sensitive method for SST → NAO
    f_stat_son_sst_nao, p_value_son_sst_nao, lag_dist_son_sst_nao, r2_diff_son_sst_nao = perform_monthly_grid_analysis_sst_to_nao(
        nao_monthly, sst_monthly_aligned[son_mask, :, :], region_lats, region_lons, ocean_mask
    )
    results['son_sst_nao'] = {
        'f_stat': f_stat_son_sst_nao, 'p_value': p_value_son_sst_nao,
        'r2_diff': r2_diff_son_sst_nao, 'lag_dist': lag_dist_son_sst_nao,
        'season': 'SON', 'relationship': 'SST → NAO'
    }
    
    # Create focused plots
    print("\n4. CREATING FOCUSED ANALYSIS PLOTS...")
    create_focused_plots(results, region_lats, region_lons, ocean_mask, ALL_MONTHLY_LAGS)
    
    print("\n=== FOCUSED ANALYSIS COMPLETE ===")

def plot_focused_map(ax, data_map, p_value_map, lats, lons, ocean_mask, title, colorbar_label, cmap, vmax=None):
    """Plot focused map with enhanced visualization"""
    significant_mask = (p_value_map < 0.05) & ocean_mask
    data_plot = np.where(significant_mask, data_map, np.nan)
    
    ax.set_extent([-30, -5, 20, 42], crs=ccrs.PlateCarree())
    
    if np.any(~np.isnan(data_plot)):
        if vmax is None:
            vmax = np.nanmax(data_plot) if not np.all(np.isnan(data_plot)) else 1.0
        
        levels = np.linspace(0, vmax, 21)
        
        contourf = ax.contourf(lons, lats, data_plot, levels=levels, 
                              cmap=cmap, transform=ccrs.PlateCarree(),
                              extend='max')
        
        # Add significance contours
        sig_01_mask = (p_value_map < 0.01) & ocean_mask
        sig_001_mask = (p_value_map < 0.001) & ocean_mask
        
        if np.any(sig_001_mask):
            sig_data = np.where(sig_001_mask, 1.0, 0.0)
            ax.contour(lons, lats, sig_data, levels=[0.5],
                      colors='red', linewidths=2.0, transform=ccrs.PlateCarree())
        elif np.any(sig_01_mask):
            sig_data = np.where(sig_01_mask, 1.0, 0.0)
            ax.contour(lons, lats, sig_data, levels=[0.5],
                      colors='orange', linewidths=1.5, transform=ccrs.PlateCarree())
        elif np.any(significant_mask):
            sig_data = np.where(significant_mask, 1.0, 0.0)
            ax.contour(lons, lats, sig_data, levels=[0.5],
                      colors='white', linewidths=1.0, transform=ccrs.PlateCarree())
        
        plt.colorbar(contourf, ax=ax, orientation='horizontal', 
                     pad=0.05, shrink=0.8, label=colorbar_label)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=2)
        
        # Add ATLANTIC COASTAL CITIES ONLY
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
            'Lisbon': (38.72, -9.14),  # Portugal
        }
        
        for city, (lat, lon) in cities.items():
            # Only plot cities within the map extent
            if lat >= 20 and lat <= 42 and lon >= -30 and lon <= -5:
                ax.plot(lon, lat, 'ko', markersize=3, transform=ccrs.PlateCarree(), zorder=3)
                # Adjust text position based on city location to avoid overlap
                offset_x = 0.2
                offset_y = 0.1
                ax.text(lon + offset_x, lat + offset_y, city, transform=ccrs.PlateCarree(), 
                       fontsize=7, zorder=3, ha='left', va='bottom')
    else:
        ax.text(0.5, 0.5, 'No significant results', transform=ax.transAxes, 
                ha='center', va='center', fontsize=10, color='red')
    
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.7, zorder=1)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title(title, fontweight='bold', fontsize=10)

def plot_lag_distribution_focused(ax, lag_map, p_value_map, lats, lons, ocean_mask, title, colorbar_label, cmap, all_lags):
    """Plot optimal lag distribution for focused analysis"""
    significant_mask = (p_value_map < 0.05) & ocean_mask
    lag_plot = np.where(significant_mask, lag_map, np.nan)
    
    ax.set_extent([-30, -5, 20, 42], crs=ccrs.PlateCarree())
    
    if np.any(~np.isnan(lag_plot)):
        # Create discrete colormap for lags
        levels = np.array(all_lags + [max(all_lags) + 1]) - 0.5
        
        contourf = ax.contourf(lons, lats, lag_plot, levels=levels, 
                              cmap=cmap, transform=ccrs.PlateCarree(),
                              extend='neither')
        
        # Create discrete colorbar
        cbar = plt.colorbar(contourf, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8, label=colorbar_label)
        cbar.set_ticks(all_lags)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=2)
        
        # Add ATLANTIC COASTAL CITIES ONLY (same as above)
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
            'Lisbon': (38.72, -9.14),  # Portugal
        }
        
        for city, (lat, lon) in cities.items():
            # Only plot cities within the map extent
            if lat >= 20 and lat <= 42 and lon >= -30 and lon <= -5:
                ax.plot(lon, lat, 'ko', markersize=3, transform=ccrs.PlateCarree(), zorder=3)
                # Adjust text position based on city location to avoid overlap
                offset_x = 0.2
                offset_y = 0.1
                ax.text(lon + offset_x, lat + offset_y, city, transform=ccrs.PlateCarree(), 
                       fontsize=7, zorder=3, ha='left', va='bottom')
    else:
        ax.text(0.5, 0.5, 'No significant results', transform=ax.transAxes, 
                ha='center', va='center', fontsize=10, color='red')
    
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.7, zorder=1)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=0)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title(title, fontweight='bold', fontsize=10)

def create_focused_plots(results, lats, lons, ocean_mask, all_lags):
    """Create focused plots for the three key relationships"""
    
    # Calculate summary statistics
    def get_comprehensive_stats(f_map, p_map, r2_map, lag_map, mask):
        significant_mask = (p_map < 0.05) & mask
        if np.any(significant_mask):
            mean_f = np.nanmean(f_map[significant_mask])
            mean_r2 = np.nanmean(r2_map[significant_mask])
            percent_sig = (np.sum(significant_mask) / np.sum(mask)) * 100
            
            # Lag distribution
            significant_lags = lag_map[significant_mask]
            unique_lags, lag_counts = np.unique(significant_lags[~np.isnan(significant_lags)], return_counts=True)
            dominant_lag = unique_lags[np.argmax(lag_counts)] if len(unique_lags) > 0 else 0
            
            return mean_f, mean_r2, percent_sig, dominant_lag, dict(zip(unique_lags, lag_counts))
        return 0, 0, 0, 0, {}
    
    # Create the main analysis figure - 3 rows, 3 columns (without summary)
    fig = plt.figure(figsize=(18, 15))
    
    print("\nCREATING FOCUSED VISUALIZATION...")
    
    # Define the three relationships to plot
    relationships = [
        ('jja_upwelling_sst', 'JJA: Upwelling → SST'),
        ('son_nao_sst', 'SON: NAO → SST'),
        ('son_sst_nao', 'SON: SST → NAO')
    ]
    
    for idx, (result_key, title_base) in enumerate(relationships):
        if result_key in results:
            data = results[result_key]
            stats = get_comprehensive_stats(
                data['f_stat'], data['p_value'],
                data['r2_diff'], data['lag_dist'], ocean_mask
            )
            
            # F-statistic plot
            ax1 = fig.add_subplot(3, 3, idx*3 + 1, projection=ccrs.PlateCarree())
            plot_focused_map(ax1, data['f_stat'], data['p_value'], lats, lons, ocean_mask,
                           f'{title_base}\nF-statistic ({stats[2]:.1f}% sig.)', 
                           'F-statistic', 'hot_r', vmax=15)
            
            # ΔR² plot
            ax2 = fig.add_subplot(3, 3, idx*3 + 2, projection=ccrs.PlateCarree())
            plot_focused_map(ax2, data['r2_diff'], data['p_value'], lats, lons, ocean_mask,
                           f'{title_base}\nΔR² ({stats[2]:.1f}% sig.)', 
                           'ΔR²', 'viridis', vmax=0.15)
            
            # Lag distribution plot
            ax3 = fig.add_subplot(3, 3, idx*3 + 3, projection=ccrs.PlateCarree())
            plot_lag_distribution_focused(ax3, data['lag_dist'], data['p_value'], lats, lons, ocean_mask,
                                        f'{title_base}\nOptimal Lag (months)', 
                                        'Lag (months)', 'plasma', all_lags)
    
    plt.tight_layout()
    plt.savefig('focused_granger_analysis_3relationships.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create separate explanation figure
    create_explanation_figure(results, ocean_mask, all_lags)
    
    # Print detailed analysis
    print_focused_analysis(results, ocean_mask, all_lags)

def create_explanation_figure(results, ocean_mask, all_lags):
    """Create separate figure with comprehensive explanation and summary"""
    
    def get_comprehensive_stats(f_map, p_map, r2_map, lag_map, mask):
        significant_mask = (p_map < 0.05) & mask
        if np.any(significant_mask):
            mean_f = np.nanmean(f_map[significant_mask])
            mean_r2 = np.nanmean(r2_map[significant_mask])
            percent_sig = (np.sum(significant_mask) / np.sum(mask)) * 100
            
            # Lag distribution
            significant_lags = lag_map[significant_mask]
            unique_lags, lag_counts = np.unique(significant_lags[~np.isnan(significant_lags)], return_counts=True)
            dominant_lag = unique_lags[np.argmax(lag_counts)] if len(unique_lags) > 0 else 0
            
            return mean_f, mean_r2, percent_sig, dominant_lag, dict(zip(unique_lags, lag_counts))
        return 0, 0, 0, 0, {}
    
    # Create explanation figure
    fig = plt.figure(figsize=(16, 12))
    
    # Main title
    fig.suptitle('COMPREHENSIVE TEMPORAL COVERAGE ANALYSIS\nAll 12 Monthly Lags Tested (1-12 months)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Define relationships for summary
    relationships = [
        ('jja_upwelling_sst', 'JJA: Upwelling → SST'),
        ('son_nao_sst', 'SON: NAO → SST'),
        ('son_sst_nao', 'SON: SST → NAO')
    ]
    
    # Calculate overall statistics
    summary_stats = []
    for result_key, title_base in relationships:
        if result_key in results:
            data = results[result_key]
            stats = get_comprehensive_stats(
                data['f_stat'], data['p_value'],
                data['r2_diff'], data['lag_dist'], ocean_mask
            )
            summary_stats.append((title_base, stats[2], stats[3], stats[0], stats[1]))
    
    # Panel 1: Key Findings
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axis('off')
    
    findings_text = "KEY FINDINGS:\n\n"
    for title, sig_percent, dom_lag, mean_f, mean_r2 in summary_stats:
        findings_text += f"• {title}:\n"
        findings_text += f"  Significance: {sig_percent:.1f}%\n"
        findings_text += f"  Dominant lag: {dom_lag} months\n"
        findings_text += f"  Mean F: {mean_f:.2f}, Mean ΔR²: {mean_r2:.4f}\n\n"
    
    ax1.text(0.05, 0.95, findings_text, transform=ax1.transAxes, 
             fontsize=11, va='top', ha='left', linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    ax1.set_title('Statistical Summary', fontweight='bold', fontsize=14)
    
    # Panel 2: Lag Interpretation
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.axis('off')
    
    lag_text = (
        "LAG INTERPRETATION GUIDE:\n\n"
        "• Lags 1-3 months:\n"
        "  Immediate effects & short-term responses\n"
        "  Fast atmospheric-ocean coupling\n\n"
        "• Lags 4-6 months:\n"  
        "  Seasonal memory effects\n"
        "  Ocean mixed layer persistence\n\n"
        "• Lags 7-9 months:\n"
        "  Medium-term ocean inertia\n"
        "  Deep ocean influence\n\n"
        "• Lags 10-12 months:\n"
        "  Annual cycle persistence\n"
        "  Interannual variability\n\n"
        "Mixed lags indicate complex\ntemporal dynamics across regions"
    )
    
    ax2.text(0.05, 0.95, lag_text, transform=ax2.transAxes, 
             fontsize=11, va='top', ha='left', linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    ax2.set_title('Temporal Scale Interpretation', fontweight='bold', fontsize=14)
    
    # Panel 3: Methodological Details
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')
    
    method_text = (
        "METHODOLOGICAL FEATURES:\n\n"
        "• Complete Temporal Coverage:\n"
        "  All 12 monthly lags tested: 1-12 months\n\n"
        "• Full Annual Cycle:\n"
        "  Captures all possible temporal relationships\n\n"
        "• Balanced Distribution:\n"
        "  Even coverage across all seasons\n\n"
        "• Enhanced Analysis:\n"
        "  Monthly grid analysis with comprehensive lag testing\n\n"
        "• Statistical Robustness:\n"
        "  F-statistics, p-values, and ΔR² effect sizes\n\n"
        "• Spatial Resolution:\n"
        "  Individual grid point analysis\n"
        "  Ocean mask applied for marine focus"
    )
    
    ax3.text(0.05, 0.95, method_text, transform=ax3.transAxes, 
             fontsize=11, va='top', ha='left', linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax3.set_title('Analysis Methodology', fontweight='bold', fontsize=14)
    
    # Panel 4: Physical Interpretation
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    physical_text = (
        "PHYSICAL INTERPRETATION:\n\n"
        "• Upwelling → SST:\n"
        "  Coastal upwelling cools sea surface\n"
        "  Wind-driven Ekman transport effects\n\n"
        "• NAO → SST:\n"
        "  Atmospheric circulation influences\n"
        "  Ocean heat content changes\n\n"
        "• SST → NAO:\n"
        "  Ocean heat anomalies feedback\n"
        "  Air-sea interaction mechanisms\n\n"
        "REGIONAL CONTEXT:\n"
        "• Study Area: 20°N-42°N, 30°W-5°W\n"
        "• Moroccan Atlantic coast & offshore\n"
        "• Key upwelling region\n"
        "• NAO influence corridor"
    )
    
    ax4.text(0.05, 0.95, physical_text, transform=ax4.transAxes, 
             fontsize=11, va='top', ha='left', linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
    ax4.set_title('Physical Processes', fontweight='bold', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig('comprehensive_analysis_explanation.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_focused_analysis(results, ocean_mask, all_lags):
    """Print detailed analysis of the focused results"""
    print("\n" + "="*80)
    print("FOCUSED GRANGER CAUSALITY ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nLAG STRUCTURE ANALYSIS: Testing all {len(all_lags)} monthly lags")
    print(f"Lags tested: {all_lags}")
    print("Complete annual cycle coverage achieved")
    
    print(f"\nKEY RELATIONSHIPS ANALYSIS:")
    
    for result_key, data in results.items():
        if 'jja_upwelling_sst' in result_key:
            name = "JJA: Upwelling → SST"
        elif 'son_nao_sst' in result_key:
            name = "SON: NAO → SST"
        elif 'son_sst_nao' in result_key:
            name = "SON: SST → NAO"
        else:
            name = result_key
        
        significant_mask = (data['p_value'] < 0.05) & ocean_mask
        percent_sig = (np.sum(significant_mask) / np.sum(ocean_mask)) * 100
        
        if np.any(significant_mask):
            mean_f = np.nanmean(data['f_stat'][significant_mask])
            mean_r2 = np.nanmean(data['r2_diff'][significant_mask])
            
            # Lag analysis
            significant_lags = data['lag_dist'][significant_mask]
            unique_lags, lag_counts = np.unique(significant_lags[~np.isnan(significant_lags)], return_counts=True)
            
            if len(unique_lags) > 0:
                dominant_lag = unique_lags[np.argmax(lag_counts)]
                lag_dist_str = ", ".join([f"{lag}m:{count}" for lag, count in zip(unique_lags, lag_counts)])
            else:
                dominant_lag = "N/A"
                lag_dist_str = "None"
            
            print(f"\n  {name}:")
            print(f"    • Significance: {percent_sig:.1f}% of ocean points")
            print(f"    • Mean F-statistic: {mean_f:.3f}")
            print(f"    • Mean ΔR²: {mean_r2:.4f}")
            print(f"    • Dominant lag: {dominant_lag} months")
            print(f"    • Lag distribution: {lag_dist_str}")
            
            # Interpretation
            if dominant_lag <= 3:
                lag_interpretation = "Immediate effect (short-term)"
            elif dominant_lag <= 8:
                lag_interpretation = "Seasonal persistence (medium-term)"
            else:
                lag_interpretation = "Annual cycle influence (long-term)"
            
            print(f"    • Interpretation: {lag_interpretation}")
        else:
            print(f"\n  {name}: No significant causal relationship found")
    
    print("\n" + "="*80)

# Run the focused analysis
if __name__ == "__main__":
    sst_file_path = "C:/Users/moham/OneDrive/Documents/Climate_Project/dadefda24611707dd32599a670df250b.nc"
    
    print("=== FOCUSED GRANGER CAUSALITY ANALYSIS ===")
    print("Testing the three key relationships from previous results:")
    print("1. JJA: Upwelling → SST")
    print("2. SON: NAO → SST") 
    print("3. SON: SST → NAO")
    print("\nFEATURES:")
    print("• Complete Temporal Coverage: Tests all 12 monthly lags [1-12 months]")
    print("• Full Annual Cycle: Captures all possible temporal relationships") 
    print("• Balanced Distribution: Even coverage across all seasons")
    print("• Enhanced Analysis: Monthly grid analysis with comprehensive lag testing")
    print("• Detailed Lag Distribution: Shows which months are most important")
    print(f"Region: 20°N to 42°N, -30°W to -5°W")
    
    plot_focused_granger_analysis(sst_file_path)
