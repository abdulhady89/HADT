import numpy as np
import random

def generate_LEO_targets(inclination_deg=98.0, altitude_km=500, num_targets=50,
                         phase_offset_deg=0.0, lon_offset_deg=0.0, lat_offset_deg=0.0):
    """
    Generate target points directly under a LEO satellite's ground track.
    
    Args:
        inclination_deg (float): Orbital inclination in degrees.
        altitude_km (float): Satellite altitude above Earth's surface (km).
        num_targets (int): Number of ground targets to generate.
        phase_offset_deg (float): Orbit phase offset (degrees).
        lon_offset_deg (float): Optional random longitude offset range (degrees).
        lat_offset_deg (float): Optional random latitude offset range (degrees).
    
    Returns:
        list of (lat, lon, elev): Target coordinates under the satellite ground track.
    """
    EARTH_RADIUS = 6378.1  # km
    MU = 398600.4418  # Earth's gravitational parameter, km^3/s^2
    
    # Orbital parameters
    semi_major_axis = EARTH_RADIUS + altitude_km
    period = 2 * np.pi * np.sqrt(semi_major_axis**3 / MU)
    
    # Time samples along the orbit
    num_points = num_targets * 2  # sample denser and pick subset
    time_steps = np.linspace(0, period, num_points)
    
    # Earth's rotation rate (deg/s)
    earth_rot_rate = 360 / 86164.0
    
    # Arrays to store ground track
    latitudes, longitudes = [], []
    
    for t in time_steps:
        mean_motion = 2 * np.pi / period
        true_anomaly = (mean_motion * t + np.radians(phase_offset_deg)) % (2 * np.pi)
        
        # Ground track latitude and longitude
        lat = np.degrees(np.arcsin(np.sin(np.radians(inclination_deg)) * np.sin(true_anomaly)))
        lon = (np.degrees(true_anomaly) - earth_rot_rate * t) % 360
        lon = lon - 360 if lon > 180 else lon
        
        latitudes.append(lat)
        longitudes.append(lon)
    
    # Select evenly spaced targets
    step = max(1, len(latitudes) // num_targets)
    targets = []
    for i in range(0, len(latitudes), step):
        lat = latitudes[i] + random.uniform(-lat_offset_deg, lat_offset_deg)
        lon = longitudes[i] + random.uniform(-lon_offset_deg, lon_offset_deg)
        targets.append((lat, lon))
        if len(targets) >= num_targets:
            break
    
    return targets

targets = generate_LEO_targets(
    inclination_deg=98.0,
    altitude_km=500,
    num_targets=50,
    phase_offset_deg=30.0,   # if you have multiple satellites
    lon_offset_deg=0.1,      # optional lateral spread
    lat_offset_deg=0.05
)

# Print first few targets
for t in targets[:5]:
    print(f"Lat: {t[0]:.3f}, Lon: {t[1]:.3f}")