"""
Starlink Telemetry Augmentation Module

This module generates synthetic telemetry data based on statistical patterns
from the Starlink Telemetry and Performance Dataset (opendatabay.com).
It augments real-time satellite position data with performance metrics.
"""

import numpy as np
from typing import Dict, List, Tuple
import random


class StarlinkTelemetryGenerator:
    """
    Generates synthetic Starlink telemetry data based on realistic statistical patterns.
    
    Dataset Schema (13 columns):
    - S_ID: Unique identifier
    - latitude, longitude: Geographic coordinates
    - elevation_m: Elevation above sea level
    - season: Spring, Summer, Fall, Winter
    - weather: rain, clear, heavy rain, etc.
    - visible_satellites: Number of visible satellites
    - serving_satellites: Number of serving satellites
    - signal_loss_db: Signal loss in dB
    - download_throughput_bps: Download speed in bps
    - upload_throughput_bps: Upload speed in bps
    - packet_loss: Percentage of packet loss
    - qoe_score: Quality of Experience score
    """
    
    # Statistical patterns derived from typical Starlink performance
    WEATHER_CONDITIONS = ['clear', 'partly_cloudy', 'cloudy', 'rain', 'heavy_rain', 'snow']
    WEATHER_WEIGHTS = [0.40, 0.25, 0.15, 0.12, 0.05, 0.03]  # Probability distribution
    
    SEASONS = ['Spring', 'Summer', 'Fall', 'Winter']
    
    # Performance baselines (mean, std_dev)
    SIGNAL_LOSS_STATS = {
        'clear': (2.5, 0.8),
        'partly_cloudy': (3.2, 1.0),
        'cloudy': (4.5, 1.5),
        'rain': (6.8, 2.0),
        'heavy_rain': (12.5, 3.5),
        'snow': (8.5, 2.5)
    }
    
    DOWNLOAD_THROUGHPUT_STATS = {
        'clear': (150e6, 30e6),        # 150 Mbps ± 30 Mbps
        'partly_cloudy': (130e6, 35e6),
        'cloudy': (100e6, 40e6),
        'rain': (70e6, 30e6),
        'heavy_rain': (30e6, 20e6),
        'snow': (50e6, 25e6)
    }
    
    UPLOAD_THROUGHPUT_STATS = {
        'clear': (20e6, 5e6),          # 20 Mbps ± 5 Mbps
        'partly_cloudy': (18e6, 5e6),
        'cloudy': (15e6, 6e6),
        'rain': (10e6, 5e6),
        'heavy_rain': (5e6, 3e6),
        'snow': (8e6, 4e6)
    }
    
    PACKET_LOSS_STATS = {
        'clear': (0.1, 0.05),          # 0.1% ± 0.05%
        'partly_cloudy': (0.2, 0.1),
        'cloudy': (0.5, 0.2),
        'rain': (1.5, 0.8),
        'heavy_rain': (5.0, 2.0),
        'snow': (2.5, 1.2)
    }
    
    def __init__(self, seed: int = None):
        """Initialize the telemetry generator with optional random seed."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def get_current_season(self) -> str:
        """Get current season (simplified - could use actual date)."""
        return random.choice(self.SEASONS)
    
    def generate_weather_condition(self, latitude: float) -> str:
        """
        Generate weather condition based on latitude.
        Higher latitudes have slightly higher chance of adverse weather.
        """
        # Adjust weights based on latitude (polar regions more adverse weather)
        lat_factor = abs(latitude) / 90.0
        adjusted_weights = [
            w * (1.2 - 0.4 * lat_factor) if i < 2 else w * (0.8 + 0.4 * lat_factor)
            for i, w in enumerate(self.WEATHER_WEIGHTS)
        ]
        # Normalize
        total = sum(adjusted_weights)
        adjusted_weights = [w / total for w in adjusted_weights]
        
        return random.choices(self.WEATHER_CONDITIONS, weights=adjusted_weights)[0]
    
    def calculate_visible_satellites(self, altitude: float, latitude: float) -> int:
        """
        Calculate number of visible satellites based on altitude and latitude.
        Higher altitude = more satellites visible.
        """
        # Base visibility increases with altitude
        base_visible = int(8 + (altitude / 100) * 2)
        
        # Add some randomness
        visible = max(4, int(np.random.normal(base_visible, 2)))
        
        return min(visible, 20)  # Cap at 20
    
    def calculate_serving_satellites(self, visible_satellites: int) -> int:
        """
        Calculate number of serving satellites.
        Typically 1-3 satellites actively serve a connection.
        """
        max_serving = min(visible_satellites, 4)
        return random.randint(1, max_serving)
    
    def generate_signal_loss(self, weather: str, altitude: float) -> float:
        """
        Generate signal loss in dB based on weather and altitude.
        Higher altitude = slightly better signal.
        """
        mean, std = self.SIGNAL_LOSS_STATS[weather]
        
        # Altitude bonus (higher altitude = less atmospheric interference)
        altitude_factor = max(0.7, 1.0 - (altitude / 2000) * 0.3)
        
        signal_loss = np.random.normal(mean * altitude_factor, std)
        return max(0.5, signal_loss)  # Minimum 0.5 dB loss
    
    def generate_throughput(self, weather: str, signal_loss: float, 
                           direction: str = 'download') -> float:
        """
        Generate throughput (download or upload) based on weather and signal loss.
        """
        stats = (self.DOWNLOAD_THROUGHPUT_STATS if direction == 'download' 
                else self.UPLOAD_THROUGHPUT_STATS)
        
        mean, std = stats[weather]
        
        # Signal loss degrades throughput
        signal_factor = max(0.3, 1.0 - (signal_loss / 20.0))
        
        throughput = np.random.normal(mean * signal_factor, std)
        return max(1e6, throughput)  # Minimum 1 Mbps
    
    def generate_packet_loss(self, weather: str, signal_loss: float) -> float:
        """Generate packet loss percentage based on weather and signal loss."""
        mean, std = self.PACKET_LOSS_STATS[weather]
        
        # Signal loss increases packet loss
        signal_factor = 1.0 + (signal_loss / 15.0)
        
        packet_loss = np.random.normal(mean * signal_factor, std)
        return max(0.0, min(100.0, packet_loss))  # Clamp between 0-100%
    
    def calculate_qoe_score(self, download_throughput: float, packet_loss: float,
                           signal_loss: float) -> float:
        """
        Calculate Quality of Experience (QoE) score (0-10 scale).
        Based on throughput, packet loss, and signal quality.
        """
        # Throughput component (0-4 points)
        throughput_score = min(4.0, (download_throughput / 1e6) / 37.5)
        
        # Packet loss component (0-3 points, inverted)
        packet_loss_score = max(0, 3.0 - (packet_loss / 2.0))
        
        # Signal quality component (0-3 points, inverted)
        signal_score = max(0, 3.0 - (signal_loss / 5.0))
        
        qoe = throughput_score + packet_loss_score + signal_score
        return round(min(10.0, max(0.0, qoe)), 2)
    
    def generate_telemetry(self, satellite_id: int, latitude: float, 
                          longitude: float, altitude: float) -> Dict:
        """
        Generate complete telemetry data for a satellite observation.
        
        Args:
            satellite_id: Unique satellite identifier
            latitude: Geographic latitude (-90 to 90)
            longitude: Geographic longitude (-180 to 180)
            altitude: Satellite altitude in km
        
        Returns:
            Dictionary containing all telemetry metrics
        """
        # Generate environmental conditions
        season = self.get_current_season()
        weather = self.generate_weather_condition(latitude)
        
        # Calculate satellite visibility
        visible_satellites = self.calculate_visible_satellites(altitude, latitude)
        serving_satellites = self.calculate_serving_satellites(visible_satellites)
        
        # Generate performance metrics
        signal_loss = self.generate_signal_loss(weather, altitude)
        download_throughput = self.generate_throughput(weather, signal_loss, 'download')
        upload_throughput = self.generate_throughput(weather, signal_loss, 'upload')
        packet_loss = self.generate_packet_loss(weather, signal_loss)
        
        # Calculate QoE score
        qoe_score = self.calculate_qoe_score(download_throughput, packet_loss, signal_loss)
        
        return {
            'S_ID': satellite_id,
            'latitude': round(latitude, 6),
            'longitude': round(longitude, 6),
            'altitude_km': round(altitude, 2),
            'season': season,
            'weather': weather,
            'visible_satellites': visible_satellites,
            'serving_satellites': serving_satellites,
            'signal_loss_db': round(signal_loss, 2),
            'download_throughput_mbps': round(download_throughput / 1e6, 2),
            'upload_throughput_mbps': round(upload_throughput / 1e6, 2),
            'packet_loss_percent': round(packet_loss, 3),
            'qoe_score': qoe_score
        }
    
    def augment_satellite_positions(self, positions: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Augment satellite position data with telemetry metrics.
        
        Args:
            positions: Dictionary mapping satellite_id to {'lat', 'lng', 'alt'}
        
        Returns:
            Dictionary with augmented telemetry data
        """
        augmented_data = {}
        
        for sat_id, pos in positions.items():
            telemetry = self.generate_telemetry(
                satellite_id=sat_id,
                latitude=pos['lat'],
                longitude=pos['lng'],
                altitude=pos['alt']
            )
            
            # Merge position and telemetry data
            augmented_data[sat_id] = {**pos, **telemetry}
        
        return augmented_data
    
    def calculate_performance_score(self, telemetry: Dict) -> float:
        """
        Calculate overall performance score (0-1) for optimization.
        Higher score = better performance.
        """
        # Normalize QoE score (0-10 -> 0-1)
        qoe_normalized = telemetry['qoe_score'] / 10.0
        
        # Normalize throughput (assume 150 Mbps is optimal)
        throughput_normalized = min(1.0, telemetry['download_throughput_mbps'] / 150.0)
        
        # Invert packet loss (lower is better)
        packet_loss_normalized = max(0.0, 1.0 - (telemetry['packet_loss_percent'] / 10.0))
        
        # Weighted average
        performance_score = (
            0.4 * qoe_normalized +
            0.3 * throughput_normalized +
            0.3 * packet_loss_normalized
        )
        
        return round(performance_score, 3)


def print_telemetry_summary(augmented_data: Dict[int, Dict]):
    """Print a summary of telemetry data for all satellites."""
    print("\n" + "="*80)
    print("STARLINK TELEMETRY SUMMARY")
    print("="*80)
    
    for sat_id, data in augmented_data.items():
        print(f"\nSatellite ID: {sat_id}")
        print(f"  Position: ({data['latitude']:.2f}°, {data['longitude']:.2f}°) @ {data['altitude_km']:.0f} km")
        print(f"  Weather: {data['weather']} | Season: {data['season']}")
        print(f"  Satellites: {data['serving_satellites']}/{data['visible_satellites']} serving/visible")
        print(f"  Signal Loss: {data['signal_loss_db']:.2f} dB")
        print(f"  Throughput: ↓{data['download_throughput_mbps']:.1f} Mbps / ↑{data['upload_throughput_mbps']:.1f} Mbps")
        print(f"  Packet Loss: {data['packet_loss_percent']:.2f}%")
        print(f"  QoE Score: {data['qoe_score']}/10")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Example usage
    generator = StarlinkTelemetryGenerator(seed=42)
    
    # Simulate some satellite positions
    example_positions = {
        25544: {'lat': 45.5, 'lng': -122.6, 'alt': 420},  # ISS-like orbit
        20580: {'lat': -23.4, 'lng': 151.2, 'alt': 547},  # Hubble-like orbit
        24876: {'lat': 65.2, 'lng': -18.1, 'alt': 850},   # Higher orbit
    }
    
    # Augment with telemetry
    augmented = generator.augment_satellite_positions(example_positions)
    
    # Print summary
    print_telemetry_summary(augmented)
    
    # Calculate performance scores
    print("\nPERFORMANCE SCORES:")
    for sat_id, data in augmented.items():
        score = generator.calculate_performance_score(data)
        print(f"  Satellite {sat_id}: {score:.3f}")
