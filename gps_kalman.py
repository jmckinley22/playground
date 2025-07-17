"""GPS-Based Vehicle Tracking with IMU Fusion using a Kalman Filter.

This script generates synthetic GPS and IMU data for a vehicle traveling
in a circular path.  It demonstrates a simple Kalman Filter that fuses
GPS position measurements with IMU acceleration measurements.  Periods
with missing GPS data are simulated to show how the filter handles
prediction-only updates.

The example converts latitude/longitude measurements to UTM coordinates
using ``pyproj`` so that all calculations can be performed in meters.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

from pyproj import Transformer


@dataclass
class KalmanFilter:
    """Simple Kalman Filter for 2D position, velocity and acceleration."""

    x: np.ndarray  # State vector [x, y, vx, vy, ax, ay]
    P: np.ndarray  # Covariance matrix
    Q: np.ndarray  # Process noise matrix
    R_gps: np.ndarray  # Measurement noise for GPS
    R_imu: np.ndarray  # Measurement noise for IMU

    def predict(self, dt: float) -> None:
        """Prediction step using constant acceleration model."""
        F = np.array([
            [1, 0, dt, 0, 0.5 * dt ** 2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt ** 2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, z: np.ndarray) -> np.ndarray:
        """GPS update step."""
        H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        return y  # residual

    def update_imu(self, z: np.ndarray) -> np.ndarray:
        """IMU update step."""
        H = np.array([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        return y  # residual


def generate_synthetic_data(
    steps: int,
    dt: float,
    gps_interval: int,
    dropout_range: Tuple[int, int],
    gps_noise: float,
    imu_noise: float,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[np.ndarray]]:
    """Generate synthetic GPS and IMU data.

    Returns
    -------
    gps_measurements : list of (lat, lon) tuples, some entries are None during dropouts
    imu_measurements : list of (ax, ay) tuples
    true_states : list of np.ndarray true state vectors
    """
    # Circular motion parameters
    R = 50.0  # meters
    omega = 0.2  # rad/s
    base_lat, base_lon = 37.0, -122.0

    proj_utm = Transformer.from_crs("epsg:4326", "epsg:32610", always_xy=True)
    proj_ll = Transformer.from_crs("epsg:32610", "epsg:4326", always_xy=True)
    base_x, base_y = proj_utm.transform(base_lon, base_lat)

    gps_data: List[Tuple[float, float] | None] = []
    imu_data: List[Tuple[float, float]] = []
    true_states: List[np.ndarray] = []

    for k in range(steps):
        t = k * dt
        x = base_x + R * np.cos(omega * t)
        y = base_y + R * np.sin(omega * t)
        vx = -R * omega * np.sin(omega * t)
        vy = R * omega * np.cos(omega * t)
        ax = -R * omega**2 * np.cos(omega * t)
        ay = -R * omega**2 * np.sin(omega * t)
        true_states.append(np.array([x, y, vx, vy, ax, ay]))

        # Add noise to IMU acceleration
        ax_meas = ax + np.random.normal(scale=imu_noise)
        ay_meas = ay + np.random.normal(scale=imu_noise)
        imu_data.append((ax_meas, ay_meas))

        if k % gps_interval == 0 and not (dropout_range[0] <= k <= dropout_range[1]):
            # GPS measurement available
            lat, lon = proj_ll.transform(x, y)
            lat += np.random.normal(scale=gps_noise)
            lon += np.random.normal(scale=gps_noise)
            gps_data.append((lat, lon))
        else:
            gps_data.append(None)

    return gps_data, imu_data, true_states


def run_simulation() -> None:
    """Run the Kalman Filter simulation with synthetic data."""
    np.random.seed(0)
    steps = 300
    dt = 0.1
    gps_interval = 10  # GPS at 1 Hz when dt=0.1
    dropout_range = (120, 200)  # simulate GPS dropout

    gps_noise = 1e-5  # noise in degrees
    imu_noise = 0.2  # acceleration noise (m/s^2)

    gps_data, imu_data, true_states = generate_synthetic_data(
        steps, dt, gps_interval, dropout_range, gps_noise, imu_noise
    )

    proj_utm = Transformer.from_crs("epsg:4326", "epsg:32610", always_xy=True)

    # Kalman Filter initialization
    x0 = np.zeros(6)
    P0 = np.eye(6) * 1.0
    Q = np.eye(6) * 0.01
    R_gps = np.eye(2) * 4.0
    R_imu = np.eye(2) * (imu_noise**2)

    kf = KalmanFilter(x=x0, P=P0, Q=Q, R_gps=R_gps, R_imu=R_imu)

    estimates = []
    predictions = []
    gps_residuals = []

    for k in range(steps):
        kf.predict(dt)
        predictions.append(kf.x.copy())

        # IMU update every step
        z_imu = np.array(imu_data[k])
        kf.update_imu(z_imu)

        if gps_data[k] is not None:
            lat, lon = gps_data[k]
            x_gps, y_gps = proj_utm.transform(lon, lat)
            z_gps = np.array([x_gps, y_gps])
            res = kf.update_gps(z_gps)
            gps_residuals.append(res)
        else:
            gps_residuals.append(np.array([np.nan, np.nan]))

        estimates.append(kf.x.copy())

    estimates = np.array(estimates)
    predictions = np.array(predictions)
    true_xy = np.array([[s[0], s[1]] for s in true_states])

    gps_xy = []
    for g in gps_data:
        if g is not None:
            gps_xy.append(proj_utm.transform(g[1], g[0]))
        else:
            gps_xy.append((np.nan, np.nan))
    gps_xy = np.array(gps_xy)

    plot_results(true_xy, gps_xy, estimates, predictions, gps_residuals, dt, dropout_range)


def plot_results(true_xy, gps_xy, est, pred, residuals, dt, dropout_range):
    """Create plots for the simulation."""
    t = np.arange(len(est)) * dt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gps_xy[:, 0], gps_xy[:, 1], "rx", label="GPS measurements")
    ax.plot(est[:, 0], est[:, 1], "b-", label="Filtered")
    ax.plot(pred[:, 0], pred[:, 1], "g--", label="Prediction")
    ax.plot(true_xy[:, 0], true_xy[:, 1], "k:", label="True path")
    ax.legend()
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_title("Kalman Filter Tracking")
    ax.grid(True)

    ax2 = fig.add_axes([0.15, 0.6, 0.25, 0.25])
    res = np.array(residuals)
    ax2.plot(t, res[:, 0], label="x residual")
    ax2.plot(t, res[:, 1], label="y residual")
    ax2.set_xlabel("Time [s]")
    ax2.set_title("GPS residuals")
    ax2.legend(fontsize="small")
    ax2.grid(True)

    ax.axvspan(t[dropout_range[0]], t[dropout_range[1]], color="yellow", alpha=0.3, label="GPS dropout")

    plt.show()


if __name__ == "__main__":
    run_simulation()
