"""
Real-time Graph Display - Shows frequency and position curves

Displays live graphs of:
- Frequency (Hz) over time
- Primary position (vertical, 0-100) over time
- Secondary position (horizontal, 0-100) over time
- Motion intensity
"""

import numpy as np
import cv2
from collections import deque
from typing import Tuple, Optional
import time


class RealtimeGraphDisplay:
    """
    Real-time graph display using OpenCV for performance.

    Shows frequency and position curves updating in real-time.
    """

    def __init__(self, width: int = 800, height: int = 600, window_seconds: float = 10.0):
        """
        Initialize graph display.

        Args:
            width: Window width in pixels
            height: Window height in pixels
            window_seconds: How many seconds to display
        """
        self.width = width
        self.height = height
        self.window_seconds = window_seconds

        # Graph areas (4 graphs stacked vertically)
        # Reserve 100px at bottom for current values panel
        self.panel_height = 90
        available_height = height - self.panel_height
        self.graph_height = available_height // 4

        self.graphs = [
            {'name': 'Frequency (Hz)', 'y_min': 0, 'y_max': 5, 'color': (0, 255, 0), 'y_offset': 0},
            {'name': 'Primary Position', 'y_min': 0, 'y_max': 100, 'color': (255, 0, 0), 'y_offset': self.graph_height},
            {'name': 'Secondary Position', 'y_min': 0, 'y_max': 100, 'color': (0, 0, 255), 'y_offset': self.graph_height * 2},
            {'name': 'Motion Intensity', 'y_min': 0, 'y_max': 100, 'color': (255, 255, 0), 'y_offset': self.graph_height * 3}
        ]

        # Data buffers
        self.frequency_data = deque(maxlen=1000)
        self.primary_data = deque(maxlen=1000)
        self.secondary_data = deque(maxlen=1000)
        self.motion_data = deque(maxlen=1000)
        self.timestamp_data = deque(maxlen=1000)

        # Window name
        self.window_name = "Real-time Frequency & Position Graphs"

    def update(self, frequency: float, primary_pos: float,
               secondary_pos: float, motion_intensity: float):
        """
        Update graphs with new data.

        Args:
            frequency: Current frequency in Hz
            primary_pos: Primary axis position (0-100)
            secondary_pos: Secondary axis position (0-100)
            motion_intensity: Motion intensity (0-100)
        """
        current_time = time.time()

        self.frequency_data.append(frequency)
        self.primary_data.append(primary_pos)
        self.secondary_data.append(secondary_pos)
        self.motion_data.append(motion_intensity)
        self.timestamp_data.append(current_time)

    def render(self) -> np.ndarray:
        """
        Render the graph display.

        Returns:
            Image array (BGR format)
        """
        # Create black background
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Draw each graph
        datasets = [
            self.frequency_data,
            self.primary_data,
            self.secondary_data,
            self.motion_data
        ]

        for graph_idx, (graph_config, data) in enumerate(zip(self.graphs, datasets)):
            self._draw_graph(img, graph_config, data)

        # Draw current values
        self._draw_current_values(img)

        return img

    def _draw_graph(self, img: np.ndarray, config: dict, data: deque):
        """Draw a single graph on the image."""
        if len(data) < 2 or len(self.timestamp_data) < 2:
            # Not enough data, just draw the frame
            self._draw_graph_frame(img, config)
            return

        # Get window of data
        timestamps = np.array(list(self.timestamp_data))
        values = np.array(list(data))

        current_time = timestamps[-1]
        cutoff_time = current_time - self.window_seconds

        # Filter to time window
        mask = timestamps >= cutoff_time
        timestamps = timestamps[mask]
        values = values[mask]

        if len(timestamps) < 2:
            self._draw_graph_frame(img, config)
            return

        # Normalize timestamps to 0-window_seconds
        timestamps = timestamps - timestamps[0]

        # Draw graph frame
        self._draw_graph_frame(img, config)

        # Convert data to pixel coordinates
        points = []
        for t, v in zip(timestamps, values):
            # X coordinate (time)
            x = int((t / self.window_seconds) * (self.width - 60) + 40)

            # Y coordinate (value)
            y_range = config['y_max'] - config['y_min']
            if y_range > 0:
                y_normalized = (v - config['y_min']) / y_range
            else:
                y_normalized = 0.5

            y = int((1.0 - y_normalized) * (self.graph_height - 40) + 20 + config['y_offset'])

            points.append((x, y))

        # Draw line connecting points
        if len(points) >= 2:
            for i in range(1, len(points)):
                cv2.line(img, points[i-1], points[i], config['color'], 2)

        # Draw current value marker
        if len(points) > 0:
            cv2.circle(img, points[-1], 4, config['color'], -1)

    def _draw_graph_frame(self, img: np.ndarray, config: dict):
        """Draw the frame and labels for a graph."""
        y_offset = config['y_offset']

        # Draw border
        cv2.rectangle(img, (30, y_offset + 10),
                     (self.width - 30, y_offset + self.graph_height - 10),
                     (100, 100, 100), 1)

        # Draw title
        cv2.putText(img, config['name'], (40, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw Y-axis labels
        y_min_text = f"{config['y_min']:.1f}"
        y_max_text = f"{config['y_max']:.1f}"

        cv2.putText(img, y_max_text, (5, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, y_min_text, (5, y_offset + self.graph_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Draw horizontal grid lines
        for i in range(1, 4):
            y = y_offset + int(i * self.graph_height / 4)
            cv2.line(img, (35, y), (self.width - 35, y), (50, 50, 50), 1)

    def _draw_current_values(self, img: np.ndarray):
        """Draw current values at the bottom."""
        if len(self.frequency_data) == 0:
            return

        # Get current values
        freq = self.frequency_data[-1]
        primary = self.primary_data[-1]
        secondary = self.secondary_data[-1]
        motion = self.motion_data[-1]

        # Calculate strokes per minute
        spm = freq * 60.0

        # Draw panel at the bottom (after all graphs)
        available_height = self.height - self.panel_height
        panel_y = available_height + 5  # 5px gap after graphs

        cv2.rectangle(img, (10, panel_y), (self.width - 10, self.height - 10),
                     (30, 30, 30), -1)
        cv2.rectangle(img, (10, panel_y), (self.width - 10, self.height - 10),
                     (100, 100, 100), 2)

        # Title
        cv2.putText(img, "Current Values:", (20, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Values in columns
        col1_x = 180
        col2_x = 400
        col3_x = 620
        value_y = panel_y + 25

        # Column 1: Frequency
        cv2.putText(img, f"Freq: {freq:.2f} Hz", (col1_x, value_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, f"SPM: {spm:.1f}", (col1_x, value_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Column 2: Positions
        cv2.putText(img, f"Primary: {primary:.1f}", (col2_x, value_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(img, f"Secondary: {secondary:.1f}", (col2_x, value_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Column 3: Motion
        cv2.putText(img, f"Motion: {motion:.1f}%", (col3_x, value_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def show(self):
        """Display the graph window."""
        img = self.render()
        cv2.imshow(self.window_name, img)

    def reset(self):
        """Clear all data."""
        self.frequency_data.clear()
        self.primary_data.clear()
        self.secondary_data.clear()
        self.motion_data.clear()
        self.timestamp_data.clear()


class CompactGraphDisplay:
    """
    Compact single-window graph display.
    Shows frequency and positions in a small overlay.
    """

    def __init__(self, width: int = 400, height: int = 300):
        """Initialize compact display."""
        self.width = width
        self.height = height
        self.frequency_history = deque(maxlen=100)
        self.primary_history = deque(maxlen=100)

    def update(self, frequency: float, primary_pos: float):
        """Update with new data."""
        self.frequency_history.append(frequency)
        self.primary_history.append(primary_pos)

    def render(self) -> np.ndarray:
        """Render compact display."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        # Draw frequency graph (top half)
        if len(self.frequency_history) >= 2:
            self._draw_mini_graph(img, self.frequency_history, 0, self.height // 2,
                                (0, 255, 0), y_max=5.0, label="Frequency (Hz)")

        # Draw position graph (bottom half)
        if len(self.primary_history) >= 2:
            self._draw_mini_graph(img, self.primary_history, self.height // 2, self.height // 2,
                                (255, 0, 0), y_max=100.0, label="Position")

        # Current values
        if len(self.frequency_history) > 0:
            freq = self.frequency_history[-1]
            pos = self.primary_history[-1]

            cv2.putText(img, f"{freq:.2f} Hz | {freq*60:.0f} SPM",
                       (10, self.height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

    def _draw_mini_graph(self, img: np.ndarray, data: deque, y_offset: int,
                        graph_height: int, color: Tuple[int, int, int],
                        y_max: float, label: str):
        """Draw a mini graph."""
        # Label
        cv2.putText(img, label, (10, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Convert data to points
        values = np.array(list(data))
        if len(values) < 2:
            return

        points = []
        for i, v in enumerate(values):
            x = int((i / len(values)) * (self.width - 20) + 10)
            y_normalized = np.clip(v / y_max, 0, 1)
            y = int((1.0 - y_normalized) * (graph_height - 40) + y_offset + 30)
            points.append((x, y))

        # Draw line
        for i in range(1, len(points)):
            cv2.line(img, points[i-1], points[i], color, 2)

    def render_overlay(self, frame: np.ndarray, position: Tuple[int, int] = (10, 10)) -> np.ndarray:
        """
        Render as overlay on another frame.

        Args:
            frame: Base frame to overlay on
            position: (x, y) position for top-left corner

        Returns:
            Frame with overlay
        """
        graph_img = self.render()

        x, y = position
        h, w = graph_img.shape[:2]

        # Ensure overlay fits
        if y + h > frame.shape[0]:
            y = frame.shape[0] - h
        if x + w > frame.shape[1]:
            x = frame.shape[1] - w

        # Blend overlay onto frame
        roi = frame[y:y+h, x:x+w]
        blended = cv2.addWeighted(roi, 0.3, graph_img, 0.7, 0)
        frame[y:y+h, x:x+w] = blended

        return frame
