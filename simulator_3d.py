"""
3D Simulator Display - Cylindrical Device with Hand Model

Visual representation of:
- Cylindrical device (shaft)
- Hand grip indicator
- Rotation angle
- Up/down position
- Stroke intensity
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import math


class Simulator3D:
    """
    3D visualization of device and hand movement.

    Shows:
    - Cylindrical shaft (vertical)
    - Hand grip (ring around shaft)
    - Rotation indicator
    - Position marker
    - Intensity color coding
    """

    def __init__(self, width: int = 400, height: int = 600):
        """
        Initialize 3D simulator.

        Args:
            width: Display width
            height: Display height
        """
        self.width = width
        self.height = height

        # Cylinder parameters
        self.cylinder_radius = 40  # pixels
        self.cylinder_height = 400  # pixels
        self.cylinder_center_x = width // 2
        self.cylinder_top_y = 50
        self.cylinder_bottom_y = self.cylinder_top_y + self.cylinder_height

        # Hand grip parameters
        self.hand_radius_outer = 60
        self.hand_radius_inner = 35
        self.hand_height = 40

        # Colors
        self.cylinder_color = (200, 200, 200)  # Light gray
        self.hand_color = (255, 200, 150)  # Skin tone
        self.rotation_color = (0, 255, 0)  # Green
        self.intensity_low = (0, 255, 0)  # Green
        self.intensity_high = (0, 0, 255)  # Red

        # Window name
        self.window_name = "3D Simulator"

    def render(self,
               position: float,  # 0-100 (bottom to top)
               rotation_angle: float,  # -180 to 180 degrees
               intensity: float,  # 0-100
               frequency: float = 0.0,  # Hz
               spm: float = 0.0  # Strokes per minute
               ) -> np.ndarray:
        """
        Render the 3D simulator view.

        Args:
            position: Vertical position (0=bottom, 100=top)
            rotation_angle: Rotation angle in degrees
            intensity: Stroke intensity (0-100)
            frequency: Current frequency
            spm: Strokes per minute

        Returns:
            Rendered image
        """
        # Create black background
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)  # Dark gray background

        # Draw cylinder (shaft)
        self._draw_cylinder(img)

        # Calculate hand position
        hand_y = self._position_to_y(position)

        # Draw hand grip
        self._draw_hand_grip(img, hand_y, rotation_angle, intensity)

        # Draw rotation indicator
        self._draw_rotation_indicator(img, hand_y, rotation_angle)

        # Draw position marker on side
        self._draw_position_marker(img, position)

        # Draw info panel
        self._draw_info_panel(img, position, rotation_angle, intensity, frequency, spm)

        # Draw intensity bar
        self._draw_intensity_bar(img, intensity)

        return img

    def _draw_cylinder(self, img: np.ndarray):
        """Draw the cylindrical shaft."""
        # Draw main cylinder body (rectangle with ellipses on top/bottom)

        # Top ellipse
        cv2.ellipse(img,
                   (self.cylinder_center_x, self.cylinder_top_y),
                   (self.cylinder_radius, self.cylinder_radius // 3),
                   0, 0, 360,
                   self.cylinder_color, -1)

        # Body rectangle
        cv2.rectangle(img,
                     (self.cylinder_center_x - self.cylinder_radius, self.cylinder_top_y),
                     (self.cylinder_center_x + self.cylinder_radius, self.cylinder_bottom_y),
                     self.cylinder_color, -1)

        # Bottom ellipse
        cv2.ellipse(img,
                   (self.cylinder_center_x, self.cylinder_bottom_y),
                   (self.cylinder_radius, self.cylinder_radius // 3),
                   0, 0, 360,
                   self.cylinder_color, -1)

        # Add shading for 3D effect
        # Left side darker
        overlay = img.copy()
        cv2.rectangle(overlay,
                     (self.cylinder_center_x - self.cylinder_radius, self.cylinder_top_y),
                     (self.cylinder_center_x - self.cylinder_radius // 2, self.cylinder_bottom_y),
                     (150, 150, 150), -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Right side lighter
        overlay = img.copy()
        cv2.rectangle(overlay,
                     (self.cylinder_center_x + self.cylinder_radius // 2, self.cylinder_top_y),
                     (self.cylinder_center_x + self.cylinder_radius, self.cylinder_bottom_y),
                     (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

        # Draw center line for reference
        cv2.line(img,
                (self.cylinder_center_x, self.cylinder_top_y),
                (self.cylinder_center_x, self.cylinder_bottom_y),
                (180, 180, 180), 1, cv2.LINE_AA)

    def _draw_hand_grip(self, img: np.ndarray, hand_y: int,
                       rotation_angle: float, intensity: float):
        """Draw the hand grip around the shaft."""
        # Color based on intensity
        color = self._get_intensity_color(intensity)

        # Draw outer ring (hand)
        cv2.ellipse(img,
                   (self.cylinder_center_x, hand_y),
                   (self.hand_radius_outer, self.hand_radius_outer // 3),
                   0, 0, 360,
                   color, -1)

        # Draw inner cutout (hole for shaft)
        cv2.ellipse(img,
                   (self.cylinder_center_x, hand_y),
                   (self.hand_radius_inner, self.hand_radius_inner // 3),
                   0, 0, 360,
                   (30, 30, 30), -1)

        # Draw hand thickness (top and bottom)
        hand_half_height = self.hand_height // 2

        # Top part of hand
        cv2.rectangle(img,
                     (self.cylinder_center_x - self.hand_radius_outer,
                      hand_y - hand_half_height),
                     (self.cylinder_center_x + self.hand_radius_outer,
                      hand_y),
                     color, -1)

        # Bottom part of hand
        cv2.rectangle(img,
                     (self.cylinder_center_x - self.hand_radius_outer,
                      hand_y),
                     (self.cylinder_center_x + self.hand_radius_outer,
                      hand_y + hand_half_height),
                     color, -1)

        # Cut out inner cylinder from hand body
        cv2.rectangle(img,
                     (self.cylinder_center_x - self.hand_radius_inner,
                      hand_y - hand_half_height),
                     (self.cylinder_center_x + self.hand_radius_inner,
                      hand_y + hand_half_height),
                     (30, 30, 30), -1)

        # Add shading
        overlay = img.copy()
        # Left side darker
        pts_left = np.array([
            [self.cylinder_center_x - self.hand_radius_outer, hand_y - hand_half_height],
            [self.cylinder_center_x - self.hand_radius_inner, hand_y - hand_half_height],
            [self.cylinder_center_x - self.hand_radius_inner, hand_y + hand_half_height],
            [self.cylinder_center_x - self.hand_radius_outer, hand_y + hand_half_height]
        ], np.int32)
        cv2.fillPoly(overlay, [pts_left], (int(color[0]*0.7), int(color[1]*0.7), int(color[2]*0.7)))
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # Draw grip lines for texture
        for i in range(5):
            line_y = hand_y - hand_half_height + i * (self.hand_height // 5)
            cv2.line(img,
                    (self.cylinder_center_x - self.hand_radius_outer, line_y),
                    (self.cylinder_center_x + self.hand_radius_outer, line_y),
                    (int(color[0]*0.8), int(color[1]*0.8), int(color[2]*0.8)),
                    1, cv2.LINE_AA)

    def _draw_rotation_indicator(self, img: np.ndarray, hand_y: int, angle: float):
        """Draw rotation angle indicator."""
        arrow_radius = self.hand_radius_outer + 10

        # Determine arrow color and direction based on angle
        if angle < -5:
            arrow_color = (0, 0, 255)  # 红色(BGR) = 向左
            direction = -1  # 向左
        elif angle > 5:
            arrow_color = (0, 255, 0)  # 绿色(BGR) = 向右
            direction = 1  # 向右
        else:
            arrow_color = (150, 150, 150)  # 灰色 = 中立
            direction = 0

        # Draw horizontal arrow (left/right)
        if direction != 0:
            arrow_length = int(arrow_radius * abs(angle) / 90.0)  # 长度与角度成正比
            arrow_length = max(30, min(arrow_length, arrow_radius))  # 限制长度

            # 起点和终点
            # direction = -1: 向左 (end_x < start_x, arrow points LEFT)
            # direction = +1: 向右 (end_x > start_x, arrow points RIGHT)
            start_x = self.cylinder_center_x
            end_x = self.cylinder_center_x + (arrow_length * direction)

            # Draw arrow from center to end point
            cv2.arrowedLine(img,
                           (start_x, hand_y),
                           (end_x, hand_y),
                           arrow_color, 4, cv2.LINE_AA, tipLength=0.4)
        else:
            # Draw small circle for neutral
            cv2.circle(img, (self.cylinder_center_x, hand_y), 8, arrow_color, -1)

        # Draw angle text
        angle_text = f"{angle:.1f}°"
        text_x = self.cylinder_center_x + arrow_radius + 15
        if angle < 0:
            text_x = self.cylinder_center_x - arrow_radius - 80

        cv2.putText(img, angle_text,
                   (text_x, hand_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   arrow_color, 2, cv2.LINE_AA)

        # Draw rotation direction label
        if abs(angle) > 5:
            direction_text = "LEFT" if angle < 0 else "RIGHT"
            cv2.putText(img, direction_text,
                       (text_x, hand_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       arrow_color, 1, cv2.LINE_AA)

        # Draw curved arc indicator
        if abs(angle) > 5:
            arc_color = (0, 255, 0) if angle > 0 else (0, 0, 255)  # 绿色(右) or 红色(左) in BGR

            # Draw arc above the arrow
            arc_y = hand_y - 15
            arc_radius_x = arrow_radius

            if angle > 0:
                # 向右的弧线
                cv2.ellipse(img,
                           (self.cylinder_center_x, arc_y),
                           (arc_radius_x, 10),
                           0, 0, min(90, int(angle)),
                           arc_color, 2, cv2.LINE_AA)
            else:
                # 向左的弧线
                cv2.ellipse(img,
                           (self.cylinder_center_x, arc_y),
                           (arc_radius_x, 10),
                           0, max(-90, int(angle)), 0,
                           arc_color, 2, cv2.LINE_AA)

    def _draw_position_marker(self, img: np.ndarray, position: float):
        """Draw position marker on the side."""
        marker_x = 30
        marker_width = 15

        # Draw scale
        scale_top_y = self.cylinder_top_y
        scale_bottom_y = self.cylinder_bottom_y
        scale_height = scale_bottom_y - scale_top_y

        # Draw scale line
        cv2.line(img, (marker_x, scale_top_y), (marker_x, scale_bottom_y),
                (150, 150, 150), 2)

        # Draw tick marks
        for i in range(11):  # 0, 10, 20, ..., 100
            tick_y = scale_top_y + int(i * scale_height / 10)
            cv2.line(img, (marker_x - 5, tick_y), (marker_x + 5, tick_y),
                    (150, 150, 150), 2)

            # Label every 25%
            if i % 2 == 0:
                label = f"{100 - i*10}"
                cv2.putText(img, label, (marker_x - 25, tick_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                           (200, 200, 200), 1, cv2.LINE_AA)

        # Draw current position marker
        current_y = self._position_to_y(position)

        # Draw marker triangle
        pts = np.array([
            [marker_x + 10, current_y],
            [marker_x + 25, current_y - 8],
            [marker_x + 25, current_y + 8]
        ], np.int32)
        cv2.fillPoly(img, [pts], (0, 255, 255))
        cv2.polylines(img, [pts], True, (0, 200, 200), 2, cv2.LINE_AA)

        # Position value
        cv2.putText(img, f"{position:.0f}",
                   (marker_x + 30, current_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 255, 255), 1, cv2.LINE_AA)

    def _draw_info_panel(self, img: np.ndarray, position: float,
                        rotation_angle: float, intensity: float,
                        frequency: float, spm: float):
        """Draw information panel at the bottom."""
        panel_y = self.height - 150
        panel_h = 140

        # Background
        cv2.rectangle(img, (0, panel_y), (self.width, self.height),
                     (20, 20, 20), -1)
        cv2.rectangle(img, (0, panel_y), (self.width, self.height),
                     (100, 100, 100), 2)

        # Title
        cv2.putText(img, "3D Simulator", (10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2, cv2.LINE_AA)

        # Info rows
        y_offset = panel_y + 50
        line_height = 20

        infos = [
            f"Position: {position:.1f}%",
            f"Rotation: {rotation_angle:.1f}°",
            f"Intensity: {intensity:.1f}%",
            f"Frequency: {frequency:.2f} Hz",
            f"SPM: {spm:.0f}"
        ]

        for i, info in enumerate(infos):
            cv2.putText(img, info,
                       (15, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_intensity_bar(self, img: np.ndarray, intensity: float):
        """Draw intensity bar on the right side."""
        bar_x = self.width - 40
        bar_width = 20
        bar_top_y = self.cylinder_top_y
        bar_height = self.cylinder_height

        # Draw bar background
        cv2.rectangle(img, (bar_x, bar_top_y),
                     (bar_x + bar_width, bar_top_y + bar_height),
                     (50, 50, 50), -1)
        cv2.rectangle(img, (bar_x, bar_top_y),
                     (bar_x + bar_width, bar_top_y + bar_height),
                     (100, 100, 100), 2)

        # Draw filled portion
        fill_height = int(bar_height * intensity / 100.0)
        fill_top_y = bar_top_y + bar_height - fill_height

        color = self._get_intensity_color(intensity)
        cv2.rectangle(img, (bar_x, fill_top_y),
                     (bar_x + bar_width, bar_top_y + bar_height),
                     color, -1)

        # Label
        cv2.putText(img, "Intensity", (bar_x - 15, bar_top_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                   (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"{intensity:.0f}%", (bar_x - 10, bar_top_y + bar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   color, 1, cv2.LINE_AA)

    def _position_to_y(self, position: float) -> int:
        """Convert position (0-100) to Y coordinate."""
        # 0 = bottom, 100 = top
        normalized = np.clip(position / 100.0, 0, 1)
        y = self.cylinder_bottom_y - int(normalized * self.cylinder_height)
        return y

    def _get_intensity_color(self, intensity: float) -> Tuple[int, int, int]:
        """Get color based on intensity (green to red gradient)."""
        # Interpolate between green and red
        ratio = np.clip(intensity / 100.0, 0, 1)

        b = int(self.intensity_low[0] * (1 - ratio) + self.intensity_high[0] * ratio)
        g = int(self.intensity_low[1] * (1 - ratio) + self.intensity_high[1] * ratio)
        r = int(self.intensity_low[2] * (1 - ratio) + self.intensity_high[2] * ratio)

        return (b, g, r)

    def show(self, position: float, rotation_angle: float, intensity: float,
             frequency: float = 0.0, spm: float = 0.0):
        """Render and display the simulator."""
        img = self.render(position, rotation_angle, intensity, frequency, spm)
        cv2.imshow(self.window_name, img)
