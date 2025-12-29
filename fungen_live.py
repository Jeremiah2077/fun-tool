"""
FunGen Live Screen - 完整独立版本

包含所有FunGen源码和模型:
- TrackerManager (FunGen实时追踪)
- DualAxisFunscript (脚本生成)
- 实时图表 (频率、位置、强度)
- 3D模拟器
- YOLO检测可视化
"""

import sys
import os
from pathlib import Path

# 无需添加外部路径 - 所有模块已在本地

import cv2
import numpy as np
import time
import mss
from typing import Optional, Dict, Tuple, List
from collections import deque
from scipy.signal import find_peaks

# 导入FunGen模块（现在是本地模块）
from tracker.tracker_manager import TrackerManager
from funscript.dual_axis_funscript import DualAxisFunscript

# 导入可视化模块
from graph_display import RealtimeGraphDisplay
from simulator_3d import Simulator3D


class DummyApp:
    """模拟应用实例"""

    class AppSettings(dict):
        """应用设置字典，支持get方法"""
        def get(self, key, default=None):
            return super().get(key, default)

    def __init__(self):
        self.hardware_acceleration_method = 'cuda' if self._has_cuda() else 'none'
        self.available_ffmpeg_hwaccels = ['cuda'] if self._has_cuda() else []

        # 追踪器需要的属性
        self.tracking_axis_mode = 'vertical'  # 轴模式: 'both', 'vertical', 'horizontal'
        self.single_axis_output_target = 'primary'  # 单轴输出目标（primary/secondary）

        # 设置字典（需要支持.get()方法）
        self.app_settings = self.AppSettings({
            'live_oscillation_dynamic_amp_enabled': True
        })

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False


class ScreenCapture:
    """屏幕捕获器"""

    def __init__(self, monitor_index: int = 1, scale: float = 0.5):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_index]
        self.scale = scale
        print(f"捕获监视器: {self.monitor} (缩放: {scale}x)")

    def capture_frame(self) -> np.ndarray:
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 缩放以提升性能
        if self.scale != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.scale)
            new_h = int(h * self.scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return frame

    def close(self):
        self.sct.close()


class FunGenLiveScreen:
    """FunGen屏幕实时追踪 - 完整版"""

    def __init__(self,
                 tracker_model_path: str,
                 tracker_mode: str = "yolo_roi",
                 target_fps: int = 30,
                 monitor_index: int = 1,
                 capture_scale: float = 0.5):
        """初始化"""
        print("=" * 70)
        print("FunGen Live Screen - 完整版本")
        print("=" * 70)
        print()

        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        # 创建应用实例
        print("初始化应用实例...")
        self.app = DummyApp()

        # 初始化TrackerManager
        print(f"初始化TrackerManager (模式: {tracker_mode})...")
        self.tracker = TrackerManager(
            app_logic_instance=self.app,
            tracker_model_path=tracker_model_path
        )
        self.tracker.set_tracking_mode(tracker_mode)

        # 屏幕捕获
        print(f"初始化屏幕捕获...")
        self.screen_capture = ScreenCapture(
            monitor_index=monitor_index,
            scale=capture_scale
        )

        # 图表显示
        print("初始化实时图表...")
        self.graph_display = RealtimeGraphDisplay(
            width=800,
            height=700,
            window_seconds=10.0
        )

        # 3D模拟器
        print("初始化3D模拟器...")
        self.simulator = Simulator3D(width=500, height=700)

        # 数据历史（用于频率和强度计算）
        self.position_history = deque(maxlen=target_fps * 10)
        self.timestamp_history = deque(maxlen=target_fps * 10)

        # 旋转计算历史
        self.velocity_history = deque(maxlen=30)  # 速度历史
        self.rotation_value = 50.0  # 当前旋转值

        # 状态
        self.running = False
        self.frame_count = 0
        self.start_time = None

        # UI状态
        self.show_overlay = True
        self.show_graphs = True
        self.show_simulator = True

        # 性能统计
        self.fps_history = []
        self.last_fps_update = time.time()
        self.frames_since_update = 0

        # 更新间隔优化
        self.update_counter = 0
        self.graph_interval = 2
        self.sim_interval = 2

        print()
        print("✓ 初始化完成！")
        print()

    def run(self):
        """运行主循环"""
        print("启动FunGen Live Screen...")
        print()
        print("三个窗口:")
        print("  1. FunGen Live (主追踪视图)")
        print("  2. Real-time Graphs (频率、位置、强度曲线)")
        print("  3. 3D Simulator (圆柱体动画)")
        print()
        print("控制键:")
        print("  S = 开始/停止追踪 (重要！！！)")
        print("  Q/ESC = 退出")
        print("  R = 重置")
        print("  G = 切换图表窗口")
        print("  D = 切换3D窗口")
        print("  O = 切换信息叠加")
        print()
        print("提示: 按 S 键开始追踪！")
        print()

        cv2.namedWindow('FunGen Live', cv2.WINDOW_NORMAL)

        try:
            while True:
                loop_start = time.time()

                # 捕获屏幕
                frame = self.screen_capture.capture_frame()

                # 计算时间戳
                if self.start_time is None:
                    self.start_time = time.time()
                elapsed_ms = int((time.time() - self.start_time) * 1000)

                # 处理帧
                if self.running:
                    # 使用FunGen TrackerManager
                    processed_frame, action_log = self.tracker.process_frame(
                        frame=frame,
                        frame_time_ms=elapsed_ms,
                        frame_index=self.frame_count
                    )

                    # 记录位置历史
                    if action_log and len(action_log) > 0:
                        last_pos = action_log[-1].get('pos', 50)
                        # 确保位置不是None
                        if last_pos is None:
                            last_pos = 50
                        self.position_history.append(last_pos)
                        self.timestamp_history.append(time.time())

                    display_frame = processed_frame
                else:
                    display_frame = frame.copy()
                    # 绘制大提示
                    self._draw_start_hint(display_frame)

                # 绘制信息叠加
                if self.show_overlay:
                    self._draw_overlay(display_frame, elapsed_ms)

                # 显示主窗口
                cv2.imshow('FunGen Live', display_frame)

                # 更新图表和3D
                self.update_counter += 1
                if self.update_counter % self.graph_interval == 0:
                    # 计算当前数据
                    frequency, spm = self._calculate_frequency()
                    intensity = self._calculate_intensity()
                    current_pos = self.position_history[-1] if len(self.position_history) > 0 else 50.0
                    # 确保位置不是None
                    if current_pos is None:
                        current_pos = 50.0

                    # 计算旋转（使用新方法）
                    rotation = self._calculate_rotation()

                    # 更新图表
                    if self.show_graphs:
                        self.graph_display.update(
                            frequency=frequency,
                            primary_pos=current_pos,
                            secondary_pos=rotation,  # 使用计算的旋转值
                            motion_intensity=intensity
                        )
                        graph_frame = self.graph_display.render()
                        cv2.imshow('Real-time Graphs', graph_frame)

                    # 更新3D - 转换旋转到角度
                    if self.show_simulator:
                        # 旋转角度: 0->-90°, 50->0°, 100->+90°
                        rotation_angle = (rotation - 50.0) * 1.8
                        sim_frame = self.simulator.render(
                            position=current_pos,
                            rotation_angle=rotation_angle,
                            intensity=intensity,
                            frequency=frequency,
                            spm=spm
                        )
                        cv2.imshow('3D Simulator', sim_frame)

                # 更新统计
                self.frame_count += 1
                self.frames_since_update += 1

                # 计算FPS
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    fps = self.frames_since_update / (current_time - self.last_fps_update)
                    self.fps_history.append(fps)
                    if len(self.fps_history) > 10:
                        self.fps_history.pop(0)

                    avg_fps = np.mean(self.fps_history)
                    status = "追踪中" if self.running else "暂停 [按S开始]"
                    actions = len(self.tracker.funscript.primary_actions) if self.tracker.funscript else 0
                    print(f"FPS: {avg_fps:.1f} | 状态: {status} | 帧: {self.frame_count} | 动作: {actions}", end='\r')

                    self.frames_since_update = 0
                    self.last_fps_update = current_time

                # 键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    if self.running:
                        self.stop_tracking()
                    else:
                        self.start_tracking()
                elif key == ord('r'):
                    self._reset()
                elif key == ord('g'):
                    self.show_graphs = not self.show_graphs
                    if not self.show_graphs:
                        cv2.destroyWindow('Real-time Graphs')
                elif key == ord('d'):
                    self.show_simulator = not self.show_simulator
                    if not self.show_simulator:
                        cv2.destroyWindow('3D Simulator')
                elif key == ord('o'):
                    self.show_overlay = not self.show_overlay

                # 控制帧率
                elapsed = time.time() - loop_start
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)

        except KeyboardInterrupt:
            print("\n\n用户中断")
        finally:
            self._cleanup()

    def start_tracking(self):
        """开始追踪"""
        if not self.running:
            self.tracker.start_tracking()
            self.running = True
            self.start_time = time.time()
            print("\n✓ 追踪已开始！")

    def stop_tracking(self):
        """停止追踪"""
        if self.running:
            self.tracker.stop_tracking()
            self.running = False
            print("\n✓ 追踪已停止")

    def _calculate_frequency(self) -> Tuple[float, float]:
        """计算频率"""
        if len(self.position_history) < 30:
            return 0.0, 0.0

        recent_count = min(len(self.position_history), self.target_fps * 3)
        positions = np.array(list(self.position_history)[-recent_count:])
        timestamps = list(self.timestamp_history)[-recent_count:]

        # 峰谷检测
        peaks, _ = find_peaks(positions, prominence=1.0)
        valleys, _ = find_peaks(-positions, prominence=1.0)

        all_extrema = sorted(list(peaks) + list(valleys))
        if len(all_extrema) < 2:
            return 0.0, 0.0

        # 计算间隔
        intervals = []
        for i in range(1, len(all_extrema)):
            idx1, idx2 = all_extrema[i-1], all_extrema[i]
            if idx1 < len(timestamps) and idx2 < len(timestamps):
                dt = timestamps[idx2] - timestamps[idx1]
                if dt > 0:
                    intervals.append(dt)

        if not intervals:
            return 0.0, 0.0

        avg_interval = np.mean(intervals)
        frequency = 1.0 / (2.0 * avg_interval) if avg_interval > 0 else 0.0
        spm = frequency * 60.0

        return frequency, spm

    def _calculate_intensity(self) -> float:
        """计算强度"""
        if len(self.position_history) < 30:
            return 0.0
        recent = list(self.position_history)[-60:]
        return min(100.0, np.max(recent) - np.min(recent))

    def _calculate_rotation(self) -> float:
        """
        计算旋转值 (0-100, 50为中立)

        策略：基于运动方向
        - 上升时向右旋转 (50-100)
        - 下降时向左旋转 (0-50)
        - 静止时回归中立 (50)
        """
        if len(self.position_history) < 5:
            return 50.0

        # 计算速度（位置变化率）
        recent_positions = list(self.position_history)[-10:]
        velocity = np.mean(np.diff(recent_positions))

        # 保存速度历史用于平滑
        self.velocity_history.append(velocity)

        # 计算平滑速度
        if len(self.velocity_history) >= 3:
            smooth_velocity = np.mean(list(self.velocity_history)[-5:])
        else:
            smooth_velocity = velocity

        # 基于速度计算旋转
        # 速度 > 0 (上升) -> 向右旋转 (50-100)
        # 速度 < 0 (下降) -> 向左旋转 (0-50)
        # 速度 ≈ 0 (静止) -> 中立 (50)

        # 将速度范围映射到旋转值
        # 假设典型速度范围是 -5 到 +5
        velocity_normalized = np.clip(smooth_velocity / 5.0, -1.0, 1.0)

        # 转换为0-100，50为中心
        rotation = 50.0 + (velocity_normalized * 50.0)

        # 添加惯性：平滑过渡到新值
        alpha = 0.7  # 平滑系数 (增加响应速度)
        self.rotation_value = alpha * rotation + (1 - alpha) * self.rotation_value

        return self.rotation_value

    def _draw_start_hint(self, frame: np.ndarray):
        """绘制开始提示"""
        h, w = frame.shape[:2]

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # 大字提示
        text = "Press S to START Tracking"
        font_scale = 1.5
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        x = (w - text_w) // 2
        y = h // 2

        cv2.putText(frame, text, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        # 小字说明
        text2 = "(S=Start | Q=Quit | G=Graphs | D=3D)"
        font_scale2 = 0.8
        (text_w2, text_h2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, font_scale2, 2)
        x2 = (w - text_w2) // 2
        y2 = y + 50

        cv2.putText(frame, text2, (x2, y2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale2, (255, 255, 255), 2)

    def _draw_overlay(self, frame: np.ndarray, elapsed_ms: int):
        """绘制完整的实时参数信息"""
        h, w = frame.shape[:2]

        # 扩大背景以显示更多参数
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 500), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        avg_fps = np.mean(self.fps_history) if self.fps_history else 0.0

        # === 标题 ===
        y = 35
        cv2.putText(frame, "FUNGEN LIVE SCREEN", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # === 状态 ===
        y += 35
        cv2.putText(frame, "=== STATUS ===", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        y += 25
        status = "TRACKING" if self.running else "PAUSED"
        status_color = (0, 255, 0) if self.running else (0, 0, 255)
        cv2.putText(frame, f"Mode: {status}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        y += 23
        fps_color = (0, 255, 0) if avg_fps >= 20 else (0, 255, 255) if avg_fps >= 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)

        y += 23
        time_str = f"{elapsed_ms / 1000:.1f}s"
        cv2.putText(frame, f"Time: {time_str}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # === 主轴参数 (L0) ===
        y += 35
        cv2.putText(frame, "=== PRIMARY AXIS (L0) ===", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        current_pos = self.position_history[-1] if len(self.position_history) > 0 else 50.0
        # 确保位置不是None
        if current_pos is None:
            current_pos = 50.0
        y += 25
        cv2.putText(frame, f"Position: {current_pos:.1f}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 位置条
        bar_x = 150
        bar_y = y - 12
        bar_w = 250
        bar_h = 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
        fill_w = int((current_pos / 100.0) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)

        # 运动范围
        if len(self.position_history) >= 30:
            recent = list(self.position_history)[-60:]
            min_pos = np.min(recent)
            max_pos = np.max(recent)
            y += 23
            cv2.putText(frame, f"Range: {min_pos:.1f} - {max_pos:.1f}", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # === 频率参数 ===
        frequency, spm = self._calculate_frequency()
        y += 35
        cv2.putText(frame, "=== FREQUENCY ===", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        y += 25
        cv2.putText(frame, f"Hz: {frequency:.2f}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y += 23
        cv2.putText(frame, f"SPM: {spm:.1f}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # === 强度参数 ===
        intensity = self._calculate_intensity()
        y += 35
        cv2.putText(frame, "=== INTENSITY ===", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        y += 25
        cv2.putText(frame, f"Amplitude: {intensity:.1f}%", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 强度条
        bar_y = y - 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
        fill_w = int((intensity / 100.0) * bar_w)
        intensity_color = (0, int(255 * (1 - intensity/100)), int(255 * intensity/100))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), intensity_color, -1)

        # === 副轴参数 (L1 - 旋转/滚动) ===
        y += 35
        cv2.putText(frame, "=== SECONDARY AXIS (L1) ===", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 计算旋转（使用统一方法）
        rotation = self._calculate_rotation()

        y += 25
        rotation_dir = "LEFT" if rotation < 45 else "RIGHT" if rotation > 55 else "CENTER"
        cv2.putText(frame, f"Rotation: {rotation:.1f} ({rotation_dir})", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 旋转条（0-50-100，50是中心）
        bar_y = y - 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
        # 中心线
        center_x = bar_x + bar_w // 2
        cv2.line(frame, (center_x, bar_y), (center_x, bar_y + bar_h), (255, 255, 255), 1)
        # 当前位置
        pos_x = bar_x + int((rotation / 100.0) * bar_w)
        cv2.circle(frame, (pos_x, bar_y + bar_h // 2), 6, (0, 0, 255), -1)

        # === Funscript动作统计 ===
        if self.tracker.funscript:
            y += 35
            cv2.putText(frame, "=== FUNSCRIPT ===", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            y += 25
            action_count = len(self.tracker.funscript.primary_actions)
            cv2.putText(frame, f"Actions: {action_count}", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if action_count > 0:
                last_action = self.tracker.funscript.primary_actions[-1]
                y += 23
                cv2.putText(frame, f"Last: {last_action['pos']} @ {last_action['at']}ms", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # === 控制提示 ===
        y += 35
        cv2.putText(frame, "Controls: S=Start/Stop G=Graphs D=3D", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _reset(self):
        """重置"""
        was_running = self.running
        if was_running:
            self.stop_tracking()

        self.tracker.funscript = DualAxisFunscript()
        self.position_history.clear()
        self.timestamp_history.clear()
        self.velocity_history.clear()
        self.rotation_value = 50.0
        self.graph_display.reset()
        self.frame_count = 0
        self.start_time = None

        if was_running:
            self.start_tracking()

        print("\n✓ 已重置")

    def _cleanup(self):
        """清理"""
        print("\n\n清理中...")

        if self.running:
            self.stop_tracking()

        # 显示最终统计
        if self.tracker.funscript and len(self.tracker.funscript.primary_actions) > 0:
            print(f"\n会话统计:")
            print(f"  - 总帧数: {self.frame_count}")
            print(f"  - 总动作数: {len(self.tracker.funscript.primary_actions)}")
            if self.start_time:
                duration = time.time() - self.start_time
                print(f"  - 持续时间: {duration:.1f}s")
            if self.fps_history:
                print(f"  - 平均FPS: {np.mean(self.fps_history):.1f}")

        cv2.destroyAllWindows()
        self.screen_capture.close()
        print("✓ 完成")


def main():
    """主入口"""
    # 使用本地models目录
    models_dir = Path(__file__).parent / "models"
    yolo_model = models_dir / "FunGen-12s-pov-1.1.0.pt"

    if not yolo_model.exists():
        print(f"错误: 模型未找到 {yolo_model}")
        print(f"请确保models/目录包含YOLO模型文件")
        return

    print("追踪模式:")
    print("  1. yolo_roi - YOLO辅助ROI (推荐)")
    print("  2. oscillation_experimental_2 - 振荡检测")
    print()

    choice = input("选择 (1-2) [默认: 1]: ").strip() or "1"
    mode = "yolo_roi" if choice == "1" else "oscillation_experimental_2"

    live = FunGenLiveScreen(
        tracker_model_path=str(yolo_model),
        tracker_mode=mode,
        target_fps=30,
        monitor_index=1,
        capture_scale=0.5
    )

    live.run()


if __name__ == "__main__":
    main()
