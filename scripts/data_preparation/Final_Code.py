# ============================================================================
# BLOCK 1/8 - IMPORTS AND CONFIGURATION CONSTANTS
# ============================================================================

import os
os.environ['YOLO_OFFLINE'] = '1'

import onnxruntime as ort
import cv2
import numpy as np
import time
import torch
import json
from ultralytics import YOLO
from collections import deque, Counter
from scipy import stats


PT_MODEL_PATH = r""
MODEL_PATH = r""
VIDEO_PATH = r""


EGO_VEHICLE = {
    'x_center': 0.42,
    'y_bottom': 0.91,
    'width': 0.79,
    'height': 0.11
}


DISTANCE_ZONES = {
    'critical': 5.0,
    'warning': 15.0,
    'caution': 30.0
}


STOP_DISTANCE_THRESHOLD = 3.0

TTC_THRESHOLDS = {
    'critical': 2.0,
    'warning': 5.0,
    'caution': 10.0
}


BRAKING_SPEED_CHANGE_THRESHOLD = -8.0
ACCELERATION_SPEED_CHANGE_THRESHOLD = 8.0
STOPPED_SPEED_THRESHOLD = 5.0
STATIONARY_RELATIVE_SPEED = 0.5


PEDESTRIAN_WARNING_DISTANCE = 15.0


TTC_MIN_CONFIDENCE = 0.4
TTC_HIGH_CONFIDENCE = 0.5


DISTANCE_TREND_FAST_THRESHOLD = -1.5
DISTANCE_TREND_SLOW_THRESHOLD = -0.3
DISTANCE_TREND_INCREASING_THRESHOLD = 0.3

LATERAL_MOVEMENT_THRESHOLD = 50
SIDE_ENTRY_GRACE_FRAMES = 15
SIDE_ENTRY_CONFIDENCE_FRAMES = 8
MIN_FRAMES_FOR_CRITICAL = 10

BBOX_OVERLAP_THRESHOLD = 0.3
LARGE_OBJECT_WIDTH_RATIO = 0.4

LANE_CHANGE_SPEED_THRESHOLD = 2.5

CRITICAL_ZONE_SIDE_MARGIN = 0.15

HIGH_PRIORITY_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'pedestrian']
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
PERSON_CLASSES = ['person', 'pedestrian', 'rider']

CLASS_PRIORITY = {
    'person': 1, 'pedestrian': 1, 'rider': 1,
    'bicycle': 2, 'motorcycle': 2, 'bike': 2,
    'car': 3, 'auto': 3, 'rickshaw': 3, 'van': 3,
    'truck': 4, 'bus': 4
}

REFERENCE_SIZES = {
    'car': (1.6, 1.8), 'truck': (2.5, 2.5), 'bus': (3.0, 2.5),
    'person': (1.7, 0.5), 'pedestrian': (1.7, 0.5), 'bicycle': (1.2, 0.6),
    'motorcycle': (1.3, 0.8), 'rider': (1.7, 0.5), 'auto': (1.8, 1.5), 
    'rickshaw': (1.8, 1.5)
}


# ============================================================================
# NEW: DECISION SMOOTHING CONFIGURATION
# ============================================================================

DECISION_SMOOTHING_PRESETS = {
    'OFF': {
        'enabled': False,
        'debounce_frames': 0,
        'name': 'OFF',
        'description': 'Raw (No Smoothing)'
    },
    'LIGHT': {
        'enabled': True,
        'debounce_frames': 3,
        'consensus_ratio': 0.5,
        'name': 'LIGHT',
        'description': 'Light Smoothing (3 frames)'
    },
    'MEDIUM': {
        'enabled': True,
        'debounce_frames': 5,
        'consensus_ratio': 0.6,
        'name': 'MEDIUM',
        'description': 'Medium Smoothing (5 frames)'
    },
    'STRONG': {
        'enabled': True,
        'debounce_frames': 8,
        'consensus_ratio': 0.65,
        'name': 'STRONG',
        'description': 'Strong Smoothing (8 frames)'
    },
    'VERY_STRONG': {
        'enabled': True,
        'debounce_frames': 12,
        'consensus_ratio': 0.7,
        'name': 'VERY_STRONG',
        'description': 'Very Strong (12 frames)'
    }
}

# Decision priority levels for smoothing logic
DECISION_PRIORITY_LEVELS = {
    'STOP': 5,
    'APPLY BRAKE': 4,
    'READY TO BRAKE': 3,
    'FOLLOW CAUTIOUSLY': 3,
    'SLOW DOWN': 2,
    'CONTINUE DRIVE': 1
}


DETECTION_MODES = {
    'CITY': {
        'name': 'CITY', 'img_size': 640, 'conf_threshold': 0.30, 
        'iou_threshold': 0.40, 'frame_skip': 0, 
        'description': 'High Accuracy', 'color': (255, 200, 0)
    },
    'HIGHWAY': {
        'name': 'HIGHWAY', 'img_size': 640, 'conf_threshold': 0.35, 
        'iou_threshold': 0.45, 'frame_skip': 0, 
        'description': 'High Speed', 'color': (0, 255, 255)
    },
    'AUTO': {
        'name': 'AUTO', 'img_size': 640, 'conf_threshold': 0.32, 
        'iou_threshold': 0.45, 'frame_skip': 0, 
        'description': 'Adaptive', 'color': (255, 0, 255)
    }
}


RESOLUTION_PRESETS = {
    'FHD': {'width': 1920, 'height': 1080, 'name': 'FHD (1920x1080)'},
    'ULTRA': {'width': 2048, 'height': 1080, 'name': 'ULTRA (2048x1080)'},
    'QHD': {'width': 2560, 'height': 1440, 'name': 'QHD (2560x1440)'}
}


DECISION_COLORS = {
    'STOP': (0, 0, 139),
    'APPLY BRAKE': (0, 0, 255),
    'READY TO BRAKE': (0, 100, 255),
    'FOLLOW CAUTIOUSLY': (0, 140, 255),
    'SLOW DOWN': (0, 215, 255),
    'CONTINUE DRIVE': (0, 255, 0)
}


DISPLAY_SCALE = 0.75
WINDOW_NAME = "ADAS Detection System"


TRACKER_MAX_DISAPPEARED = 5
TRACKER_IOU_THRESHOLD = 0.35
DUPLICATE_IOU_THRESHOLD = 0.6


SPEED_ESTIMATION_COUNTS = {
    'very_high': (8, 25.0),
    'high': (4, 45.0),
    'medium': (2, 60.0),
    'low': (0, 80.0)
}


MODE_SELECTION_DENSITY_HIGH = 8
MODE_SELECTION_SPEED_HIGH = 80
MODE_SELECTION_SPEED_LOW = 40
MODE_SELECTION_DENSITY_LOW = 3

# ============================================================================
# BLOCK 2/8 - TRACKED OBJECT CLASS
# ============================================================================

class TrackedObject:
    def __init__(self, obj_id, bbox, class_id, confidence, distance, timestamp):
        self.id = obj_id
        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence
        
        self.distance_history = deque(maxlen=15)
        self.timestamp_history = deque(maxlen=15)
        self.bbox_history = deque(maxlen=10)
        self.class_history = deque(maxlen=25)
        self.speed_history = deque(maxlen=10)
        
        self.distance_history.append(distance)
        self.timestamp_history.append(timestamp)
        self.bbox_history.append(bbox)
        self.class_history.append(class_id)
        
        self.relative_speed = 0.0
        self.object_speed = 0.0
        self.ttc = None
        self.ttc_confidence = 0.0
        
        self.distance_trend = "stable"
        self.distance_change_rate = 0.0
        self.object_behavior = "moving"
        
        self.frames_tracked = 1
        self.frames_since_update = 0
        self.last_seen_frame = 0
        self.class_stable = False
        
        self.lateral_movement = 0.0
        self.is_side_entry = False
        self.entry_side = "center"
        self.frames_in_path = 0
        self.lateral_speed = 0.0
        self.initial_x_position = (bbox[0] + bbox[2]) / 2
        
    def update(self, bbox, class_id, confidence, distance, timestamp, frame_num):
        self.bbox = bbox
        self.confidence = confidence
        self.frames_tracked += 1
        self.frames_since_update = 0
        self.last_seen_frame = frame_num
        
        self.class_history.append(class_id)
        self.class_id = self._get_stable_class()
        self.distance_history.append(distance)
        self.timestamp_history.append(timestamp)
        self.bbox_history.append(bbox)
    
    def _get_stable_class(self):
        if len(self.class_history) < 8:
            return self.class_history[-1]
        
        recent = list(self.class_history)[-20:]
        counts = Counter(recent)
        most_common, count = counts.most_common(1)[0]
        
        stability_ratio = count / len(recent)
        self.class_stable = stability_ratio > 0.75
        
        if not self.class_stable and self.frames_tracked > 15:
            return self.class_id
        
        return most_common
    
    def analyze_trends(self, ego_speed_kmh):
        if len(self.distance_history) < 7:
            return
        
        distances = np.array(list(self.distance_history)[-10:])
        times = np.array(list(self.timestamp_history)[-10:])
        times = times - times[0]
        
        if times[-1] < 0.2:
            return
        
        try:
            slope, _, r_val, _, _ = stats.linregress(times, distances)
            self.distance_change_rate = slope
            
            if slope < DISTANCE_TREND_FAST_THRESHOLD:
                self.distance_trend = "decreasing_fast"
            elif slope < DISTANCE_TREND_SLOW_THRESHOLD:
                self.distance_trend = "decreasing"
            elif slope > DISTANCE_TREND_INCREASING_THRESHOLD:
                self.distance_trend = "increasing"
            else:
                self.distance_trend = "stable"
            
            self.relative_speed = -slope
            ego_ms = ego_speed_kmh / 3.6
            obj_ms = ego_ms - self.relative_speed
            self.object_speed = obj_ms * 3.6
            self.speed_history.append(self.object_speed)
            
            if abs(self.relative_speed) < STATIONARY_RELATIVE_SPEED and self.object_speed < STOPPED_SPEED_THRESHOLD:
                self.object_behavior = "stopped"
            elif len(self.speed_history) >= 7:
                speeds = list(self.speed_history)
                speed_change = speeds[-1] - speeds[0]
                if speed_change < BRAKING_SPEED_CHANGE_THRESHOLD:
                    self.object_behavior = "braking"
                elif speed_change > ACCELERATION_SPEED_CHANGE_THRESHOLD:
                    self.object_behavior = "accelerating"
                else:
                    self.object_behavior = "moving"
            
            self.ttc_confidence = min(r_val ** 2, 1.0) * min(len(distances) / 10.0, 1.0)
            
        except:
            self.distance_trend = "stable"
            self.object_behavior = "moving"
    
    def analyze_lateral_movement(self, img_width):
        if len(self.bbox_history) < 3:
            return
        
        current_bbox = self.bbox_history[-1]
        previous_bbox = self.bbox_history[-3] if len(self.bbox_history) >= 3 else self.bbox_history[0]
        
        current_x = (current_bbox[0] + current_bbox[2]) / 2
        previous_x = (previous_bbox[0] + previous_bbox[2]) / 2
        
        self.lateral_movement = abs(current_x - previous_x)
        
        if len(self.bbox_history) >= 3:
            time_diff = len(self.bbox_history) * 0.033
            self.lateral_speed = abs(current_x - previous_x) / time_diff if time_diff > 0 else 0
        
        center_x = img_width / 2
        left_threshold = img_width * 0.3
        right_threshold = img_width * 0.7
        
        if self.frames_tracked <= SIDE_ENTRY_GRACE_FRAMES:
            if self.initial_x_position < left_threshold:
                self.entry_side = "left"
                self.is_side_entry = True
            elif self.initial_x_position > right_threshold:
                self.entry_side = "right"
                self.is_side_entry = True
            else:
                self.entry_side = "center"
                self.is_side_entry = False
        
        if self.lateral_movement > LATERAL_MOVEMENT_THRESHOLD:
            self.object_behavior = "lane_changing"
    
    def is_stable_threat(self):
        if self.is_side_entry and self.frames_in_path < SIDE_ENTRY_CONFIDENCE_FRAMES:
            return False
        
        if self.lateral_speed > LANE_CHANGE_SPEED_THRESHOLD:
            return False
        
        if self.frames_tracked < MIN_FRAMES_FOR_CRITICAL and self.is_side_entry:
            return False
        
        return True
    
    def get_threat_confidence(self):
        confidence = 1.0
        
        if self.is_side_entry:
            confidence *= 0.5
        
        if self.lateral_speed > LANE_CHANGE_SPEED_THRESHOLD:
            confidence *= 0.3
        
        if self.frames_in_path < SIDE_ENTRY_CONFIDENCE_FRAMES:
            confidence *= (self.frames_in_path / SIDE_ENTRY_CONFIDENCE_FRAMES)
        
        if self.object_behavior == "lane_changing":
            confidence *= 0.4
        
        return confidence
    
    def calculate_ttc(self, ego_speed_kmh):
        dist = list(self.distance_history)[-1] if self.distance_history else None
        if not dist or self.ttc_confidence < TTC_MIN_CONFIDENCE:
            return None
        if self.relative_speed < STATIONARY_RELATIVE_SPEED:
            return float('inf')
        ttc = dist / self.relative_speed
        return ttc if 0 < ttc < 60 else None
    
    def get_centroid(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def is_valid(self, max_frames=5):
        return self.frames_since_update < max_frames


# ============================================================================
# OBJECT TRACKER CLASS
# ============================================================================

class ObjectTracker:
    def __init__(self, max_disappeared=TRACKER_MAX_DISAPPEARED, iou_threshold=TRACKER_IOU_THRESHOLD):
        self.next_id = 0
        self.objects = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.class_names = []
    
    def set_class_names(self, names):
        self.class_names = names
    
    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter
        return inter / union if union > 0 else 0
    
    def classes_compatible(self, c1, c2):
        if c1 == c2:
            return True
        if not self.class_names:
            return False
        
        n1 = self.class_names[c1].lower() if c1 < len(self.class_names) else ""
        n2 = self.class_names[c2].lower() if c2 < len(self.class_names) else ""
        
        groups = [
            {'car', 'truck', 'bus', 'van'},
            {'motorcycle', 'bicycle', 'bike'},
            {'person', 'pedestrian', 'rider'},
            {'auto', 'rickshaw'}
        ]
        
        for g in groups:
            if n1 in g and n2 in g:
                return True
        return False
    
    def match_objects(self, boxes, classes, scores, distances, timestamp, frame_num):
        for obj_id in list(self.objects.keys()):
            self.objects[obj_id].frames_since_update += 1
        
        if len(self.objects) == 0:
            for bbox, cls, score, dist in zip(boxes, classes, scores, distances):
                obj = TrackedObject(self.next_id, bbox, cls, score, dist, timestamp)
                obj.last_seen_frame = frame_num
                self.objects[self.next_id] = obj
                self.next_id += 1
            return list(self.objects.values())
        
        if len(boxes) == 0:
            self.objects = {k: v for k, v in self.objects.items() if v.is_valid(self.max_disappeared)}
            return list(self.objects.values())
        
        used_new, used_old = set(), set()
        
        for obj_id, obj in self.objects.items():
            best_score, best_idx = 0, -1
            for idx, (bbox, cls) in enumerate(zip(boxes, classes)):
                if idx in used_new:
                    continue
                
                iou = self.calculate_iou(obj.bbox, bbox)
                
                if obj.class_id == cls:
                    class_bonus = 1.4
                elif self.classes_compatible(obj.class_id, cls):
                    class_bonus = 0.7
                else:
                    class_bonus = 0.2
                
                score = iou * class_bonus
                
                if score > self.iou_threshold and score > best_score:
                    best_score, best_idx = score, idx
            
            if best_idx >= 0:
                obj.update(boxes[best_idx], classes[best_idx], scores[best_idx], 
                          distances[best_idx], timestamp, frame_num)
                used_new.add(best_idx)
                used_old.add(obj_id)
        
        for idx in range(len(boxes)):
            if idx not in used_new:
                obj = TrackedObject(self.next_id, boxes[idx], classes[idx], 
                                   scores[idx], distances[idx], timestamp)
                obj.last_seen_frame = frame_num
                self.objects[self.next_id] = obj
                self.next_id += 1
        
        self.objects = {k: v for k, v in self.objects.items() if v.is_valid(self.max_disappeared)}
        
        return list(self.objects.values())
    
    def get_all_objects(self):
        return list(self.objects.values())
    
    def reset(self):
        self.objects = {}
        self.next_id = 0

# ============================================================================
# BLOCK 3/8 - DECISION SMOOTHER CLASS (NEW!) + UPDATED SAFETY CONFIG
# ============================================================================

class DecisionSmoother:
    """
    Handles decision smoothing with configurable debouncing to reduce fluctuations
    while maintaining real-time detection accuracy
    """
    def __init__(self, preset='MEDIUM'):
        self.preset = preset
        self.config = DECISION_SMOOTHING_PRESETS[preset].copy()
        
        # History tracking
        self.decision_history = deque(maxlen=20)
        self.priority_history = deque(maxlen=20)
        self.threats_history = deque(maxlen=20)
        
        # Current state
        self.raw_decision = "CONTINUE DRIVE"
        self.raw_priority = 1
        self.raw_threats = []
        
        self.smoothed_decision = "CONTINUE DRIVE"
        self.smoothed_priority = 1
        self.smoothed_threats = []
        
        # Statistics
        self.total_decisions = 0
        self.smoothing_applied_count = 0
        self.last_change_frame = 0
        
    def set_preset(self, preset):
        """Change smoothing preset"""
        if preset in DECISION_SMOOTHING_PRESETS:
            self.preset = preset
            self.config = DECISION_SMOOTHING_PRESETS[preset].copy()
            print(f"✓ Smoothing: {self.config['name']} - {self.config['description']}")
    
    def update(self, decision, priority, threats, frame_num):
        """
        Update with new raw decision and return smoothed decision
        
        Args:
            decision: Raw decision string
            priority: Raw priority level
            threats: List of threat objects
            frame_num: Current frame number
        
        Returns:
            tuple: (smoothed_decision, smoothed_priority, smoothed_threats, decision_changed)
        """
        # Store raw values
        self.raw_decision = decision
        self.raw_priority = priority
        self.raw_threats = threats
        
        # Add to history
        self.decision_history.append(decision)
        self.priority_history.append(priority)
        self.threats_history.append(threats)
        
        self.total_decisions += 1
        
        # If smoothing is disabled, return raw values
        if not self.config['enabled']:
            self.smoothed_decision = decision
            self.smoothed_priority = priority
            self.smoothed_threats = threats
            return decision, priority, threats, False
        
        # Apply smoothing logic
        smoothed_dec, smoothed_pri, smoothed_thr, changed = self._apply_smoothing(frame_num)
        
        if changed:
            self.last_change_frame = frame_num
            self.smoothing_applied_count += 1
        
        self.smoothed_decision = smoothed_dec
        self.smoothed_priority = smoothed_pri
        self.smoothed_threats = smoothed_thr
        
        return smoothed_dec, smoothed_pri, smoothed_thr, changed
    
    def _apply_smoothing(self, frame_num):
        """
        Apply smoothing algorithm based on decision history
        
        Strategy:
        1. For critical decisions (STOP, APPLY BRAKE), reduce debounce for safety
        2. For lower priority decisions, apply full debouncing
        3. Use consensus voting from recent history
        4. Prioritize higher severity decisions in case of tie
        """
        debounce_frames = self.config['debounce_frames']
        consensus_ratio = self.config.get('consensus_ratio', 0.6)
        
        # Not enough history yet
        if len(self.decision_history) < min(3, debounce_frames):
            return self.raw_decision, self.raw_priority, self.raw_threats, False
        
        # Get recent history (up to debounce_frames)
        recent_decisions = list(self.decision_history)[-debounce_frames:]
        recent_priorities = list(self.priority_history)[-debounce_frames:]
        recent_threats = list(self.threats_history)[-debounce_frames:]
        
        # Check if we have a critical situation (priority >= 4)
        has_critical = any(p >= 4 for p in recent_priorities)
        
        # Reduce debouncing for critical situations (faster response)
        if has_critical:
            effective_frames = max(2, debounce_frames // 2)
            recent_decisions = recent_decisions[-effective_frames:]
            recent_priorities = recent_priorities[-effective_frames:]
            recent_threats = recent_threats[-effective_frames:]
        
        # Count decision occurrences
        decision_counts = Counter(recent_decisions)
        
        # Check for consensus
        most_common_decision, count = decision_counts.most_common(1)[0]
        consensus_met = count / len(recent_decisions) >= consensus_ratio
        
        # If consensus is met, use the consensus decision
        if consensus_met:
            # Find the highest priority associated with this decision
            matching_priorities = [p for d, p in zip(recent_decisions, recent_priorities) 
                                  if d == most_common_decision]
            consensus_priority = max(matching_priorities) if matching_priorities else self.raw_priority
            
            # Get most recent threats for this decision
            consensus_threats = self.raw_threats
            for i in range(len(recent_decisions) - 1, -1, -1):
                if recent_decisions[i] == most_common_decision:
                    consensus_threats = recent_threats[i]
                    break
            
            changed = (most_common_decision != self.smoothed_decision)
            return most_common_decision, consensus_priority, consensus_threats, changed
        
        else:
            # No clear consensus - prioritize safety (higher priority decision)
            max_priority_idx = recent_priorities.index(max(recent_priorities))
            safety_decision = recent_decisions[max_priority_idx]
            safety_priority = recent_priorities[max_priority_idx]
            safety_threats = recent_threats[max_priority_idx]
            
            changed = (safety_decision != self.smoothed_decision)
            return safety_decision, safety_priority, safety_threats, changed
    
    def get_smoothing_stats(self):
        """Return statistics about smoothing performance"""
        if self.total_decisions == 0:
            return {
                'total': 0,
                'smoothed': 0,
                'ratio': 0.0,
                'preset': self.preset
            }
        
        return {
            'total': self.total_decisions,
            'smoothed': self.smoothing_applied_count,
            'ratio': self.smoothing_applied_count / self.total_decisions,
            'preset': self.preset,
            'enabled': self.config['enabled'],
            'debounce_frames': self.config['debounce_frames']
        }
    
    def reset(self):
        """Reset all history and statistics"""
        self.decision_history.clear()
        self.priority_history.clear()
        self.threats_history.clear()
        self.total_decisions = 0
        self.smoothing_applied_count = 0
        self.last_change_frame = 0


# ============================================================================
# UPDATED SAFETY CONFIG CLASS (with smoothing controls)
# ============================================================================

class SafetyConfig:
    def __init__(self):
        self.trapezoid_area = True
        self.speed_adaptive = True
        self.distance_estimation = True
        self.use_onnx = False
        self.show_safety_overlay = True
        self.trapezoid_gradient = True
        
        self.show_stats_panel = True
        self.show_safety_panel = True
        self.show_mode_panel = True
        
        self.paused = False
        self.recording = False
        self.record_start_frame = 0
        self.record_end_frame = 0
        
        self.manual_speed_mode = True
        self.current_speed = 40.0
        self.speed_step = 5.0
        self.speed_history = deque(maxlen=10)
        self.min_speed = 0.0
        self.max_speed = 200.0
        
        self.camera_focal_length = 700
        self.camera_height = 1.2
        self.resolution_mode = 'FHD'
        
        # NEW: Decision smoothing controls
        self.show_both_decisions = True  # Show both raw and smoothed
        self.smoothing_preset = 'MEDIUM'  # Current smoothing preset
        self.decision_smoother = DecisionSmoother(self.smoothing_preset)
    
    def toggle_trapezoid(self):
        self.trapezoid_area = not self.trapezoid_area
        print(f"✓ Trapezoid: {'ON' if self.trapezoid_area else 'OFF'}")
    
    def toggle_speed_adaptive(self):
        self.speed_adaptive = not self.speed_adaptive
        print(f"✓ Speed Adaptive: {'ON' if self.speed_adaptive else 'OFF'}")
    
    def toggle_distance_estimation(self):
        self.distance_estimation = not self.distance_estimation
        print(f"✓ Distance: {'ON' if self.distance_estimation else 'OFF'}")
    
    def toggle_safety_overlay(self):
        self.show_safety_overlay = not self.show_safety_overlay
        print(f"✓ Overlay: {'ON' if self.show_safety_overlay else 'OFF'}")
    
    def toggle_stats_panel(self):
        self.show_stats_panel = not self.show_stats_panel
        print(f"✓ Stats Panel: {'ON' if self.show_stats_panel else 'OFF'}")
    
    def toggle_safety_panel(self):
        self.show_safety_panel = not self.show_safety_panel
        print(f"✓ Safety Panel: {'ON' if self.show_safety_panel else 'OFF'}")
    
    def toggle_mode_panel(self):
        self.show_mode_panel = not self.show_mode_panel
        print(f"✓ Mode Panel: {'ON' if self.show_mode_panel else 'OFF'}")
    
    def toggle_model(self):
        self.use_onnx = not self.use_onnx
        print(f"✓ Model: {'ONNX' if self.use_onnx else 'PT'}")
    
    def toggle_pause(self):
        self.paused = not self.paused
        print(f"✓ Video: {'PAUSED' if self.paused else 'PLAYING'}")
    
    def toggle_speed_mode(self):
        self.manual_speed_mode = not self.manual_speed_mode
        print(f"✓ Speed Mode: {'MANUAL' if self.manual_speed_mode else 'AUTO'}")
    
    # NEW: Smoothing control methods
    def toggle_both_decisions(self):
        """Toggle between showing both decisions or only smoothed"""
        self.show_both_decisions = not self.show_both_decisions
        mode = "BOTH (Raw + Smoothed)" if self.show_both_decisions else "SMOOTHED ONLY"
        print(f"✓ Display Mode: {mode}")
    
    def cycle_smoothing_preset(self):
        """Cycle through smoothing presets"""
        presets = list(DECISION_SMOOTHING_PRESETS.keys())
        current_idx = presets.index(self.smoothing_preset)
        next_idx = (current_idx + 1) % len(presets)
        self.smoothing_preset = presets[next_idx]
        self.decision_smoother.set_preset(self.smoothing_preset)
    
    def increase_debounce(self):
        """Increase debounce frames by 1"""
        if self.decision_smoother.config['enabled']:
            current = self.decision_smoother.config['debounce_frames']
            self.decision_smoother.config['debounce_frames'] = min(current + 1, 20)
            print(f"✓ Debounce Frames: {self.decision_smoother.config['debounce_frames']}")
    
    def decrease_debounce(self):
        """Decrease debounce frames by 1"""
        if self.decision_smoother.config['enabled']:
            current = self.decision_smoother.config['debounce_frames']
            self.decision_smoother.config['debounce_frames'] = max(current - 1, 1)
            print(f"✓ Debounce Frames: {self.decision_smoother.config['debounce_frames']}")
    
    def start_recording(self, frame_num):
        if not self.recording:
            self.recording = True
            self.record_start_frame = frame_num
            print(f"🔴 RECORDING STARTED at frame {frame_num}")
    
    def stop_recording(self, frame_num):
        if self.recording:
            self.recording = False
            self.record_end_frame = frame_num
            print(f"⏹ RECORDING STOPPED at frame {frame_num}")
            print(f"📹 Recorded frames: {self.record_start_frame} to {self.record_end_frame}")
    
    def increase_speed(self):
        if self.manual_speed_mode:
            self.current_speed = min(self.current_speed + self.speed_step, self.max_speed)
            print(f"✓ Speed: {self.current_speed:.1f} km/h")
    
    def decrease_speed(self):
        if self.manual_speed_mode:
            self.current_speed = max(self.current_speed - self.speed_step, self.min_speed)
            print(f"✓ Speed: {self.current_speed:.1f} km/h")
    
    def cycle_resolution(self):
        modes = list(RESOLUTION_PRESETS.keys())
        idx = modes.index(self.resolution_mode)
        self.resolution_mode = modes[(idx + 1) % len(modes)]
        print(f"✓ Resolution: {RESOLUTION_PRESETS[self.resolution_mode]['name']}")
    
    def estimate_speed(self, count):
        if self.manual_speed_mode:
            return
        
        for key, (threshold, speed) in SPEED_ESTIMATION_COUNTS.items():
            if count > threshold:
                est = speed
                break
        else:
            est = SPEED_ESTIMATION_COUNTS['low'][1]
        
        self.speed_history.append(est)
        self.current_speed = np.clip(np.mean(self.speed_history), self.min_speed, self.max_speed)
    
    def get_adjusted_zones(self, detection_mode='AUTO'):
        if not self.speed_adaptive:
            return DISTANCE_ZONES
        
        speed_mult = 1 + ((self.current_speed - 40) / 100)
        
        if detection_mode == 'HIGHWAY':
            mode_mult = 1.3
        elif detection_mode == 'CITY':
            mode_mult = 0.8
        else:
            mode_mult = 1.0
        
        total_mult = speed_mult * mode_mult
        
        return {
            'critical': np.clip(DISTANCE_ZONES['critical'] * total_mult, 3.0, 10.0),
            'warning': np.clip(DISTANCE_ZONES['warning'] * total_mult, 10.0, 30.0),
            'caution': np.clip(DISTANCE_ZONES['caution'] * total_mult, 20.0, 50.0)
        }

# ============================================================================
# BLOCK 4/8 - AUTO MODE SELECTOR, ADAPTIVE DETECTOR, TRAPEZOID CONFIG
# ============================================================================

class AutoModeSelector:
    def __init__(self):
        self.enabled = True
        self.speed_history = deque(maxlen=30)
        self.detection_history = deque(maxlen=30)
        self.current_mode = 'AUTO'
        self.reason = 'Initial'
    
    def select_mode(self, speed, count):
        if not self.enabled:
            return self.current_mode, self.reason
        
        self.speed_history.append(speed)
        self.detection_history.append(count)
        
        avg_speed = np.mean(self.speed_history) if len(self.speed_history) > 5 else speed
        avg_det = np.mean(self.detection_history) if len(self.detection_history) > 5 else count
        
        if avg_det > MODE_SELECTION_DENSITY_HIGH:
            self.current_mode, self.reason = 'CITY', f'High density ({int(avg_det)} obj)'
        elif avg_speed > MODE_SELECTION_SPEED_HIGH and avg_det < MODE_SELECTION_DENSITY_LOW:
            self.current_mode, self.reason = 'HIGHWAY', f'High speed ({avg_speed:.0f} km/h)'
        elif avg_speed < MODE_SELECTION_SPEED_LOW:
            self.current_mode, self.reason = 'CITY', f'Low speed ({avg_speed:.0f} km/h)'
        else:
            self.current_mode, self.reason = 'AUTO', f'Medium ({avg_speed:.0f} km/h)'
        
        return self.current_mode, self.reason


class AdaptiveDetector:
    def __init__(self, initial_mode='AUTO'):
        self.current_mode = initial_mode
        self.manual_override = False
        self.auto_selector = AutoModeSelector()
    
    def get_current_params(self):
        return DETECTION_MODES[self.current_mode]
    
    def set_mode(self, mode, manual=True):
        if mode in DETECTION_MODES:
            self.current_mode = mode
            self.manual_override = manual
            self.auto_selector.enabled = not manual
            print(f"✓ Mode: {mode} ({'Manual' if manual else 'Auto'})")
    
    def update_auto_mode(self, speed, count):
        if not self.manual_override:
            new_mode, reason = self.auto_selector.select_mode(speed, count)
            if new_mode != self.current_mode:
                self.set_mode(new_mode, manual=False)
        return self.auto_selector.reason
    
    def enable_auto_mode(self):
        self.manual_override = False
        self.auto_selector.enabled = True
        print("✓ Auto mode ENABLED")


class TrapezoidConfig:
    def __init__(self):
        self.config_file = "trapezoid_config.json"
        self.step = 0.01
        
        self.horizon = 0.6
        self.vp_x = 0.47
        self.bottom_width = 0.35
        self.top_width_ratio = 0.36
        self.left_offset = -0.22
        self.right_offset = 0.22
        
        self.load_config()
    
    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.horizon = config.get('horizon', self.horizon)
                    self.vp_x = config.get('vp_x', self.vp_x)
                    self.bottom_width = config.get('bottom_width', self.bottom_width)
                    self.top_width_ratio = config.get('top_width_ratio', self.top_width_ratio)
                    self.left_offset = config.get('left_offset', self.left_offset)
                    self.right_offset = config.get('right_offset', self.right_offset)
                print(f"✓ Loaded trapezoid config from {self.config_file}")
            except Exception as e:
                print(f"⚠ Could not load config: {e}")
    
    def save_config(self):
        config = {
            'horizon': round(self.horizon, 3),
            'vp_x': round(self.vp_x, 3),
            'bottom_width': round(self.bottom_width, 3),
            'top_width_ratio': round(self.top_width_ratio, 3),
            'left_offset': round(self.left_offset, 3),
            'right_offset': round(self.right_offset, 3),
            'saved_at': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"\n✓ TRAPEZOID CONFIG SAVED to {os.path.abspath(self.config_file)}\n")
        except Exception as e:
            print(f"✗ Failed to save config: {e}")
    
    def reset_to_defaults(self):
        self.horizon = 0.35
        self.vp_x = 0.42
        self.bottom_width = 0.35
        self.top_width_ratio = 0.18
        self.left_offset = 0.0
        self.right_offset = 0.0
        print("✓ Reset to default trapezoid values")
    
    def adjust_horizon_up(self):
        self.horizon = max(0.1, self.horizon - self.step)
        print(f"✓ Horizon: {self.horizon:.3f}")
    
    def adjust_horizon_down(self):
        self.horizon = min(0.6, self.horizon + self.step)
        print(f"✓ Horizon: {self.horizon:.3f}")
    
    def adjust_vp_left(self):
        self.vp_x = max(0.2, self.vp_x - self.step)
        print(f"✓ VP X: {self.vp_x:.3f}")
    
    def adjust_vp_right(self):
        self.vp_x = min(0.8, self.vp_x + self.step)
        print(f"✓ VP X: {self.vp_x:.3f}")
    
    def adjust_width_narrow(self):
        self.bottom_width = max(0.2, self.bottom_width - self.step * 2)
        print(f"✓ Bottom Width: {self.bottom_width:.3f}")
    
    def adjust_width_wide(self):
        self.bottom_width = min(0.9, self.bottom_width + self.step * 2)
        print(f"✓ Bottom Width: {self.bottom_width:.3f}")
    
    def adjust_top_narrow(self):
        self.top_width_ratio = max(0.05, self.top_width_ratio - 0.01)
        print(f"✓ Top Width Ratio: {self.top_width_ratio:.3f}")
    
    def adjust_top_wide(self):
        self.top_width_ratio = min(0.5, self.top_width_ratio + 0.01)
        print(f"✓ Top Width Ratio: {self.top_width_ratio:.3f}")
    
    def adjust_left_angle_in(self):
        self.left_offset += self.step
        print(f"✓ Left Offset: {self.left_offset:.3f}")
    
    def adjust_left_angle_out(self):
        self.left_offset -= self.step
        print(f"✓ Left Offset: {self.left_offset:.3f}")
    
    def adjust_right_angle_in(self):
        self.right_offset -= self.step
        print(f"✓ Right Offset: {self.right_offset:.3f}")
    
    def adjust_right_angle_out(self):
        self.right_offset += self.step
        print(f"✓ Right Offset: {self.right_offset:.3f}")


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

safety_config = SafetyConfig()
trapezoid_config = TrapezoidConfig()


# ============================================================================
# HELPER FUNCTIONS - PREPROCESSING & INFERENCE
# ============================================================================

def preprocess_frame(frame, img_size):
    input_frame = cv2.resize(frame, (img_size, img_size))
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_frame = input_frame.astype('float32') / 255.0
    input_frame = np.transpose(input_frame, (2, 0, 1))
    return np.expand_dims(input_frame, axis=0)


def scale_boxes(img_shape, boxes, img_size):
    if len(boxes) == 0:
        return boxes
    scaled_boxes = boxes.copy()
    scaled_boxes[:, 0] *= img_shape[1] / img_size
    scaled_boxes[:, 1] *= img_shape[0] / img_size
    scaled_boxes[:, 2] *= img_shape[1] / img_size
    scaled_boxes[:, 3] *= img_shape[0] / img_size
    x1 = scaled_boxes[:, 0] - scaled_boxes[:, 2] / 2
    y1 = scaled_boxes[:, 1] - scaled_boxes[:, 3] / 2
    x2 = scaled_boxes[:, 0] + scaled_boxes[:, 2] / 2
    y2 = scaled_boxes[:, 1] + scaled_boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def postprocess(outputs, img_shape, conf_thres, iou_thres, img_size):
    predictions = np.transpose(outputs[0], (0, 2, 1))[0]
    boxes, class_scores = predictions[:, :4], predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)
    valid = confidences > conf_thres
    if not np.any(valid):
        return [], [], []
    boxes = scale_boxes(img_shape, boxes[valid], img_size)
    confidences = confidences[valid]
    class_ids = class_ids[valid]
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_thres, iou_thres)
    if len(indices) > 0:
        indices = indices.flatten()
        return boxes[indices], class_ids[indices], confidences[indices]
    return [], [], []


def run_inference_pt(frame, img_size, conf_thres, iou_thres):
    results = pt_model.predict(frame, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=False, device=DEVICE)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
    classes = results[0].boxes.cls.cpu().numpy().astype(int) if len(results[0].boxes) > 0 else np.array([])
    scores = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
    return boxes, classes, scores


def run_inference_onnx(frame, img_size, conf_thres, iou_thres):
    input_frame = preprocess_frame(frame, img_size)
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_frame})
    return postprocess(outputs, frame.shape, conf_thres, iou_thres, img_size)


def calculate_iou_simple(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def remove_duplicate_detections(boxes, classes, scores):
    if len(boxes) == 0:
        return boxes, classes, scores
    
    keep, removed = [], set()
    
    for i in range(len(boxes)):
        if i in removed:
            continue
        
        cls_i = CLASS_NAMES[classes[i]].lower() if classes[i] < len(CLASS_NAMES) else "unknown"
        is_person_i = any(p in cls_i for p in PERSON_CLASSES)
        is_vehicle_i = any(v in cls_i for v in VEHICLE_CLASSES)
        
        should_keep = True
        
        for j in range(len(boxes)):
            if i == j or j in removed:
                continue
            
            cls_j = CLASS_NAMES[classes[j]].lower() if classes[j] < len(CLASS_NAMES) else "unknown"
            is_person_j = any(p in cls_j for p in PERSON_CLASSES)
            is_vehicle_j = any(v in cls_j for v in VEHICLE_CLASSES)
            
            iou = calculate_iou_simple(boxes[i], boxes[j])
            
            if iou > DUPLICATE_IOU_THRESHOLD:
                if is_person_i and is_vehicle_j:
                    should_keep = False
                    removed.add(i)
                    break
                elif is_vehicle_i and is_person_j:
                    removed.add(j)
        
        if should_keep:
            keep.append(i)
    
    if len(keep) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return boxes[keep], classes[keep], scores[keep]


def estimate_distance(bbox, img_shape, class_name="car"):
    if not safety_config.distance_estimation:
        return None
    
    x1, y1, x2, y2 = bbox
    bh, bw = y2 - y1, x2 - x1
    
    if bh <= 10:
        return None
    
    ref_h, ref_w = REFERENCE_SIZES.get(class_name.lower(), (1.6, 1.8))
    
    dh = (ref_h * safety_config.camera_focal_length) / bh
    dw = (ref_w * safety_config.camera_focal_length) / bw
    
    distance = np.clip((dh * 0.7 + dw * 0.3), 1.0, 100.0)
    
    h = img_shape[0]
    ego_y = int(h * EGO_VEHICLE['y_bottom'])
    
    if y2 > ego_y:
        distance = min(distance, 3.0)
    else:
        perspective_factor = (ego_y - y2) / (ego_y - h * trapezoid_config.horizon)
        perspective_factor = max(0.1, min(perspective_factor, 1.0))
        distance = distance * (0.5 + 0.5 * perspective_factor)
    
    return max(1.0, distance)

# ============================================================================
# BLOCK 5/8 - GEOMETRY & PATH FUNCTIONS + ADAS DECISION LOGIC
# ============================================================================

def get_ego_vehicle_box(img_shape):
    h, w = img_shape[:2]
    ego_width = int(w * EGO_VEHICLE['width'])
    ego_height = int(h * EGO_VEHICLE['height'])
    ego_x_center = int(w * EGO_VEHICLE['x_center'])
    ego_y_hood = int(h * EGO_VEHICLE['y_bottom'])
    y1 = ego_y_hood - ego_height // 2
    y2 = ego_y_hood + ego_height // 2
    x1 = ego_x_center - ego_width // 2
    x2 = ego_x_center + ego_width // 2
    return [x1, y1, x2, y2]


def get_trapezoid_safety_area(img_shape):
    h, w = img_shape[:2]
    
    bottom_y = int(h * 0.855)
    ego_center_x = EGO_VEHICLE['x_center']
    
    bottom_width = trapezoid_config.bottom_width
    
    bl_x = int(w * (ego_center_x - bottom_width / 2)) + int(w * trapezoid_config.left_offset)
    br_x = int(w * (ego_center_x + bottom_width / 2)) + int(w * trapezoid_config.right_offset)
    
    top_y = int(h * trapezoid_config.horizon)
    vanishing_x = int(w * trapezoid_config.vp_x)
    top_width = bottom_width * trapezoid_config.top_width_ratio
    
    tl_x = int(vanishing_x - w * top_width / 2)
    tr_x = int(vanishing_x + w * top_width / 2)
    
    return np.array([
        [bl_x, bottom_y],
        [tl_x, top_y],
        [tr_x, top_y],
        [br_x, bottom_y]
    ], dtype=np.int32)


def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def is_in_path(box, img_shape, allow_partial=False):
    x1, y1, x2, y2 = box
    h, w = img_shape[:2]
    
    if not safety_config.trapezoid_area:
        return True, 1.0
    
    if y2 < h * 0.3:
        return False, 0.0
    
    trap = get_trapezoid_safety_area(img_shape)
    
    cx = (x1 + x2) / 2
    cy = y2
    
    center_in = is_point_in_polygon((cx, cy), trap)
    
    if center_in:
        return True, 1.0
    
    if allow_partial:
        bbox_width = x2 - x1
        
        if bbox_width > w * LARGE_OBJECT_WIDTH_RATIO:
            bottom_left = (x1, y2)
            bottom_right = (x2, y2)
            bottom_center_left = (x1 + bbox_width * 0.25, y2)
            bottom_center_right = (x2 - bbox_width * 0.25, y2)
            
            points_in = sum([
                is_point_in_polygon(bottom_left, trap),
                is_point_in_polygon(bottom_right, trap),
                is_point_in_polygon(bottom_center_left, trap),
                is_point_in_polygon(bottom_center_right, trap)
            ])
            
            if points_in >= 2:
                overlap_ratio = points_in / 4.0
                return True, overlap_ratio
        
        num_samples = 5
        points_to_check = []
        for i in range(num_samples):
            sample_x = x1 + (x2 - x1) * (i / (num_samples - 1))
            points_to_check.append((sample_x, y2))
        
        points_in = sum([is_point_in_polygon(pt, trap) for pt in points_to_check])
        overlap_ratio = points_in / num_samples
        
        if overlap_ratio >= BBOX_OVERLAP_THRESHOLD:
            return True, overlap_ratio
    
    return False, 0.0


def get_class_priority(class_name):
    return CLASS_PRIORITY.get(class_name.lower(), 3)


def format_ttc(ttc):
    if ttc is None:
        return "N/A"
    elif ttc == float('inf'):
        return "∞"
    elif ttc > 60:
        return ">60s"
    else:
        return f"{ttc:.1f}s"


def resize_frame_to_resolution(frame):
    res = RESOLUTION_PRESETS[safety_config.resolution_mode]
    return cv2.resize(frame, (res['width'], res['height']))


# ============================================================================
# ADAS DECISION LOGIC (Returns RAW decision for smoother)
# ============================================================================

def make_adas_decision(tracked_objects, img_shape, detection_mode):
    """
    Make ADAS decision based on tracked objects
    Returns RAW decision, priority, and threats for smoothing
    """
    if len(tracked_objects) == 0:
        return "CONTINUE DRIVE", 1, []
    
    zones = safety_config.get_adjusted_zones(detection_mode)
    decision = "CONTINUE DRIVE"
    priority = 1
    threats = []
    
    h, w = img_shape[:2]
    
    for obj in tracked_objects:
        cname = CLASS_NAMES[obj.class_id] if obj.class_id < len(CLASS_NAMES) else "unknown"
        
        if not any(p.lower() in cname.lower() for p in HIGH_PRIORITY_CLASSES):
            continue
        
        in_path, overlap_ratio = is_in_path(obj.bbox, img_shape, allow_partial=True)
        
        if in_path:
            obj.frames_in_path += 1
        else:
            obj.frames_in_path = 0
            continue
        
        if len(obj.distance_history) < 5:
            continue
        
        obj.analyze_trends(safety_config.current_speed)
        obj.analyze_lateral_movement(w)
        
        distance = list(obj.distance_history)[-1]
        class_priority_val = get_class_priority(cname)
        ttc = obj.calculate_ttc(safety_config.current_speed)
        
        is_person = any(p in cname.lower() for p in ['person', 'pedestrian', 'rider'])
        
        threat_confidence = obj.get_threat_confidence()
        is_stable = obj.is_stable_threat()
        
        threat_info = {
            'class': cname,
            'distance': distance,
            'distance_str': f"{distance:.1f}m",
            'object_speed': obj.object_speed,
            'relative_speed': obj.relative_speed * 3.6,
            'trend': obj.distance_trend,
            'behavior': obj.object_behavior,
            'ttc': ttc,
            'ttc_str': format_ttc(ttc),
            'box': obj.bbox,
            'object_id': obj.id,
            'class_priority': class_priority_val,
            'decision': 'CONTINUE DRIVE',
            'priority': 1,
            'confidence': threat_confidence,
            'overlap': overlap_ratio,
            'side_entry': obj.is_side_entry,
            'entry_side': obj.entry_side
        }
        
        # Decision logic hierarchy (same as before)
        if distance <= STOP_DISTANCE_THRESHOLD and is_stable and threat_confidence > 0.6:
            threat_info['decision'] = 'STOP'
            threat_info['priority'] = 5
            threat_info['reason'] = f"STOP: {cname.upper()} at {distance:.1f}m!"
            threats.append(threat_info)
            if priority < 5:
                decision, priority = 'STOP', 5
        
        elif distance <= STOP_DISTANCE_THRESHOLD and (obj.is_side_entry or threat_confidence <= 0.6):
            threat_info['decision'] = 'SLOW DOWN'
            threat_info['priority'] = 2
            threat_info['reason'] = f"Side entry {obj.entry_side}: {cname} at {distance:.1f}m"
            threats.append(threat_info)
            if priority < 2:
                decision, priority = 'SLOW DOWN', 2
        
        elif is_person and distance <= PEDESTRIAN_WARNING_DISTANCE and is_stable:
            threat_info['decision'] = 'STOP'
            threat_info['priority'] = 4
            threat_info['reason'] = f"PEDESTRIAN at {distance:.1f}m - STOP!"
            threats.append(threat_info)
            if priority < 4:
                decision, priority = 'STOP', 4
        
        elif (obj.object_behavior == "stopped" and 
              distance <= zones['warning'] and 
              abs(obj.relative_speed) < STATIONARY_RELATIVE_SPEED and
              is_stable):
            threat_info['decision'] = 'STOP'
            threat_info['priority'] = 4
            threat_info['reason'] = f"{cname.upper()} STOPPED at {distance:.1f}m"
            threats.append(threat_info)
            if priority < 4:
                decision, priority = 'STOP', 4
        
        elif distance <= zones['critical'] and is_stable:
            threat_info['decision'] = 'APPLY BRAKE'
            threat_info['priority'] = 4
            threat_info['reason'] = f"CRITICAL DISTANCE: {distance:.1f}m"
            threats.append(threat_info)
            if priority < 4:
                decision, priority = 'APPLY BRAKE', 4
        
        elif (ttc is not None and ttc != float('inf') and 
              ttc < TTC_THRESHOLDS['critical'] and 
              obj.ttc_confidence > TTC_HIGH_CONFIDENCE and
              is_stable):
            threat_info['decision'] = 'APPLY BRAKE'
            threat_info['priority'] = 4
            threat_info['reason'] = f"TTC CRITICAL: {ttc:.1f}s at {distance:.1f}m"
            threats.append(threat_info)
            if priority < 4:
                decision, priority = 'APPLY BRAKE', 4
        
        elif (obj.object_behavior == "braking" and 
              zones['critical'] < distance <= zones['warning'] and
              threat_confidence > 0.5):
            threat_info['decision'] = 'READY TO BRAKE'
            threat_info['priority'] = 3
            threat_info['reason'] = f"{cname.upper()} BRAKING at {distance:.1f}m"
            threats.append(threat_info)
            if priority < 3:
                decision, priority = 'READY TO BRAKE', 3
        
        elif (ttc is not None and ttc != float('inf') and 
              ttc < TTC_THRESHOLDS['warning'] and 
              zones['critical'] < distance <= zones['warning']):
            threat_info['decision'] = 'FOLLOW CAUTIOUSLY'
            threat_info['priority'] = 3
            threat_info['reason'] = f"TTC Warning: {ttc:.1f}s at {distance:.1f}m"
            threats.append(threat_info)
            if priority < 3:
                decision, priority = 'FOLLOW CAUTIOUSLY', 3
        
        elif (zones['critical'] < distance <= zones['warning'] and 
              obj.distance_trend in ['decreasing', 'decreasing_fast'] and
              threat_confidence > 0.4):
            threat_info['decision'] = 'FOLLOW CAUTIOUSLY'
            threat_info['priority'] = 3
            threat_info['reason'] = f"Closing in warning zone: {distance:.1f}m"
            threats.append(threat_info)
            if priority < 3:
                decision, priority = 'FOLLOW CAUTIOUSLY', 3
        
        elif (ttc is not None and ttc != float('inf') and 
              ttc < TTC_THRESHOLDS['caution'] and 
              zones['warning'] < distance <= zones['caution']):
            threat_info['decision'] = 'SLOW DOWN'
            threat_info['priority'] = 2
            threat_info['reason'] = f"TTC Caution: {ttc:.1f}s at {distance:.1f}m"
            threats.append(threat_info)
            if priority < 2:
                decision, priority = 'SLOW DOWN', 2
        
        elif (zones['warning'] < distance <= zones['caution'] and 
              obj.distance_trend in ['decreasing', 'decreasing_fast']):
            threat_info['decision'] = 'SLOW DOWN'
            threat_info['priority'] = 2
            threat_info['reason'] = f"Closing in caution zone: {distance:.1f}m"
            threats.append(threat_info)
            if priority < 2:
                decision, priority = 'SLOW DOWN', 2
        
        else:
            threat_info['decision'] = 'CONTINUE DRIVE'
            threat_info['priority'] = 1
            if obj.is_side_entry:
                threat_info['reason'] = f"Side {obj.entry_side}: {distance:.1f}m (monitoring)"
            elif obj.distance_trend == 'increasing':
                threat_info['reason'] = f"Moving away: {distance:.1f}m"
            elif overlap_ratio < 1.0:
                threat_info['reason'] = f"Partial overlap ({overlap_ratio:.0%}): {distance:.1f}m"
            elif ttc and ttc != float('inf'):
                threat_info['reason'] = f"Safe: {distance:.1f}m, TTC:{ttc:.1f}s"
            else:
                threat_info['reason'] = f"Safe distance: {distance:.1f}m"
            threats.append(threat_info)
    
    return decision, priority, threats


def get_distance_color_for_gradient(distance, max_distance=50):
    zones = safety_config.get_adjusted_zones()
    
    if distance < zones['critical']:
        return (0, 0, 255)
    elif distance < zones['warning']:
        return (0, 140, 255)
    elif distance < zones['caution']:
        return (0, 215, 255)
    else:
        return (0, 255, 0)
    
    # ============================================================================
# BLOCK 6/8 - DRAWING FUNCTIONS (WITH DUAL DECISION DISPLAY)
# ============================================================================

def draw_trapezoid_with_gradient(frame):
    if not safety_config.show_safety_overlay or not safety_config.trapezoid_area:
        return frame
    
    h, w = frame.shape[:2]
    trap = get_trapezoid_safety_area(frame.shape)
    bl, tl, tr, br = trap
    by, ty = bl[1], tl[1]
    th = by - ty
    
    if safety_config.trapezoid_gradient:
        overlay = frame.copy()
        
        zones = safety_config.get_adjusted_zones()
        distances = [0, zones['critical'], zones['warning'], zones['caution'], 50]
        
        for i in range(len(distances) - 1):
            d1, d2 = distances[i], distances[i + 1]
            r1, r2 = d1 / 50.0, d2 / 50.0
            y1 = int(by - r1 * th)
            y2 = int(by - r2 * th)
            
            x1_left = int(bl[0] + r1 * (tl[0] - bl[0]))
            x1_right = int(br[0] + r1 * (tr[0] - br[0]))
            x2_left = int(bl[0] + r2 * (tl[0] - bl[0]))
            x2_right = int(br[0] + r2 * (tr[0] - br[0]))
            
            pts = np.array([[x1_left, y1], [x2_left, y2], [x2_right, y2], [x1_right, y1]], dtype=np.int32)
            color = get_distance_color_for_gradient(d1)
            cv2.fillPoly(overlay, [pts], color)
        
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    else:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [trap], (255, 255, 0))
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
    
    cv2.polylines(frame, [trap], True, (255, 255, 100), 2)
    
    ego = get_ego_vehicle_box(frame.shape)
    trap_bl, trap_tl, trap_tr, trap_br = trap
    ego_tl = (ego[0], ego[1])
    ego_tr = (ego[2], ego[1])
    
    cv2.line(frame, tuple(trap_bl), ego_tl, (255, 100, 255), 2, cv2.LINE_AA)
    cv2.line(frame, tuple(trap_br), ego_tr, (255, 100, 255), 2, cv2.LINE_AA)
    
    zones = safety_config.get_adjusted_zones()
    marker_distances = [zones['critical'], zones['warning'], zones['caution'], 50]
    
    for dist in marker_distances:
        r = dist / 50.0
        yp = int(by - r * th)
        lx = int(bl[0] + r * (tl[0] - bl[0]))
        rx = int(br[0] + r * (tr[0] - br[0]))
        
        if dist <= zones['critical']:
            line_color = (0, 0, 255)
        elif dist <= zones['warning']:
            line_color = (0, 140, 255)
        elif dist <= zones['caution']:
            line_color = (0, 215, 255)
        else:
            line_color = (0, 255, 0)
        
        cv2.line(frame, (lx, yp), (rx, yp), line_color, 2)
        
        txt = f"{dist:.0f}m"
        ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        
        cv2.rectangle(frame, (lx - ts[0] - 10, yp - ts[1] - 4), (lx - 2, yp + 2), (0, 0, 0), -1)
        cv2.putText(frame, txt, (lx - ts[0] - 8, yp), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.rectangle(frame, (rx + 2, yp - ts[1] - 4), (rx + ts[0] + 10, yp + 2), (0, 0, 0), -1)
        cv2.putText(frame, txt, (rx + 4, yp), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame


# ============================================================================
# UPDATED draw_safety_panel function
# Replace the existing draw_safety_panel function (around line 1620)
# ============================================================================

# ============================================================================
# UPDATED draw_safety_panel - with tighter spacing
# ============================================================================

def draw_safety_panel(frame):
    """Updated to show smoothing status - POSITIONED BELOW MODE PANEL"""
    if not safety_config.show_safety_panel:
        return frame
    
    h, w = frame.shape[:2]
    
    px = w - 340  # Align with detection mode panel
    py = 260      # Below detection mode panel
    pw = 330      # Same width as detection mode panel
    ph = 160      # Height for all features + smoothing info
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (120, 220, 255), 1)
    cv2.rectangle(frame, (px, py), (px + pw, py + 28), (120, 220, 255), -1)
    cv2.putText(frame, "SAFETY FEATURES", (px + 10, py + 19), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    
    y = py + 48
    features = [
        ("[1] Trapezoid", safety_config.trapezoid_area),
        ("[2] Speed Adapt", safety_config.speed_adaptive),
        ("[3] Distance", safety_config.distance_estimation),
        ("[4] Overlay", safety_config.show_safety_overlay)
    ]
    for name, active in features:
        cv2.putText(frame, name, (px + 10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        status = "ON" if active else "OFF"
        col = (100, 255, 100) if active else (120, 120, 120)
        cv2.putText(frame, status, (px + 240, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        y += 17
    
    # Adjust line position upward
    y -= 4 # Move line up
    
    # Smoothing info section separator line
    cv2.line(frame, (px + 10, y), (px + pw - 10, y), (120, 220, 255), 1)
    
    # NEW: Reduced spacing - pushes [D] Smoothing text UP
    y += 15  # Reduced from 15 to 8 (7 pixels higher)
    
    smoother_config = safety_config.decision_smoother.config
    smooth_enabled = smoother_config['enabled']
    
    cv2.putText(frame, "[D] Smoothing", (px + 10, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    status = safety_config.smoothing_preset
    col = (100, 255, 100) if smooth_enabled else (120, 120, 120)
    cv2.putText(frame, status, (px + 240, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    
    # NEW: Reduced spacing - pushes "Frames:" text UP
    y += 18  # Reduced from 20 to 16 (4 pixels higher)
    
    if smooth_enabled:
        debounce = smoother_config['debounce_frames']
        cv2.putText(frame, f"Frames: {debounce}", (px + 240, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    
    return frame


def draw_mode_indicator(frame, detector, fps, num_obj, fc, tf, reason):
    if not safety_config.show_mode_panel:
        return frame
    
    h, w = frame.shape[:2]
    mp = detector.get_current_params()
    bx, by, bw, bh = w - 340, 10, 330, 240
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), mp['color'], 1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 28), mp['color'], -1)
    cv2.putText(frame, "DETECTION MODE", (bx + 10, by + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    
    y = by + 48
    ms = "(Auto)" if not detector.manual_override else "(Manual)"
    cv2.putText(frame, f"Mode: {mp['name']} {ms}", (bx + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mp['color'], 1, cv2.LINE_AA)
    if not detector.manual_override:
        cv2.putText(frame, f"Reason: {reason}", (bx + 10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        y += 18
    cv2.putText(frame, f"{mp['description']}", (bx + 10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(frame, (bx + 10, y + 26), (bx + bw - 10, y + 26), mp['color'], 1)
    cv2.putText(frame, f"Img Size: {mp['img_size']}x{mp['img_size']}", (bx + 10, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {mp['conf_threshold']:.2f}", (bx + 10, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"IoU: {mp['iou_threshold']:.2f}", (bx + 10, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(frame, (bx + 10, by + bh - 75), (bx + bw - 10, by + bh - 75), mp['color'], 1)
    
    ri = RESOLUTION_PRESETS[safety_config.resolution_mode]
    cv2.putText(frame, "[R] Resolution:", (bx + 10, by + bh - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{ri['width']}x{ri['height']}", (bx + 10, by + bh - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    mt = f"[M] Model: {'ONNX' if safety_config.use_onnx else 'PT'}"
    cv2.putText(frame, mt, (bx + 10, by + bh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1, cv2.LINE_AA)
    
    return frame


# ============================================================================
# FIX for draw_adas_overlay function
# Replace the entire function starting from line ~1707
# ============================================================================

def draw_adas_overlay(frame, raw_decision, raw_priority, raw_threats, 
                      smoothed_decision, smoothed_priority, smoothed_threats, 
                      tracked_objects):
    """
    UPDATED: Draw ADAS overlay with both RAW and SMOOTHED decisions
    
    Args:
        frame: Video frame
        raw_decision: Real-time raw decision
        raw_priority: Raw priority level
        raw_threats: Raw threat list
        smoothed_decision: Smoothed decision
        smoothed_priority: Smoothed priority
        smoothed_threats: Smoothed threat list
        tracked_objects: List of TrackedObject instances
    """
    h, w = frame.shape[:2]
    
    # Draw bounding boxes and labels for tracked objects
    for obj in tracked_objects:
        x1, y1, x2, y2 = map(int, obj.bbox)
        cn = CLASS_NAMES[obj.class_id] if obj.class_id < len(CLASS_NAMES) else "Unknown"
        dist = list(obj.distance_history)[-1] if obj.distance_history else None
        
        if obj.frames_tracked >= 5 and dist:
            lbl = f"{cn} #{obj.id} | D:{dist:.1f}m"
            ttc_display = format_ttc(obj.ttc)
            lbl2 = f"S:{obj.object_speed:.0f}km/h | TTC:{ttc_display}"
        else:
            lbl = f"{cn} #{obj.id} | {dist:.1f}m" if dist else f"{cn} #{obj.id}"
            lbl2 = f"Track: {obj.frames_tracked}f"
        
        # Use smoothed threats for coloring
        is_threat = any(t['object_id'] == obj.id for t in smoothed_threats)
        if is_threat:
            threat = next(t for t in smoothed_threats if t['object_id'] == obj.id)
            if threat['priority'] == 5:
                col = (0, 0, 139)
            elif threat['priority'] == 4:
                col = (0, 0, 255)
            elif threat['priority'] == 3:
                col = (0, 140, 255)
            elif threat['priority'] == 2:
                col = (0, 215, 255)
            else:
                col = (0, 255, 0)
        else:
            col = (100, 100, 100)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        
        ls = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(frame, (x1, y1 - ls[1] - 6), (x1 + ls[0] + 4, y1), col, -1)
        cv2.putText(frame, lbl, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
        if obj.frames_tracked >= 5 and dist:
            ls2 = cv2.getTextSize(lbl2, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            cv2.rectangle(frame, (x1, y2 + 1), (x1 + ls2[0] + 4, y2 + ls2[1] + 7), (0, 0, 0), -1)
            cv2.putText(frame, lbl2, (x1 + 2, y2 + ls2[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 255), 1, cv2.LINE_AA)
    
    # Draw ego vehicle box
    ego = get_ego_vehicle_box(frame.shape)
    cv2.rectangle(frame, (ego[0], ego[1]), (ego[2], ego[3]), (255, 100, 255), 2)
    
    ts = cv2.getTextSize("EGO VEHICLE", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    tx = ego[0] + (ego[2] - ego[0] - ts[0]) // 2
    ty = ego[1] - 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (tx - 4, ty - ts[1] - 4), (tx + ts[0] + 4, ty + 2), (255, 100, 255), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.rectangle(frame, (tx - 4, ty - ts[1] - 4), (tx + ts[0] + 4, ty + 2), (255, 100, 255), 1)
    cv2.putText(frame, "EGO VEHICLE", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Speed display
    sm = "M" if safety_config.manual_speed_mode else "A"
    st = f"[{sm}] {safety_config.current_speed:.1f} km/h"
    sy = ego[3] + 20
    ss = cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    sx = ego[0] + (ego[2] - ego[0] - ss[0]) // 2
    sov = frame.copy()
    cv2.rectangle(sov, (sx - 4, sy - ss[1] - 4), (sx + ss[0] + 4, sy + 2), (0, 0, 0), -1)
    cv2.addWeighted(sov, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (sx - 4, sy - ss[1] - 4), (sx + ss[0] + 4, sy + 2), (100, 255, 255), 1)
    cv2.putText(frame, st, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 1, cv2.LINE_AA)
    
    # ========================================================================
    # NEW: DUAL DECISION DISPLAY (RAW + SMOOTHED)
    # ========================================================================
    
    # Initialize common variables
    bw = 360  # box width for dual mode
    by = 10   # box y position
    threats_y = 0
    threat_box_x = 0
    threat_box_width = 400
    
    if safety_config.show_both_decisions:
        # Display BOTH raw and smoothed decisions side by side
        
        # RAW decision (left side)
        raw_col = DECISION_COLORS.get(raw_decision, (255, 255, 255))
        bh = 85
        bx1 = (w // 2) - bw - 10
        
        cv2.rectangle(frame, (bx1, by), (bx1 + bw, by + bh), raw_col, -1)
        cv2.rectangle(frame, (bx1, by), (bx1 + bw, by + bh), (255, 255, 255), 2)
        cv2.putText(frame, "RAW (Real-time)", (bx1 + 10, by + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
        dts = cv2.getTextSize(raw_decision, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        dtx = bx1 + (bw - dts[0]) // 2
        dty = by + 35 + (bh - 35 + dts[1]) // 2
        cv2.putText(frame, raw_decision, (dtx, dty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, raw_decision, (dtx, dty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        # SMOOTHED decision (right side)
        smooth_col = DECISION_COLORS.get(smoothed_decision, (255, 255, 255))
        bx2 = (w // 2) + 10
        
        cv2.rectangle(frame, (bx2, by), (bx2 + bw, by + bh), smooth_col, -1)
        cv2.rectangle(frame, (bx2, by), (bx2 + bw, by + bh), (255, 255, 255), 2)
        
        preset_name = safety_config.smoothing_preset
        cv2.putText(frame, f"SMOOTHED ({preset_name})", (bx2 + 10, by + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
        dts2 = cv2.getTextSize(smoothed_decision, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        dtx2 = bx2 + (bw - dts2[0]) // 2
        dty2 = by + 35 + (bh - 35 + dts2[1]) // 2
        cv2.putText(frame, smoothed_decision, (dtx2, dty2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, smoothed_decision, (dtx2, dty2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Set threats display position (centered below both boxes)
        threats_y = by + bh + 12
        threat_box_x = bx1  # Start from left box position
        threat_box_width = (bx2 + bw) - bx1  # Span across both boxes
        
    else:
        # Display ONLY smoothed decision (centered)
        smooth_col = DECISION_COLORS.get(smoothed_decision, (255, 255, 255))
        bw = 400
        bh = 70
        bx = (w - bw) // 2
        
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), smooth_col, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
        
        dts = cv2.getTextSize(smoothed_decision, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        dtx = bx + (bw - dts[0]) // 2
        dty = by + (bh + dts[1]) // 2
        cv2.putText(frame, smoothed_decision, (dtx, dty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, smoothed_decision, (dtx, dty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Set threats display position
        threats_y = by + bh + 12
        threat_box_x = bx
        threat_box_width = bw
    
    # ========================================================================
    # Threat list (use smoothed threats)
    # ========================================================================
    if smoothed_threats:
        tth = 24 + (min(len(smoothed_threats), 3) * 50)
        tov = frame.copy()
        cv2.rectangle(tov, (threat_box_x, threats_y), (threat_box_x + threat_box_width, threats_y + tth), (0, 0, 0), -1)
        cv2.addWeighted(tov, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (threat_box_x, threats_y), (threat_box_x + threat_box_width, threats_y + tth), (255, 255, 255), 1)
        cv2.putText(frame, f"THREATS: {len(smoothed_threats)}", (threat_box_x + 10, threats_y + 16), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1, cv2.LINE_AA)
        
        yo = threats_y + 36
        sorted_threats = sorted(smoothed_threats, key=lambda x: x['priority'], reverse=True)[:3]
        for i, t in enumerate(sorted_threats):
            tt = f"{i+1}. {t['class']} #{t['object_id']} | {t['distance_str']} | {t['trend'].upper()}"
            cv2.putText(frame, tt, (threat_box_x + 8, yo), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
            yo += 16
            
            tt2 = f"   Spd:{t['object_speed']:.0f}km/h | Rel:{t['relative_speed']:.0f}km/h | {t['behavior'].upper()}"
            cv2.putText(frame, tt2, (threat_box_x + 8, yo), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1, cv2.LINE_AA)
            yo += 16
            
            tt3 = f"   {t['reason']}"
            cv2.putText(frame, tt3, (threat_box_x + 8, yo), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 200, 255), 1, cv2.LINE_AA)
            yo += 18
    
    return frame


def draw_stats_panel(frame, fps, num_obj, fc, tf, tracker, process_time):
    """Updated to show smoothing statistics"""
    if not safety_config.show_stats_panel:
        return frame
    
    h, w = frame.shape[:2]
    
    # Extended height for smoothing stats
    panel_height = 220
    
    tov = frame.copy()
    cv2.rectangle(tov, (10, 10), (290, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(tov, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (10, 10), (290, panel_height), (100, 255, 100), 1)
    
    y = 38
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Time: {process_time*1000:.1f}ms", (20, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Objects: {num_obj}", (20, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Tracked: {len(tracker.objects)}", (20, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {fc}/{tf}", (20, y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Speed: {safety_config.current_speed:.0f} km/h", (20, y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)
    
    # NEW: Smoothing stats
    cv2.line(frame, (20, y + 142), (280, y + 142), (100, 255, 100), 1)
    y += 156
    
    stats = safety_config.decision_smoother.get_smoothing_stats()
    cv2.putText(frame, f"Smooth: {stats['preset']}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1, cv2.LINE_AA)
    
    if stats['enabled']:
        y += 18
        cv2.putText(frame, f"Debounce: {stats['debounce_frames']}f", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
    
    if safety_config.recording:
        cv2.putText(frame, "REC", (250, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    
    return frame


def draw_pause_indicator(frame):
    if not safety_config.paused:
        return frame
    
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    txt = "PAUSED"
    fs, th = 2.0, 3
    ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0]
    tx = (w - ts[0]) // 2
    ty = (h + ts[1]) // 2
    pad = 20
    
    cv2.rectangle(frame, (tx - pad, ty - ts[1] - pad), (tx + ts[0] + pad, ty + pad), (0, 0, 0), -1)
    cv2.rectangle(frame, (tx - pad, ty - ts[1] - pad), (tx + ts[0] + pad, ty + pad), (0, 255, 255), 2)
    cv2.putText(frame, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), th, cv2.LINE_AA)
    
    inst = "Press SPACE to resume | Arrow keys to navigate"
    ins = cv2.getTextSize(inst, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    ix = (w - ins[0]) // 2
    cv2.putText(frame, inst, (ix, ty + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    
    return frame


def draw_recording_indicator(frame):
    if not safety_config.recording:
        return frame
    
    h, w = frame.shape[:2]
    
    cv2.rectangle(frame, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 35), (0, 0, 255), 2)
    
    blink = int(time.time() * 2) % 2
    if blink:
        cv2.circle(frame, (20, 17), 8, (0, 0, 255), -1)
    
    cv2.putText(frame, f"RECORDING - Frame {safety_config.record_start_frame} onwards", 
                (40, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    
    return frame


def show_help_window():
    """Updated help window with smoothing controls"""
    help_img = np.zeros((1100, 900, 3), dtype=np.uint8)  # Increased size
    help_img[:] = (40, 40, 40)
    
    cv2.rectangle(help_img, (0, 0), (900, 45), (100, 200, 255), -1)
    cv2.putText(help_img, "ADAS KEYBOARD SHORTCUTS [H]", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    y = 50
    sections = [
        ("DETECTION MODES", [
            ("H", "Highway Mode"),
            ("A", "Auto Mode (Adaptive)")
        ]),
        ("SAFETY FEATURES", [
            ("1", "Toggle Trapezoid Area"),
            ("2", "Toggle Speed Adaptive Zones"),
            ("3", "Toggle Distance Estimation"),
            ("4", "Toggle Safety Overlay")
        ]),
        ("DECISION SMOOTHING (NEW!)", [
            ("D", "Cycle Smoothing Preset (OFF/LIGHT/MEDIUM/STRONG/VERY_STRONG)"),
            ("B", "Toggle Display Mode (Raw+Smoothed / Smoothed Only)"),
            ("9", "Decrease Debounce Frames"),
            ("0", "Increase Debounce Frames")
        ]),
        ("PANEL CONTROLS", [
            ("Z", "Toggle Stats Panel"),
            ("X", "Toggle Safety Panel"),
            ("C", "Toggle Mode Panel")
        ]),
        ("TRAPEZOID ADJUSTMENT", [
            ("Up/Down Arrow", "Adjust Horizon Line"),
            ("Left/Right Arrow", "Move Vanishing Point"),
            ("W / N", "Bottom Width (Wide/Narrow)"),
            ("T / Y", "Top Width (Wide/Narrow)"),
            ("[ / ]", "Left Angle (In/Out)"),
            ("; / '", "Right Angle (In/Out)"),
            ("O", "Save Trapezoid Config"),
            ("L", "Load Trapezoid Config")
        ]),
        ("PLAYBACK & RECORDING", [
            ("SPACE", "Pause/Resume Video"),
            ("Left/Right (Paused)", "Navigate Frames"),
            ("Page Up/Down", "Skip 10 Frames"),
            ("7", "Start Recording"),
            ("8", "Stop Recording")
        ]),
        ("SYSTEM CONTROLS", [
            ("M", "Toggle Model (PT/ONNX)"),
            ("R", "Cycle Resolution"),
            ("S", "Toggle Speed Mode (Manual/Auto)"),
            ("+ / =", "Increase Ego Speed"),
            ("- / _", "Decrease Ego Speed"),
            ("Q", "Quit Application")
        ]),
        ("DECISION PRIORITIES", [
            ("Priority 5", "STOP (<3m)"),
            ("Priority 4", "APPLY BRAKE / CRITICAL"),
            ("Priority 3", "READY TO BRAKE / FOLLOW CAUTIOUSLY"),
            ("Priority 2", "SLOW DOWN"),
            ("Priority 1", "CONTINUE DRIVE")
        ])
    ]
    
    for section_title, items in sections:
        cv2.rectangle(help_img, (10, y - 18), (890, y + 2), (80, 80, 80), -1)
        cv2.putText(help_img, section_title, (20, y - 3), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
        
        for key, desc in items:
            cv2.putText(help_img, key, (25, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(help_img, desc, (240, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
            y += 22
        
        y += 5
    
    footer_y = help_img.shape[0] - 15
    cv2.putText(help_img, "Press 'H' to close this window", 
                (help_img.shape[1]//2 - 130, footer_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 1, cv2.LINE_AA)
    
    return help_img


def create_video_writer(input_cap, output_path, fps=30):
    width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    res = RESOLUTION_PRESETS[safety_config.resolution_mode]
    width, height = res['width'], res['height']
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"⚠ Failed to create video writer: {output_path}")
        return None
    
    print(f"✓ Video writer created: {output_path} ({width}x{height} @ {fps}fps)")
    return writer
# ============================================================================
# BLOCK 7/8 - MODEL INITIALIZATION + MAIN PROCESSING LOOP
# ============================================================================

# GPU/CPU Detection
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"✓ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    DEVICE = 'cuda:0'
else:
    print("⚠ No GPU detected - using CPU")
    DEVICE = 'cpu'

# Load Models
print("⏳ Loading models...")
pt_model = YOLO(PT_MODEL_PATH)
CLASS_NAMES = list(pt_model.names.values())
print(f"✓ PyTorch model loaded ({len(CLASS_NAMES)} classes)")

if CUDA_AVAILABLE:
    pt_model.to(DEVICE)
    print(f"✓ PyTorch model moved to {DEVICE}")

try:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if CUDA_AVAILABLE else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(MODEL_PATH, providers=providers)
    active_provider = ort_session.get_providers()[0]
    print(f"✓ ONNX model loaded on {active_provider}")
except Exception as e:
    print(f"⚠ ONNX failed: {e}")
    safety_config.use_onnx = False

# Initialize components
object_tracker = ObjectTracker(max_disappeared=TRACKER_MAX_DISAPPEARED, iou_threshold=TRACKER_IOU_THRESHOLD)
object_tracker.set_class_names(CLASS_NAMES)
detector = AdaptiveDetector('AUTO')

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Print startup information
print("\n" + "="*80)
print("ENHANCED ADAS DETECTION SYSTEM - GPU ACCELERATED + DECISION SMOOTHING")
print("="*80)
print("MODES: [H]-Highway [A]-Auto")
print("SAFETY: [1]-Trapezoid [2]-Speed [3]-Distance [4]-Overlay")
print("SMOOTHING: [D]-Cycle Preset [B]-Toggle Display [9/0]-Debounce")
print("PANELS: [Z]-Stats [X]-Safety [C]-Mode")
print("TRAPEZOID: Arrows=Horizon/VP | W/N=Bottom | T/Y=Top | [/]/;/'=Angles")
print("           [O]-Save [L]-Load")
print("PLAYBACK: [SPACE]-Pause | Arrows-Navigate | [7/8]-Record")
print("SYSTEM: [M]-Model [R]-Resolution [S]-Speed [+/-]-Speed [H]-Help [Q]-Quit")
print("="*80 + "\n")

# Processing state variables
frame_count, total_time = 0, 0
last_boxes, last_classes, last_scores = [], [], []
current_scale = DISPLAY_SCALE
current_frame = None

# Decision state variables (NEW for smoothing)
raw_decision, raw_priority, raw_threats = "CONTINUE DRIVE", 1, []
smoothed_decision, smoothed_priority, smoothed_threats = "CONTINUE DRIVE", 1, []
current_tracked_objects = []
current_fps, process_time = 0, 0.001
auto_mode_reason = "Initial"

# Video recording
video_writer = None
show_help = False
help_window = None

print("✓ Starting video processing...")
print(f"✓ Resolution: {RESOLUTION_PRESETS[safety_config.resolution_mode]['name']}")
print(f"✓ Speed Mode: {'MANUAL' if safety_config.manual_speed_mode else 'AUTO'}")
print(f"✓ Speed: {safety_config.current_speed:.1f} km/h")
print(f"✓ Model: {'ONNX' if safety_config.use_onnx else 'PT'}")
print(f"✓ Smoothing: {safety_config.smoothing_preset}")
print(f"✓ Display: {'BOTH (Raw+Smoothed)' if safety_config.show_both_decisions else 'SMOOTHED ONLY'}\n")

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

while True:
    if not safety_config.paused:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = resize_frame_to_resolution(frame)
        frame_count += 1
        current_timestamp = time.time()
        mode_params = detector.get_current_params()
        
        # Run inference on appropriate frames
        if frame_count % (mode_params['frame_skip'] + 1) == 0:
            start_time = time.time()
            try:
                # Run detection
                if safety_config.use_onnx:
                    boxes, classes, scores = run_inference_onnx(
                        frame, mode_params['img_size'], 
                        mode_params['conf_threshold'], mode_params['iou_threshold']
                    )
                else:
                    boxes, classes, scores = run_inference_pt(
                        frame, mode_params['img_size'], 
                        mode_params['conf_threshold'], mode_params['iou_threshold']
                    )
                
                # Remove duplicates
                boxes, classes, scores = remove_duplicate_detections(boxes, classes, scores)
                
                # Estimate distances
                distances = [
                    estimate_distance(box, frame.shape, 
                                    CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "car") or 50.0
                    for box, cls in zip(boxes, classes)
                ]
                
                # Track objects
                tracked_objects = object_tracker.match_objects(
                    boxes, classes, scores, distances, current_timestamp, frame_count
                )
                
                # Analyze trends
                for obj in tracked_objects:
                    obj.analyze_trends(safety_config.current_speed)
                
                process_time = time.time() - start_time
                total_time += process_time
                last_boxes, last_classes, last_scores = boxes, classes, scores
                
            except Exception as e:
                print(f"⚠ Inference error: {e}")
                boxes, classes, scores = last_boxes, last_classes, last_scores
                tracked_objects = object_tracker.get_all_objects()
                process_time = 0.001
        else:
            boxes, classes, scores = last_boxes, last_classes, last_scores
            tracked_objects = object_tracker.get_all_objects()
            process_time = 0.001
        
        # Estimate ego speed
        safety_config.estimate_speed(len(tracked_objects))
        
        # Update auto mode
        auto_mode_reason = detector.update_auto_mode(safety_config.current_speed, len(tracked_objects))
        
        # ====================================================================
        # NEW: GET RAW DECISION AND APPLY SMOOTHING
        # ====================================================================
        
        # Get RAW decision from ADAS logic
        raw_decision, raw_priority, raw_threats = make_adas_decision(
            tracked_objects, frame.shape, detector.current_mode
        )
        
        # Apply smoothing to get SMOOTHED decision
        smoothed_decision, smoothed_priority, smoothed_threats, decision_changed = \
            safety_config.decision_smoother.update(
                raw_decision, raw_priority, raw_threats, frame_count
            )
        
        # Optional: Log when smoothed decision changes
        if decision_changed and smoothed_priority >= 3:
            print(f"⚠ Frame {frame_count}: Decision changed to [{smoothed_decision}] (Priority: {smoothed_priority})")
        
        # ====================================================================
        
        fps = 1 / process_time if process_time > 0 else 0
        
        # Store current state for pause functionality
        current_frame = frame.copy()
        current_tracked_objects = tracked_objects
        current_fps = fps
    
    else:
        # Paused state - use stored frame
        if current_frame is not None:
            frame = current_frame.copy()
            tracked_objects = current_tracked_objects
            fps = current_fps
        else:
            continue
    
    # ========================================================================
    # DRAW ALL OVERLAYS (using both raw and smoothed decisions)
    # ========================================================================
    
    frame = draw_trapezoid_with_gradient(frame)
    
    # Pass BOTH raw and smoothed decisions to overlay
    frame = draw_adas_overlay(
        frame, 
        raw_decision, raw_priority, raw_threats,
        smoothed_decision, smoothed_priority, smoothed_threats,
        tracked_objects
    )
    
    frame = draw_mode_indicator(frame, detector, fps, len(tracked_objects), 
                                frame_count, total_frames, auto_mode_reason)
    frame = draw_safety_panel(frame)
    frame = draw_stats_panel(frame, fps, len(tracked_objects), frame_count, 
                             total_frames, object_tracker, process_time)
    frame = draw_recording_indicator(frame)
    frame = draw_pause_indicator(frame)
    
    # Record if active
    if safety_config.recording and video_writer is not None:
        video_writer.write(frame)
    
    # Display frame
    if current_scale != 1.0:
        dw, dh = int(frame.shape[1] * current_scale), int(frame.shape[0] * current_scale)
        display_frame = cv2.resize(frame, (dw, dh))
    else:
        display_frame = frame
    
    cv2.imshow(WINDOW_NAME, display_frame)
    
    # Show help window if toggled
    if show_help:
        if help_window is None:
            help_window = show_help_window()
        cv2.imshow("Keyboard Shortcuts", help_window)
    else:
        try:
            cv2.destroyWindow("Keyboard Shortcuts")
        except:
            pass
        help_window = None
    
    # ========================================================================
    # KEYBOARD INPUT HANDLING
    # ========================================================================
    
    wait_time = 1 if not safety_config.paused else 0
    key = cv2.waitKeyEx(wait_time)
    
    if key == ord('q'):
        print("\n✓ User quit")
        break
    
    elif key == ord(' '):
        safety_config.toggle_pause()
    
    elif key == ord('h'):
        show_help = not show_help
        print(f"✓ Help: {'SHOWN' if show_help else 'HIDDEN'}")
    
    # ====================================================================
    # NEW: SMOOTHING CONTROLS
    # ====================================================================
    
    elif key == ord('d'):
        safety_config.cycle_smoothing_preset()
    
    elif key == ord('b'):
        safety_config.toggle_both_decisions()
    
    elif key == ord('9'):
        safety_config.decrease_debounce()
    
    elif key == ord('0'):
        safety_config.increase_debounce()
    
    # ====================================================================
    
    # Detection mode controls
    elif key == ord('a'):
        detector.enable_auto_mode()
    
    # Safety feature toggles
    elif key == ord('1'):
        safety_config.toggle_trapezoid()
    
    elif key == ord('2'):
        safety_config.toggle_speed_adaptive()
    
    elif key == ord('3'):
        safety_config.toggle_distance_estimation()
    
    elif key == ord('4'):
        safety_config.toggle_safety_overlay()
    
    # Panel toggles
    elif key == ord('z'):
        safety_config.toggle_stats_panel()
    
    elif key == ord('x'):
        safety_config.toggle_safety_panel()
    
    elif key == ord('c'):
        safety_config.toggle_mode_panel()
    
    # System controls
    elif key == ord('m'):
        safety_config.toggle_model()
    
    elif key == ord('r'):
        safety_config.cycle_resolution()
    
    elif key == ord('s'):
        safety_config.toggle_speed_mode()
    
    elif key == ord('+') or key == ord('='):
        safety_config.increase_speed()
    
    elif key == ord('-') or key == ord('_'):
        safety_config.decrease_speed()
    
    # Recording controls
    elif key == ord('7'):
        if not safety_config.recording:
            safety_config.start_recording(frame_count)
            output_path = f"adas_recording_{int(time.time())}.mp4"
            video_writer = create_video_writer(cap, output_path, video_fps)
    
    elif key == ord('8'):
        if safety_config.recording:
            safety_config.stop_recording(frame_count)
            if video_writer is not None:
                video_writer.release()
                video_writer = None
                print(f"✓ Video saved successfully")
    
    # Trapezoid adjustment controls
    elif key == 2490368 or key == 65362:  # Up arrow
        trapezoid_config.adjust_horizon_up()
    
    elif key == 2621440 or key == 65364:  # Down arrow
        trapezoid_config.adjust_horizon_down()
    
    elif key == 2424832 or key == 65361:  # Left arrow
        if safety_config.paused:
            new_pos = max(0, frame_count - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            frame_count = new_pos
            safety_config.paused = False
            print(f"⏪ Frame {frame_count}")
        else:
            trapezoid_config.adjust_vp_left()
    
    elif key == 2555904 or key == 65363:  # Right arrow
        if safety_config.paused:
            new_pos = min(total_frames - 1, frame_count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            frame_count = new_pos
            safety_config.paused = False
            print(f"⏩ Frame {frame_count}")
        else:
            trapezoid_config.adjust_vp_right()
    
    elif key == 2162688 or key == 65365:  # Page Up
        new_pos = max(0, frame_count - 11)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        frame_count = new_pos
        if safety_config.paused:
            safety_config.paused = False
        print(f"⏪⏪ Frame {frame_count}")
    
    elif key == 2228224 or key == 65366:  # Page Down
        new_pos = min(total_frames - 1, frame_count + 9)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        frame_count = new_pos
        if safety_config.paused:
            safety_config.paused = False
        print(f"⏩⏩ Frame {frame_count}")
    
    elif key == ord('w'):
        trapezoid_config.adjust_width_wide()
    
    elif key == ord('n'):
        trapezoid_config.adjust_width_narrow()
    
    elif key == ord('t'):
        trapezoid_config.adjust_top_wide()
    
    elif key == ord('y'):
        trapezoid_config.adjust_top_narrow()
    
    elif key == ord('['):
        trapezoid_config.adjust_left_angle_in()
    
    elif key == ord(']'):
        trapezoid_config.adjust_left_angle_out()
    
    elif key == ord(';'):
        trapezoid_config.adjust_right_angle_in()
    
    elif key == ord("'"):
        trapezoid_config.adjust_right_angle_out()
    
    elif key == ord('o'):
        trapezoid_config.save_config()
    
    elif key == ord('l'):
        trapezoid_config.load_config()

# ============================================================================
# BLOCK 8/8 - CLEANUP AND SHUTDOWN (FINAL)
# ============================================================================

# Cleanup resources
if video_writer is not None:
    video_writer.release()
    print("✓ Recording saved and closed")

cap.release()
cv2.destroyAllWindows()

# ============================================================================
# FINAL STATISTICS AND REPORT
# ============================================================================

print("\n" + "="*80)
print("ADAS SYSTEM SHUTDOWN - FINAL REPORT")
print("="*80)

# Processing statistics
avg_fps = frame_count / total_time if total_time > 0 else 0
print(f"\n📊 PROCESSING STATISTICS:")
print(f"  ├─ Total frames processed: {frame_count}/{total_frames}")
print(f"  ├─ Total processing time: {total_time:.2f}s")
print(f"  ├─ Average FPS: {avg_fps:.2f}")
print(f"  └─ Average frame time: {(total_time/frame_count)*1000:.2f}ms" if frame_count > 0 else "  └─ No frames processed")

# Detection statistics
print(f"\n🎯 DETECTION STATISTICS:")
print(f"  ├─ Detection mode: {detector.current_mode}")
print(f"  ├─ Model used: {'ONNX' if safety_config.use_onnx else 'PyTorch'}")
print(f"  ├─ Device: {DEVICE}")
print(f"  ├─ Total tracked objects: {object_tracker.next_id}")
print(f"  └─ Active objects at end: {len(object_tracker.objects)}")

# Smoothing statistics (NEW!)
smoothing_stats = safety_config.decision_smoother.get_smoothing_stats()
print(f"\n🔄 DECISION SMOOTHING STATISTICS:")
print(f"  ├─ Smoothing preset: {smoothing_stats['preset']}")
print(f"  ├─ Smoothing enabled: {'YES' if smoothing_stats['enabled'] else 'NO'}")
if smoothing_stats['enabled']:
    print(f"  ├─ Debounce frames: {smoothing_stats['debounce_frames']}")
    print(f"  ├─ Total decisions: {smoothing_stats['total']}")
    print(f"  ├─ Smoothed changes: {smoothing_stats['smoothed']}")
    if smoothing_stats['total'] > 0:
        stability_ratio = (smoothing_stats['total'] - smoothing_stats['smoothed']) / smoothing_stats['total'] * 100
        print(f"  └─ Stability gain: {stability_ratio:.1f}% (fewer decision changes)")
else:
    print(f"  └─ Raw decisions used: {smoothing_stats['total']}")

# Safety configuration
print(f"\n⚙️ SAFETY CONFIGURATION:")
print(f"  ├─ Trapezoid area: {'ON' if safety_config.trapezoid_area else 'OFF'}")
print(f"  ├─ Speed adaptive: {'ON' if safety_config.speed_adaptive else 'OFF'}")
print(f"  ├─ Distance estimation: {'ON' if safety_config.distance_estimation else 'OFF'}")
print(f"  ├─ Safety overlay: {'ON' if safety_config.show_safety_overlay else 'OFF'}")
print(f"  ├─ Display mode: {'BOTH (Raw+Smoothed)' if safety_config.show_both_decisions else 'SMOOTHED ONLY'}")
print(f"  └─ Ego speed: {safety_config.current_speed:.1f} km/h ({safety_config.manual_speed_mode and 'MANUAL' or 'AUTO'})")

# Recording information
if safety_config.recording or safety_config.record_end_frame > 0:
    print(f"\n📹 RECORDING INFORMATION:")
    if safety_config.record_end_frame > safety_config.record_start_frame:
        recorded_frames = safety_config.record_end_frame - safety_config.record_start_frame
        print(f"  ├─ Recorded frames: {safety_config.record_start_frame} to {safety_config.record_end_frame}")
        print(f"  ├─ Total recorded: {recorded_frames} frames")
        print(f"  └─ Duration: {recorded_frames / video_fps:.2f}s")
    else:
        print(f"  └─ Recording was started but not properly stopped")

# Performance summary
print(f"\n⚡ PERFORMANCE SUMMARY:")
if avg_fps >= 25:
    perf_rating = "EXCELLENT"
    perf_emoji = "🟢"
elif avg_fps >= 15:
    perf_rating = "GOOD"
    perf_emoji = "🟡"
elif avg_fps >= 10:
    perf_rating = "ACCEPTABLE"
    perf_emoji = "🟠"
else:
    perf_rating = "NEEDS OPTIMIZATION"
    perf_emoji = "🔴"

print(f"  {perf_emoji} Performance: {perf_rating} ({avg_fps:.2f} FPS)")

if CUDA_AVAILABLE:
    print(f"  ✓ GPU acceleration was active")
else:
    print(f"  ⚠ Running on CPU (consider GPU for better performance)")

# Smoothing effectiveness summary
if smoothing_stats['enabled'] and smoothing_stats['total'] > 0:
    print(f"\n💡 SMOOTHING EFFECTIVENESS:")
    stability_ratio = (smoothing_stats['total'] - smoothing_stats['smoothed']) / smoothing_stats['total'] * 100
    
    if stability_ratio >= 80:
        smooth_rating = "VERY STABLE"
        smooth_emoji = "🟢"
    elif stability_ratio >= 60:
        smooth_rating = "STABLE"
        smooth_emoji = "🟡"
    elif stability_ratio >= 40:
        smooth_rating = "MODERATE"
        smooth_emoji = "🟠"
    else:
        smooth_rating = "LOW FILTERING"
        smooth_emoji = "🔴"
    
    print(f"  {smooth_emoji} Stability: {smooth_rating} ({stability_ratio:.1f}%)")
    print(f"  ├─ Decision fluctuations reduced by {stability_ratio:.1f}%")
    print(f"  └─ Preset '{smoothing_stats['preset']}' with {smoothing_stats['debounce_frames']} frame buffer")

# Tips and recommendations
print(f"\n💭 TIPS:")
tips_shown = False

if not CUDA_AVAILABLE:
    print(f"  • Install CUDA-enabled PyTorch for 3-5x faster processing")
    tips_shown = True

if avg_fps < 15:
    print(f"  • Try HIGHWAY mode for faster processing (lower accuracy)")
    print(f"  • Reduce resolution or increase frame_skip in detection params")
    tips_shown = True

if not smoothing_stats['enabled']:
    print(f"  • Enable smoothing (press D) to reduce decision fluctuations")
    tips_shown = True
elif smoothing_stats['debounce_frames'] < 5:
    print(f"  • Increase debounce frames (press 0) for more stable decisions")
    tips_shown = True

if not safety_config.show_both_decisions:
    print(f"  • Press B to see both RAW and SMOOTHED decisions side-by-side")
    tips_shown = True

if not tips_shown:
    print(f"  ✓ System is optimally configured!")

print("\n" + "="*80)
print("✓ ADAS SYSTEM TERMINATED SUCCESSFULLY")
print("="*80)

# Optional: Save session report to file
try:
    report_filename = f"adas_session_report_{int(time.time())}.txt"
    with open(report_filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ADAS DETECTION SYSTEM - SESSION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PROCESSING STATISTICS:\n")
        f.write(f"  Total frames: {frame_count}/{total_frames}\n")
        f.write(f"  Processing time: {total_time:.2f}s\n")
        f.write(f"  Average FPS: {avg_fps:.2f}\n\n")
        
        f.write("DETECTION CONFIGURATION:\n")
        f.write(f"  Mode: {detector.current_mode}\n")
        f.write(f"  Model: {'ONNX' if safety_config.use_onnx else 'PyTorch'}\n")
        f.write(f"  Device: {DEVICE}\n")
        f.write(f"  Total tracked objects: {object_tracker.next_id}\n\n")
        
        f.write("SMOOTHING CONFIGURATION:\n")
        f.write(f"  Preset: {smoothing_stats['preset']}\n")
        f.write(f"  Enabled: {smoothing_stats['enabled']}\n")
        if smoothing_stats['enabled']:
            f.write(f"  Debounce frames: {smoothing_stats['debounce_frames']}\n")
            f.write(f"  Total decisions: {smoothing_stats['total']}\n")
            f.write(f"  Decision changes: {smoothing_stats['smoothed']}\n")
            if smoothing_stats['total'] > 0:
                stability = (smoothing_stats['total'] - smoothing_stats['smoothed']) / smoothing_stats['total'] * 100
                f.write(f"  Stability gain: {stability:.1f}%\n")
        f.write("\n")
        
        f.write("SAFETY FEATURES:\n")
        f.write(f"  Trapezoid area: {safety_config.trapezoid_area}\n")
        f.write(f"  Speed adaptive: {safety_config.speed_adaptive}\n")
        f.write(f"  Distance estimation: {safety_config.distance_estimation}\n")
        f.write(f"  Display mode: {'BOTH' if safety_config.show_both_decisions else 'SMOOTHED'}\n")
        f.write(f"  Ego speed: {safety_config.current_speed:.1f} km/h\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"📄 Session report saved: {report_filename}\n")
except Exception as e:
    print(f"⚠ Could not save session report: {e}\n")

print("Thank you for using the ADAS Detection System!")
print("Press any key to exit...")