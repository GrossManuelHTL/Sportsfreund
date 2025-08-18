"""
State Machine System for Exercise Recognition
Handles state transitions and rep counting based on pose data
"""
from typing import Dict, List, Any, Optional
from enum import Enum
import json


class ExerciseState:
    """Represents a state in the exercise state machine"""
    def __init__(self, name: str, conditions: Dict[str, Any]):
        self.name = name
        self.conditions = conditions
        self.entry_time = None


class StateMachine:
    """Exercise state machine for rep counting and movement tracking"""

    def __init__(self, exercise_config: Dict[str, Any]):
        self.exercise_name = exercise_config['name']
        self.states = {}
        self.current_state = None
        self.rep_count = 0
        self.state_history = []
        self.frame_time = 0

        self._load_states(exercise_config.get('states', {}))

        initial_state = exercise_config.get('initial_state', 'standing')
        self.current_state = initial_state

        self.rep_pattern = exercise_config.get('rep_pattern')
        self.pattern_progress = 0

        self.frames_in_current_state = 0
        self.state_confidence = 0
        self.min_confidence_for_transition = 2  # Frames needed to confirm transition

        self.last_rep_frame = 0
        self.min_frames_between_reps = 0  # Minimum frames between rep counts

    def _load_states(self, states_config: Dict[str, Any]):
        """Load states from configuration"""

        for state_name, state_data in states_config.items():
            self.states[state_name] = ExerciseState(
                name=state_name,
                conditions=state_data.get('conditions', {})
            )

    def update(self, pose_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update state machine with new pose data"""
        self.frame_time += 1
        self.frames_in_current_state += 1

        if not pose_data:
            return self._get_status()

        # Check for state transitions with confidence
        potential_new_state = self._check_transitions(pose_data)

        if potential_new_state and potential_new_state != self.current_state:
            self.state_confidence += 1


            # Only transition if we're confident (multiple consecutive frames)
            if self.state_confidence >= self.min_confidence_for_transition:
                # Also check minimum duration in current state
                current_state_obj = self.states.get(self.current_state)
                if current_state_obj:
                    self._transition_to_state(potential_new_state)
        else:
            # Reset confidence if we don't consistently see the same new state
            self.state_confidence = max(0, self.state_confidence - 1)

        self._check_rep_completion()

        return self._get_status()

    def _check_transitions(self, pose_data: Dict[str, Any]) -> Optional[str]:
        """Check for state transitions based on pose data"""
        angles = pose_data.get('angles', {})
        positions = pose_data.get('positions', {})
        for state_name, state_obj in self.states.items():
            if self.current_state == state_name:
                continue # Skip current state
            if self._evaluate_state_conditions(state_obj.conditions, angles, positions):
                return state_name
        return None

    def _evaluate_state_conditions(self, conditions: Dict[str, Any], angles: Dict, positions: Dict) -> bool:
        """Generisch: prüfe alle Bedingungen (min/max Winkel, Positionen)"""
        try:
            for key, val in conditions.items():
                if key.endswith('angle'):
                    joint = key.replace('min_', '').replace('max_', '').replace('_angle', '')
                    angle = angles.get(joint, 1000)
                    print("Current angle for", joint, "is", angle)
                    if angle is not 1000: # we'll find something smarter later...
                        if key.startswith('min_') and angle < val:
                            return False
                        if key.startswith('max_') and angle > val:
                            return False
                elif key.endswith('position'):
                    joint = key.replace('min_', '').replace('max_', '').replace('_position', '')
                    pos = positions.get(joint, {}).get('y', 1000) # same here
                    if pos is not 1000:
                        if key.startswith('min_') and pos < val:
                            return False
                        if key.startswith('max_') and pos > val:
                            return False
            return True
        except Exception as e:
            print(f"Error evaluating conditions: {e}")
            return False

    def _transition_to_state(self, new_state: str):
        """Transition to a new state"""
        self.state_history.append({
            'from': self.current_state,
            'to': new_state,
            'frame': self.frame_time,
            'duration': self.frames_in_current_state
        })

        print(f"State transition: {self.current_state} -> {new_state} (frame {self.frame_time})")

        self.current_state = new_state
        self.frames_in_current_state = 0
        self.state_confidence = 0

        if new_state in self.states:
            self.states[new_state].entry_time = self.frame_time

    def _check_rep_completion(self):
        """Checks if the current state matches the expected rep pattern"""
        if not self.rep_pattern or len(self.rep_pattern) < 2:
            return
        if self.frame_time - self.last_rep_frame < self.min_frames_between_reps:
            return

        expected_state = self.rep_pattern[self.pattern_progress]

        if self.current_state == expected_state:
            self.pattern_progress += 1
            if self.pattern_progress == len(self.rep_pattern):
                self.rep_count += 1
                self.last_rep_frame = self.frame_time
                self.pattern_progress = 0
                print(f"🎉 REP COMPLETED! Count: {self.rep_count} (Pattern: {' → '.join(self.rep_pattern)})")
        elif self.current_state == self.rep_pattern[0]:
            self.pattern_progress = 1
        elif self.current_state not in self.rep_pattern:
            self.pattern_progress = 0

    def _get_status(self) -> Dict[str, Any]:
        """Get current state machine status"""
        return {
            'current_state': self.current_state,
            'rep_count': self.rep_count,
            'state_history': self.state_history[-3:],  # Last 3 transitions
            'frame_time': self.frame_time,
            'frames_in_state': self.frames_in_current_state,
            'state_confidence': self.state_confidence,
            'sequence_progress': self.pattern_progress
        }

    def reset(self):
        """Reset state machine"""
        self.current_state = 'standing'
        self.rep_count = 0
        self.state_history = []
        self.frame_time = 0
        self.frames_in_current_state = 0
        self.state_confidence = 0
        self.pattern_progress = 0
        self.last_rep_frame = 0
        print("state machine reset")
