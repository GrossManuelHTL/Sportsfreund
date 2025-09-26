import time
from typing import Dict, List, Any, Callable, Optional

from . import FeedbackHandler
from ...models.feedback import FeedbackType, FeedbackItem
from collections import deque, defaultdict
from statistics import mean

class ErrorChecker:
    """Simple generic rule-based error checker.

    - Uses exercise_config.error_checks to evaluate simple conditions per-frame.
    - Supports smoothing (rolling mean) and debounce to avoid noisy reports.
    - Collects errors per-rep and aggregates per-set.
    - Ignores 'safety' special handling; treats all checks as warnings/errors.
    """

    def __init__(self, exercise_config: Dict[str, Any], smooth_window: int = 5, debounce_frames: int = 4):
        self.exercise_name = exercise_config.get('name', 'unknown')
        self.error_checks = exercise_config.get('error_checks', [])
        # normalize checks: ensure each has an id
        self.checks = []
        for idx, c in enumerate(self.error_checks):
            cid = c.get('name') or c.get('message') or f'check_{idx}'
            normalized = dict(c)
            normalized['id'] = cid
            # Downgrade any explicit 'safety' type to 'warning' to match user preference
            if normalized.get('type') == 'safety':
                normalized['type'] = 'warning'
            self.checks.append(normalized)

        # smoothing and debounce
        self.smooth_window = smooth_window
        self.debounce_frames = debounce_frames

        # per-angle history for smoothing
        self.angle_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.smooth_window))
        # debounce counters per check id
        self.debounce_counters: Dict[str, int] = defaultdict(int)
        # flags to avoid repeated reporting within the same rep
        self.reported_in_rep: Dict[str, bool] = defaultdict(bool)

        # per-rep and per-set storage
        self.current_rep_errors: set = set()
        self.reps: List[Dict[str, Any]] = []  # each rep: { 'errors': [ids], 'duration': float }
        self.sets: List[Dict[str, Any]] = []  # each set: { 'reps': [...], 'aggregated_errors': {...} }

        # counters across session
        self.error_counts_total: Dict[str, int] = defaultdict(int)

        self.feedback_handler: Optional[FeedbackHandler] = None

    def set_feedback_handler(self, feedback_handler: FeedbackHandler):
        """Attach a FeedbackHandler instance for optional delivery."""
        self.feedback_handler = feedback_handler

    def _smooth_angle(self, name: str, value: float) -> float:
        """Append value to history and return rolling mean."""
        try:
            self.angle_history[name].append(value)
            # compute mean of available samples
            vals = list(self.angle_history[name])
            if not vals:
                return value
            return mean(vals)
        except Exception:
            return value

    def _evaluate_condition(self, condition: Dict[str, Any], angles: Dict[str, float], positions: Dict[str, Any]) -> bool:
        """Evaluate a single condition. Return True if condition indicates a problem."""
        try:
            ctype = condition.get('type')
            if ctype == 'angle_range':
                name = condition['angle']
                min_v = condition.get('min', 0)
                max_v = condition.get('max', 180)
                current = angles.get(name)
                if current is None:
                    return False
                # use smoothed value
                current = self._smooth_angle(name, float(current))
                # problem if outside [min, max]
                return not (min_v <= current <= max_v)

            if ctype == 'angle_threshold':
                name = condition['joint'] or condition.get('angle')
                op = condition.get('operator', '>')
                threshold = condition.get('threshold')
                current = angles.get(name)
                if current is None or threshold is None:
                    return False
                current = self._smooth_angle(name, float(current))
                if op == '>':
                    return current > threshold
                if op == '<':
                    return current < threshold
                if op == '>=':
                    return current >= threshold
                if op == '<=':
                    return current <= threshold

            if ctype == 'position_threshold':
                pname = condition['position']
                axis = condition.get('axis', 'y')
                threshold = condition.get('threshold')
                op = condition.get('operator', '>')
                pos = positions.get(pname, {})
                current = pos.get(axis) if isinstance(pos, dict) else None
                if current is None or threshold is None:
                    return False
                if op == '>':
                    return current > threshold
                if op == '<':
                    return current < threshold
                if op == '>=':
                    return current >= threshold
                if op == '<=':
                    return current <= threshold

            if ctype == 'angle_difference':
                j1 = condition['joint1']
                j2 = condition['joint2']
                max_diff = condition.get('max_difference', 999)
                a1 = angles.get(j1)
                a2 = angles.get(j2)
                if a1 is None or a2 is None:
                    return False
                a1 = self._smooth_angle(j1, float(a1))
                a2 = self._smooth_angle(j2, float(a2))
                return abs(a1 - a2) > max_diff

            # unknown condition types are ignored
            return False
        except Exception:
            return False

    def process_frame(self, pose_data: Dict[str, Any], state: Optional[str] = None) -> List[FeedbackItem]:
        """Process a single frame of pose data.

        Returns a list of newly generated FeedbackItems for this frame (debounced).
        """
        if not pose_data:
            return []

        angles = pose_data.get('angles', {})
        positions = pose_data.get('positions', {})

        # collect debounced candidates (not yet reported in this rep)
        candidates: List[Dict[str, Any]] = []
        for check in self.checks:
            cid = check['id']
            condition = check.get('condition', {})

            problem = self._evaluate_condition(condition, angles, positions)

            if problem:
                self.debounce_counters[cid] += 1
            else:
                self.debounce_counters[cid] = 0

            if self.debounce_counters[cid] >= self.debounce_frames and not self.reported_in_rep[cid]:
                candidates.append({'check': check, 'cid': cid, 'priority': int(check.get('priority', 1))})

        if not candidates:
            return []

        # choose highest priority candidate to report now
        candidates.sort(key=lambda x: x['priority'], reverse=True)
        chosen = candidates[0]

        # mark as reported and update counts; queue non-chosen as pending_feedback
        chosen_item: Optional[FeedbackItem] = None
        for c in candidates:
            cid = c['cid']
            check = c['check']
            msg = check.get('message', cid)
            ftype = FeedbackType.WARNING if check.get('type') != 'error' else FeedbackType.ERROR

            # record occurrence
            self.current_rep_errors.add(cid)
            self.reported_in_rep[cid] = True
            self.error_counts_total[cid] += 1

            item = FeedbackItem(msg, ftype, priority=check.get('priority', 1))
            # chosen -> immediate delivery via text/custom callbacks
            if c is chosen:
                chosen_item = item
                if self.feedback_handler:
                    self.feedback_handler.add_feedback(msg, ftype, immediate=True, priority=check.get('priority', 1))
            else:
                # queue others for later delivery
                if self.feedback_handler:
                    self.feedback_handler.add_feedback(msg, ftype, immediate=False, priority=check.get('priority', 1))

        return [chosen_item] if chosen_item else []

    def on_rep_end(self, duration: float = 0.0):
        """Call when a repetition ends. Store collected errors for the rep and reset per-rep state."""
        self.reps.append({
            'errors': list(self.current_rep_errors),
            'duration': duration,
            'timestamp': time.time()
        })
        # reset per-rep structures
        self.current_rep_errors.clear()
        self.reported_in_rep = defaultdict(bool)
        self.debounce_counters = defaultdict(int)

    def on_set_end(self):
        """Call when a set ends. Aggregate reps into a set summary and return a summary JSON dict."""
        if not self.reps:
            summary = {'reps': [], 'aggregated_errors': {}}
            self.sets.append(summary)
            return summary

        # aggregate errors across reps in this set
        error_counts: Dict[str, int] = defaultdict(int)
        for rep in self.reps:
            for e in rep.get('errors', []):
                error_counts[e] += 1

        # build human readable messages for errors that occur in at least 1 rep
        aggregated = {}
        total_reps = len(self.reps)
        messages = []
        for cid, count in error_counts.items():
            # find check definition
            check = next((c for c in self.checks if c['id'] == cid), None)
            msg = check.get('message') if check else cid
            aggregated[cid] = {'count': count, 'message': msg}
            # include in spoken feedback if occurs in >=30% reps or at least 2 reps
            if count >= max(2, int(0.3 * total_reps)):
                messages.append(f"Bei {count}/{total_reps} Wiederholungen: {msg}")

        if not messages:
            messages.append("✅ Gute Form! Keine häufigen Fehler erkannt.")

        summary = {
            'reps': list(self.reps),
            'aggregated_errors': aggregated,
            'feedback_texts': messages,
            'timestamp': time.time()
        }

        self.sets.append(summary)
        # reset reps for next set
        self.reps = []
        return summary

    def get_session_summary(self) -> Dict[str, Any]:
        """Return a summary for the whole session (all sets)."""
        return {
            'exercise': self.exercise_name,
            'sets': list(self.sets),
            'totals': dict(self.error_counts_total),
            'generated_at': time.time()
        }

    def get_final_feedback(self) -> List[str]:
        """Produce concise final feedback messages based on accumulated error counts.

        Returns a list of German feedback strings suitable for display or TTS.
        """
        messages: List[str] = []

        if not self.error_counts_total:
            return ["✅ Gute Form! Keine großen Probleme erkannt."]

        for cid, count in sorted(self.error_counts_total.items(), key=lambda x: x[1], reverse=True):
            # find check definition
            check = next((c for c in self.checks if c['id'] == cid), None)
            msg = check.get('message') if check else cid

            # simple thresholds for verbosity
            if count > 60:
                messages.append(f"🔴 Häufig: {msg} ({count}x)")
            elif count > 40:
                messages.append(f"🟡 Gelegentlich: {msg} ({count}x)")
            elif count > 10:
                messages.append(f"🔵 Selten: {msg} ({count}x)")
            else:
                # very rare issues - include only if no other messages
                messages.append(f"ℹ️ {msg} ({count}x)")

        if not messages:
            messages.append("✅ Gute Form! Keine großen Probleme erkannt.")

        return messages

    def reset_all(self):
        """Reset the entire checker state."""
        self.angle_history = defaultdict(lambda: deque(maxlen=self.smooth_window))
        self.debounce_counters = defaultdict(int)
        self.reported_in_rep = defaultdict(bool)
        self.current_rep_errors = set()
        self.reps = []
        self.sets = []
        self.error_counts_total = defaultdict(int)

