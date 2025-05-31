"""
Hilfsfunktionen für den AI-Coach.
"""

import os
import json
import numpy as np
import cv2
import time
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime

def create_directory_if_not_exists(directory: str):
    """
    Erstellt ein Verzeichnis, falls es nicht existiert.

    Args:
        directory: Pfad zum Verzeichnis
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Verzeichnis erstellt: {directory}")

def save_to_json(data: Dict[str, Any], filepath: str):
    """
    Speichert Daten im JSON-Format.

    Args:
        data: Zu speichernde Daten
        filepath: Pfad zur JSON-Datei
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Daten gespeichert in: {filepath}")

def load_from_json(filepath: str) -> Dict[str, Any]:
    """
    Lädt Daten aus einer JSON-Datei.

    Args:
        filepath: Pfad zur JSON-Datei

    Returns:
        Geladene Daten als Dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def visualize_pose_sequence(pose_sequence: np.ndarray,
                          frame_indices: Optional[List[int]] = None,
                          figsize: Tuple[int, int] = (20, 10)):
    """
    Visualisiert eine Sequenz von Posen.

    Args:
        pose_sequence: Array von Pose-Daten mit Form [frames, keypoints, coordinates]
        frame_indices: Indizes der zu visualisierenden Frames (optional)
        figsize: Größe der Abbildung
    """
    if frame_indices is None:
        # Wähle einige Frames gleichmäßig verteilt aus der Sequenz
        n_frames = len(pose_sequence)
        num_samples = min(8, n_frames)
        frame_indices = [int(i * n_frames / num_samples) for i in range(num_samples)]

    fig, axes = plt.subplots(1, len(frame_indices), figsize=figsize)

    # Wenn nur ein Frame ausgewählt wurde
    if len(frame_indices) == 1:
        axes = [axes]

    for i, frame_idx in enumerate(frame_indices):
        if frame_idx < len(pose_sequence):
            pose = pose_sequence[frame_idx]

            # Plotte die Verbindungen zwischen den Keypoints
            # Dies ist eine vereinfachte Version - für eine vollständige Implementierung
            # sollte die MediaPipe-Pose-Topologie verwendet werden

            # Plotte die Keypoints
            axes[i].scatter(pose[:, 0], -pose[:, 1], c=pose[:, 2], cmap='viridis',
                            s=20, alpha=0.7)

            # Verbinde einige Keypoints (stark vereinfacht)
            # In einer vollständigen Implementierung würden alle Verbindungen der
            # MediaPipe-Pose-Topologie gezeichnet werden

            # Setze die Achsengrenzen
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(-1, 0)
            axes[i].set_title(f"Frame {frame_idx}")
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def sliding_window(pose_data: List[np.ndarray],
                window_size: int,
                stride: int = 1) -> List[np.ndarray]:
    """
    Erstellt überlappende Fenster aus einer Liste von Pose-Daten.

    Args:
        pose_data: Liste von Pose-Daten für jeden Frame
        window_size: Größe des Fensters (Anzahl der Frames)
        stride: Schrittweite für das Fenster

    Returns:
        Liste von Fenstern (Sequenzen)
    """
    windows = []
    n_frames = len(pose_data)

    if n_frames < window_size:
        # Wenn die Anzahl der Frames kleiner als die Fenstergröße ist,
        # fülle das Fenster mit Nullen auf
        window = pose_data + [np.zeros_like(pose_data[0])] * (window_size - n_frames)
        windows.append(np.array(window))
    else:
        # Erstelle überlappende Fenster
        for i in range(0, n_frames - window_size + 1, stride):
            window = pose_data[i:i+window_size]
            windows.append(np.array(window))

    return windows

def draw_feedback_on_frame(frame: np.ndarray,
                         feedback_text: str,
                         position: Tuple[int, int] = (30, 30),
                         font_scale: float = 0.7,
                         color: Tuple[int, int, int] = (0, 255, 0),
                         thickness: int = 2) -> np.ndarray:
    """
    Zeichnet Feedback-Text auf einen Frame.

    Args:
        frame: Der Frame, auf dem gezeichnet werden soll
        feedback_text: Der zu zeichnende Text
        position: Position des Textes (x, y)
        font_scale: Skalierungsfaktor für die Schriftgröße
        color: Textfarbe (B, G, R)
        thickness: Linienstärke des Textes

    Returns:
        Frame mit gezeichnetem Text
    """
    # Kopiere den Frame, um Seiteneffekte zu vermeiden
    frame_copy = frame.copy()

    # Teile den Text in Zeilen auf, wenn er zu lang ist
    max_width = frame.shape[1] - 60  # Etwas Abstand zum Rand
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Berechne die Breite des Textes
    text_size = cv2.getTextSize(feedback_text, font, font_scale, thickness)[0]

    if text_size[0] > max_width:
        words = feedback_text.split(' ')
        lines = []
        current_line = words[0]

        for word in words[1:]:
            test_line = current_line + ' ' + word
            test_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]

            if test_size[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)

        # Zeichne jede Zeile
        y_position = position[1]
        for line in lines:
            cv2.putText(frame_copy, line, (position[0], y_position),
                      font, font_scale, color, thickness)
            y_position += int(text_size[1] * 1.5)  # Zeilenabstand
    else:
        # Zeichne den gesamten Text in einer Zeile
        cv2.putText(frame_copy, feedback_text, position,
                  font, font_scale, color, thickness)

    return frame_copy

def get_timestamp() -> str:
    """
    Gibt einen formatierten Zeitstempel zurück.

    Returns:
        Zeitstempel im Format YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def initialize_video_writer(input_stream,
                          output_path: str,
                          fourcc_code: str = 'mp4v') -> cv2.VideoWriter:
    """
    Initialisiert einen VideoWriter für die Ausgabe.

    Args:
        input_stream: Eingabe-Videostream (z.B. cv2.VideoCapture)
        output_path: Pfad für die Ausgabedatei
        fourcc_code: FourCC-Code für den Codec

    Returns:
        Initialisierter VideoWriter
    """
    width = int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_stream.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Berechnet die Frames pro Sekunde.

    Args:
        start_time: Startzeit der Messung
        frame_count: Anzahl der verarbeiteten Frames

    Returns:
        FPS-Wert
    """
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    return fps
