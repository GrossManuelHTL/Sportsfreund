"""
Hauptdatei für den AI-Coach mit Live-Feedback.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple

# Importiere aus dem lokalen Modul
from .config import (
    DATA_DIR, SAMPLES_DIR, MODEL_DIR, VIDEO_WIDTH, VIDEO_HEIGHT,
    SEQUENCE_LENGTH, CAMERA_ID, DISPLAY_POSE, DISPLAY_FEEDBACK
)
from .pose_extraction import PoseExtractor
from .model import SequenceLabelingModel
from .feedback import FeedbackGenerator
from .utils import (
    create_directory_if_not_exists, get_timestamp,
    sliding_window, draw_feedback_on_frame,
    initialize_video_writer, calculate_fps
)

def parse_args():
    """
    Verarbeitet die Kommandozeilenargumente.

    Returns:
        Geparste Argumente
    """
    parser = argparse.ArgumentParser(description='AI-Coach mit Live-Feedback')

    parser.add_argument('--mode', type=str, default='webcam',
                      choices=['webcam', 'video', 'train'],
                      help='Betriebsmodus: webcam, video oder train')

    parser.add_argument('--exercise', type=str, default='squat',
                      choices=['squat', 'pushup'],
                      help='Übungstyp für die Analyse')

    parser.add_argument('--video', type=str,
                      help='Pfad zum Eingabevideo (nur im Video-Modus)')

    parser.add_argument('--output', type=str,
                      help='Pfad zum Speichern des Ausgabevideos (optional)')

    parser.add_argument('--camera', type=int, default=CAMERA_ID,
                      help='Kamera-ID für Webcam-Modus (default: 0)')

    parser.add_argument('--model', type=str, default='exercise_model',
                      help='Name des zu verwendenden Modells')

    parser.add_argument('--train_data', type=str,
                      help='Pfad zum Trainings-Datensatz (nur im Train-Modus)')

    args = parser.parse_args()

    # Einfache Validierung
    if args.mode == 'video' and args.video is None:
        parser.error("Im Video-Modus muss --video angegeben werden.")

    if args.mode == 'train' and args.train_data is None:
        parser.error("Im Train-Modus muss --train_data angegeben werden.")

    return args

def load_training_data(data_path: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    Lädt die Trainingsdaten und deren Annotationen.

    Diese Funktion unterstützt zwei Modi:
    1. Laden von Videos und zugehörigen Annotations-Dateien (JSON)
    2. Direktes Laden von Videos mit automatischer Annotation

    Args:
        data_path: Pfad zum Trainingsdatensatz (Verzeichnis oder spezifische Datei)

    Returns:
        Tuple mit:
            - Liste von Pose-Sequenzen
            - Liste entsprechender Labels
    """
    print(f"Lade Trainingsdaten von: {data_path}")

    sequences = []
    labels = []
    pose_extractor = PoseExtractor()

    # Prüfe, ob der Pfad ein Verzeichnis ist
    if os.path.isdir(data_path):
        # Sammle alle Video-Dateien (.mp4) und Annotations-Dateien (.json)
        video_files = []
        annotation_files = {}

        for root, _, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.mp4'):
                    video_files.append(file_path)
                elif file.endswith('.json') and 'annotation' in file.lower():
                    # Extrahiere den Basisnamen des Videos, für das diese Annotation gilt
                    base_name = os.path.splitext(file)[0].replace('_annotation', '')
                    annotation_files[base_name] = file_path

        # Verarbeite jedes Videomaterial
        for video_path in video_files:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"Verarbeite Video: {video_path}")

            # Extrahiere Pose-Daten aus dem Video
            pose_data = pose_extractor.process_video(video_path)

            # Suche nach einer Annotations-Datei für dieses Video
            annotation_path = None
            for key in annotation_files:
                if key in base_name:
                    annotation_path = annotation_files[key]
                    break

            # Verarbeite das Video basierend auf den verfügbaren Annotationen
            if annotation_path:
                # Wenn eine Annotations-Datei gefunden wurde, lade diese
                print(f"Lade Annotationen aus: {annotation_path}")
                try:
                    with open(annotation_path, 'r') as f:
                        annotations = json.load(f)

                    # Verarbeite die Frames basierend auf den Annotationen
                    # Angenommen, die Annotationen haben das Format:
                    # { "frames": [{"frame_idx": 0, "label": 0}, {"frame_idx": 10, "label": 1}, ...] }

                    if 'frames' in annotations:
                        frame_annotations = annotations['frames']
                        labeled_frames = {}

                        # Konvertiere in ein Dictionary für schnelleren Zugriff
                        for anno in frame_annotations:
                            labeled_frames[anno['frame_idx']] = anno['label']

                        # Teile die Pose-Daten in überlappende Fenster auf
                        windows = sliding_window(pose_data, SEQUENCE_LENGTH, stride=10)

                        # Weise jedem Fenster ein Label zu (verwende das Mehrheitslabel)
                        for window_idx, window in enumerate(windows):
                            start_frame = window_idx * 10
                            end_frame = start_frame + SEQUENCE_LENGTH
                            window_labels = []

                            for i in range(start_frame, end_frame):
                                if i in labeled_frames:
                                    window_labels.append(labeled_frames[i])

                            if window_labels:
                                # Verwende das häufigste Label als Fensterlabel
                                from collections import Counter
                                label = Counter(window_labels).most_common(1)[0][0]
                                sequences.append(window)
                                labels.append(label)
                    else:
                        print(f"Warnung: Unbekanntes Annotations-Format in {annotation_path}")

                except Exception as e:
                    print(f"Fehler beim Laden der Annotationen: {str(e)}")
                    # Wenn die Annotationen nicht geladen werden können, fahre mit automatischer Annotation fort
                    print("Verwende automatische Annotation als Fallback.")
                    _auto_annotate_video(pose_data, sequences, labels)
            else:
                # Wenn keine Annotations-Datei gefunden wurde, verwende automatische Annotation
                print(f"Keine Annotationsdatei für {video_path} gefunden. Verwende automatische Annotation.")
                _auto_annotate_video(pose_data, sequences, labels)

    elif os.path.isfile(data_path):
        # Wenn ein einzelnes Video angegeben ist
        if data_path.endswith('.mp4'):
            print(f"Verarbeite einzelnes Video: {data_path}")
            pose_data = pose_extractor.process_video(data_path)

            # Suche nach einer Annotations-Datei im gleichen Verzeichnis
            base_name = os.path.splitext(os.path.basename(data_path))[0]
            annotation_path = os.path.join(os.path.dirname(data_path), f"{base_name}_annotation.json")

            if os.path.exists(annotation_path):
                # Lade Annotationen und verarbeite sie wie oben
                print(f"Lade Annotationen aus: {annotation_path}")
                try:
                    with open(annotation_path, 'r') as f:
                        annotations = json.load(f)

                    if 'frames' in annotations:
                        # Verarbeite ähnlich wie oben
                        # ...
                        pass
                except Exception as e:
                    print(f"Fehler beim Laden der Annotationen: {str(e)}")
                    _auto_annotate_video(pose_data, sequences, labels)
            else:
                print(f"Keine Annotationsdatei gefunden. Verwende automatische Annotation.")
                _auto_annotate_video(pose_data, sequences, labels)
        else:
            print(f"Warnung: Unbekanntes Dateiformat: {data_path}")

    print(f"Trainings-Datensatz erstellt: {len(sequences)} Sequenzen mit {len(labels)} Labels")
    return sequences, labels

def _auto_annotate_video(pose_data: List[np.ndarray], sequences: List[np.ndarray], labels: List[int]):
    """
    Hilfsfunktion zur automatischen Annotation von Videos basierend auf einfachen Heuristiken.
    Dies ist ein Fallback, wenn keine expliziten Annotationen vorhanden sind.

    Die Funktion erkennt Bewegungsmuster und klassifiziert die Frames in:
    - Pause (0): Wenig Bewegung
    - Wiederholung läuft (1): Deutliche Bewegung erkannt
    - Wiederholung beendet (2): Übergang von Bewegung zu Pause

    Args:
        pose_data: Liste der Pose-Daten für jeden Frame
        sequences: Liste, zu der die Sequenzen hinzugefügt werden
        labels: Liste, zu der die Labels hinzugefügt werden
    """
    # Einfache Bewegungserkennung: Berechne die Bewegung zwischen aufeinanderfolgenden Frames
    movement_scores = []

    # Berechne Bewegungsscores zwischen aufeinanderfolgenden Frames
    for i in range(1, len(pose_data)):
        # Berechne den durchschnittlichen Bewegungsscore für Unterkörper-Keypoints (Hüfte, Knie, Füße)
        prev_frame = pose_data[i-1]
        curr_frame = pose_data[i]

        # Relevante Keypoints (Hüfte, Knie, Knöchel)
        keypoints = [23, 24, 25, 26, 27, 28]  # Indizes für Hüfte, Knie, Knöchel

        # Berechne durchschnittliche Bewegung für diese Keypoints
        movement = 0
        valid_points = 0

        for kp in keypoints:
            # Prüfe, ob der Keypoint valide ist (anhand der Visibility)
            if prev_frame[kp, 2] > 0.5 and curr_frame[kp, 2] > 0.5:
                # Berechne euklidischen Abstand der x,y-Koordinaten
                dist = np.sqrt(np.sum((prev_frame[kp, :2] - curr_frame[kp, :2])**2))
                movement += dist
                valid_points += 1

        if valid_points > 0:
            movement_scores.append(movement / valid_points)
        else:
            movement_scores.append(0)

    # Füge eine 0 am Anfang hinzu, da wir bei i=1 beginnen
    movement_scores.insert(0, 0)

    # Glätte die Bewegungsscores mit einem gleitenden Durchschnitt
    window_size = 5
    smoothed_scores = np.convolve(movement_scores, np.ones(window_size)/window_size, mode='same')

    # Definiere Schwellenwerte für die Bewegungserkennung
    movement_threshold = np.mean(smoothed_scores) * 1.5

    # Klassifiziere jeden Frame
    frame_labels = []
    previous_state = 0  # Startzustand ist Pause

    for i, score in enumerate(smoothed_scores):
        if score > movement_threshold:
            # Signifikante Bewegung erkannt -> Wiederholung läuft
            current_state = 1
        else:
            # Wenig Bewegung
            if previous_state == 1:
                # Übergang von Bewegung zu Pause -> Wiederholung beendet
                current_state = 2
            else:
                # Weiterhin in Pause
                current_state = 0

        frame_labels.append(current_state)
        previous_state = 1 if current_state == 1 else 0  # Setze previous_state, behandle "Wiederholung beendet" als temporären Zustand

    # Teile die Pose-Daten in überlappende Fenster auf
    windows = sliding_window(pose_data, SEQUENCE_LENGTH, stride=10)

    # Weise jedem Fenster ein Label zu (nehme das Label des letzten Frames)
    for window_idx, window in enumerate(windows):
        start_frame = window_idx * 10
        end_frame = min(start_frame + SEQUENCE_LENGTH, len(frame_labels))

        if end_frame > start_frame:
            # Überprüfe, ob die Wiederholung im Fenster beendet wird
            if 2 in frame_labels[start_frame:end_frame]:
                label = 2  # Wiederholung beendet
            elif 1 in frame_labels[start_frame:end_frame]:
                label = 1  # Wiederholung läuft
            else:
                label = 0  # Pause

            sequences.append(window)
            labels.append(label)

def train_model(args):
    """
    Trainiert das Modell mit den angegebenen Daten.

    Args:
        args: Kommandozeilenargumente
    """
    print("Starte Modelltraining...")

    # Lade Trainingsdaten
    sequences, labels = load_training_data(args.train_data)

    if not sequences:
        print("Keine Trainingsdaten gefunden.")
        return

    print(f"Geladene Daten: {len(sequences)} Sequenzen")

    # Teile in Trainings- und Validierungsdaten
    from sklearn.model_selection import train_test_split
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )

    # Initialisiere und trainiere das Modell
    model = SequenceLabelingModel(model_name=args.model, use_saved_model=False)

    # Optional: Berechne Klassengewichte für unausgewogene Daten
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    unique_classes = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_labels)
    class_weight_dict = {i: weight for i, weight in zip(unique_classes, class_weights)}

    # Trainiere das Modell
    history = model.train(
        train_sequences, train_labels,
        val_sequences, val_labels,
        class_weights=class_weight_dict
    )

    print("Training abgeschlossen!")

def process_video(args):
    """
    Verarbeitet ein Video und gibt Feedback.

    Args:
        args: Kommandozeilenargumente
    """
    print(f"Verarbeite Video: {args.video}")

    # Initialisiere Komponenten
    pose_extractor = PoseExtractor()
    model = SequenceLabelingModel(model_name=args.model)
    feedback_generator = FeedbackGenerator()
    feedback_generator.set_exercise_type(args.exercise)

    # Öffne das Video
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Fehler beim Öffnen des Videos: {args.video}")
        return

    # Video-Eigenschaften
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialisiere Videoschreiber, wenn ein Ausgabepfad angegeben ist
    output_writer = None
    if args.output:
        output_writer = initialize_video_writer(cap, args.output)

    # Verarbeitungs-Loop
    frame_idx = 0
    pose_buffer = []
    feedback_text = ""
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extrahiere Pose-Daten
            frame_with_landmarks, pose_data = pose_extractor.extract_pose_from_frame(frame)

            # Füge die Pose-Daten zum Puffer hinzu
            pose_buffer.append(pose_data)

            # Wenn genügend Frames im Puffer sind, führe eine Vorhersage durch
            if len(pose_buffer) >= SEQUENCE_LENGTH:
                # Nehme die letzten SEQUENCE_LENGTH Frames
                current_sequence = np.array(pose_buffer[-SEQUENCE_LENGTH:])

                # Führe die Vorhersage durch
                prediction = model.predict_sequence(current_sequence)[0]  # Nehme den aktuellen Frame

                # Generiere Feedback
                current_time = time.time()
                feedback = feedback_generator.process_frame_prediction(
                    prediction, pose_data, current_time
                )

                if feedback:
                    feedback_text = feedback_generator.get_formatted_feedback(feedback)
                    print(f"Feedback: {feedback_text}")

            # Zeichne Feedback auf den Frame, wenn vorhanden
            if DISPLAY_FEEDBACK and feedback_text:
                frame_with_landmarks = draw_feedback_on_frame(
                    frame_with_landmarks, feedback_text
                )

            # Zeige den Frame an, wenn gewünscht
            if DISPLAY_POSE:
                cv2.imshow('AI-Coach', frame_with_landmarks)

            # Schreibe den Frame, wenn ein Ausgabepfad angegeben ist
            if output_writer:
                output_writer.write(frame_with_landmarks)

            # Fortschrittsanzeige
            frame_idx += 1
            if frame_idx % 100 == 0:
                progress = frame_idx / total_frames * 100
                fps_value = calculate_fps(start_time, frame_idx)
                print(f"Fortschritt: {progress:.1f}% ({frame_idx}/{total_frames}) - {fps_value:.2f} FPS")

            # Beenden, wenn 'q' gedrückt wird
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if output_writer:
            output_writer.release()
        cv2.destroyAllWindows()

    # Zeige eine Zusammenfassung
    summary = feedback_generator.get_summary()
    print("\nZusammenfassung:")
    print(f"Übung: {summary['exercise_name']}")
    print(f"Gesamtzahl der Wiederholungen: {summary['total_repetitions']}")
    print(f"Durchschnittliche Qualität: {summary['average_quality']:.1f}%")

    if summary['common_errors']:
        print("Häufigste Fehler:")
        for error, count in summary['common_errors']:
            print(f"- {error}: {count}x")

def process_webcam(args):
    """
    Verarbeitet den Webcam-Stream und gibt Live-Feedback.

    Args:
        args: Kommandozeilenargumente
    """
    print(f"Starte Webcam-Modus mit Kamera ID: {args.camera}")

    # Initialisiere Komponenten
    pose_extractor = PoseExtractor()
    model = SequenceLabelingModel(model_name=args.model)
    feedback_generator = FeedbackGenerator()
    feedback_generator.set_exercise_type(args.exercise)

    # Öffne die Webcam
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"Fehler beim Öffnen der Webcam mit ID: {args.camera}")
        return

    # Setze Webcam-Eigenschaften
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

    # Initialisiere Videoschreiber, wenn ein Ausgabepfad angegeben ist
    output_writer = None
    if args.output:
        timestamp = get_timestamp()
        if not args.output.endswith('.mp4'):
            output_path = f"{args.output}_{args.exercise}_{timestamp}.mp4"
        else:
            output_path = args.output
        output_writer = initialize_video_writer(cap, output_path)

    # Verarbeitungs-Loop
    frame_idx = 0
    pose_buffer = []
    feedback_text = ""
    start_time = time.time()

    try:
        print("Drücke 'q' zum Beenden")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extrahiere Pose-Daten
            frame_with_landmarks, pose_data = pose_extractor.extract_pose_from_frame(frame)

            # Füge die Pose-Daten zum Puffer hinzu
            pose_buffer.append(pose_data)

            # Behalte nur die neuesten SEQUENCE_LENGTH Frames
            if len(pose_buffer) > SEQUENCE_LENGTH:
                pose_buffer.pop(0)

            # Wenn genügend Frames im Puffer sind, führe eine Vorhersage durch
            if len(pose_buffer) == SEQUENCE_LENGTH:
                # Nehme die letzten SEQUENCE_LENGTH Frames
                current_sequence = np.array(pose_buffer)

                # Führe die Vorhersage durch
                prediction = model.predict_sequence(current_sequence)[-1]  # Nehme den letzten Frame

                # Generiere Feedback
                current_time = time.time()
                feedback = feedback_generator.process_frame_prediction(
                    prediction, pose_data, current_time
                )

                if feedback:
                    feedback_text = feedback_generator.get_formatted_feedback(feedback)
                    print(f"Feedback: {feedback_text}")

            # Zeichne Feedback auf den Frame, wenn vorhanden
            if DISPLAY_FEEDBACK and feedback_text:
                frame_with_landmarks = draw_feedback_on_frame(
                    frame_with_landmarks, feedback_text
                )

            # FPS anzeigen
            if frame_idx % 30 == 0:
                fps_value = calculate_fps(start_time, frame_idx)
                cv2.putText(
                    frame_with_landmarks, f"FPS: {fps_value:.1f}",
                    (10, frame_with_landmarks.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )

            # Zeige den Frame an
            cv2.imshow('AI-Coach', frame_with_landmarks)

            # Schreibe den Frame, wenn ein Ausgabepfad angegeben ist
            if output_writer:
                output_writer.write(frame_with_landmarks)

            frame_idx += 1

            # Beenden, wenn 'q' gedrückt wird
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if output_writer:
            output_writer.release()
        cv2.destroyAllWindows()

    # Zeige eine Zusammenfassung
    summary = feedback_generator.get_summary()
    print("\nZusammenfassung:")
    print(f"Übung: {summary['exercise_name']}")
    print(f"Gesamtzahl der Wiederholungen: {summary['total_repetitions']}")
    print(f"Durchschnittliche Qualität: {summary['average_quality']:.1f}%")

    if summary['common_errors']:
        print("Häufigste Fehler:")
        for error, count in summary['common_errors']:
            print(f"- {error}: {count}x")

def main():
    """
    Hauptfunktion des Programms.
    """
    # Parse Kommandozeilenargumente
    args = parse_args()

    # Stelle sicher, dass die erforderlichen Verzeichnisse existieren
    create_directory_if_not_exists(DATA_DIR)
    create_directory_if_not_exists(SAMPLES_DIR)
    create_directory_if_not_exists(MODEL_DIR)

    # Je nach Modus die entsprechende Funktion aufrufen
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'video':
        process_video(args)
    else:  # webcam
        process_webcam(args)

if __name__ == "__main__":
    main()
