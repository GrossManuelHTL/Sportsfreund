import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from pose_extractor import PoseExtractor
from exercise_manager import ExerciseManager


class ExerciseModelTrainer:
    def __init__(self, exercise_name):
        """
        Initialisiert den Trainer für das Übungsmodell.

        Args:
            exercise_config: Pfad zur Exercise-Konfig oder Konfig-Dictionary
        """
        self.manager = ExerciseManager()
        exercise_config = self.manager.get_exercise_config(exercise_name)

        self.pose_extractor = PoseExtractor(exercise_config)


        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, self.pose_extractor.get_model_name())

        # Pfad für das Phasenmodell (für Wiederholungserkennung)
        base_name, ext = os.path.splitext(self.pose_extractor.get_model_name())
        self.phase_model_path = os.path.join(self.model_dir, f"{base_name}_phases{ext}")

    def build_model(self, input_shape, num_categories):
        """
        Erstellt ein LSTM-Modell für die Klassifikation von Bewegungssequenzen.

        Args:
            input_shape: Form der Eingabedaten (Sequenzlänge, Anzahl Features)
            num_categories: Anzahl der Kategorien für die Ausgabeschicht

        Returns:
            Kompiliertes Keras-Modell
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(num_categories, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_model(self, video_dir, epochs=50, batch_size=16, validation_split=0.2):
        """
        Trainiert das Modell mit Videos aus einem Verzeichnis.

        Args:
            video_dir: Verzeichnis mit Trainingsvideos
            epochs: Anzahl der Trainingsiterationen
            batch_size: Batch-Größe für das Training
            validation_split: Anteil der Daten für die Validierung

        Returns:
            True bei erfolgreichem Training, False sonst
        """
        print(f"Training für {self.pose_extractor.get_exercise_name()} gestartet...")
        print("Lade Trainingsdaten...")
        X, y, filenames = self.pose_extractor.load_training_data(video_dir)

        if len(X) == 0:
            print("Keine Trainingsdaten gefunden!")
            return False

        print(f"Geladen: {len(X)} Videos")
        print(f"Kategorien: {self.pose_extractor.get_categories()}")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=42
        )

        input_shape = (X_train.shape[1], X_train.shape[2])
        num_categories = len(y[0])

        model = self.build_model(input_shape, num_categories)

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(self.model_path, save_best_only=True)
        ]

        print("Starte Training...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )

        # Modell speichern
        model.save(self.model_path)
        print(f"Modell gespeichert unter {self.model_path}")

        # Ergebnisse ausgeben
        val_loss, val_acc = model.evaluate(X_val, y_val)
        print(f"Validierungsgenauigkeit: {val_acc:.4f}")

        return True

    def train_phase_model(self, phase_video_dir, epochs=50, batch_size=16, validation_split=0.2):
        """
        Trainiert ein Modell zur Erkennung von Übungsphasen.

        Args:
            phase_video_dir: Verzeichnis mit Trainingsvideos für die Phasenerkennung
            epochs: Anzahl der Trainingsiterationen
            batch_size: Batch-Größe für das Training
            validation_split: Anteil der Daten für die Validierung

        Returns:
            True bei erfolgreichem Training, False sonst
        """
        print(f"Phase-Training für {self.pose_extractor.get_exercise_name()} gestartet...")
        print("Lade Phasen-Trainingsdaten...")

        X, y, filenames = self.pose_extractor.load_phase_training_data(phase_video_dir)

        if len(X) == 0:
            print("Keine Phasen-Trainingsdaten gefunden!")
            return False

        print(f"Geladen: {len(X)} Videos")
        phases = self.pose_extractor.get_phases()
        print(f"Phasen: {phases}")

        if not phases:
            print("Keine Phasen in der Konfiguration definiert!")
            return False

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=42
        )

        input_shape = (X_train.shape[1], X_train.shape[2])
        num_phases = len(phases)

        model = self.build_model(input_shape, num_phases)

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(self.phase_model_path, save_best_only=True)
        ]

        print("Starte Phasen-Training...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )

        # Modell speichern
        model.save(self.phase_model_path)
        print(f"Phasen-Modell gespeichert unter {self.phase_model_path}")

        # Ergebnisse ausgeben
        val_loss, val_acc = model.evaluate(X_val, y_val)
        print(f"Validierungsgenauigkeit (Phasen): {val_acc:.4f}")

        return True

    def train_all_models(self, exercise_dir, epochs=50, batch_size=16, validation_split=0.2):
        """
        Trainiert sowohl das Haupt-Übungsmodell als auch das Phasenmodell

        Args:
            exercise_dir: Hauptverzeichnis der Übung
            epochs: Anzahl der Trainingsiterationen
            batch_size: Batch-Größe für das Training
            validation_split: Anteil der Daten für die Validierung

        Returns:
            (bool, bool): Status des Trainings (Hauptmodell, Phasenmodell)
        """
        # Trainiere das Hauptmodell für die Übungsbewertung
        main_result = self.train_model(os.path.join(exercise_dir), epochs, batch_size, validation_split)

        # Trainiere das Phasenmodell für die Wiederholungserkennung
        phase_result = self.train_phase_model(os.path.join(exercise_dir, "phases"), epochs, batch_size, validation_split)

        return main_result, phase_result

