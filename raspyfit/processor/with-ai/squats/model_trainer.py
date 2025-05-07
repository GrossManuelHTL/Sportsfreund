import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from pose_extractor import PoseExtractor


class ExerciseModelTrainer:
    def __init__(self,
                 error_categories=None,
                 sequence_length=30,
                 model_path="exercise_model.h5"):
        """
        Initialisiert den Trainer für das Übungsmodell.

        Args:
            error_categories: Liste mit möglichen Fehlerkategorien
            sequence_length: Anzahl der Frames, die für jede Übung standardisiert werden
            model_path: Pfad zum Speichern des trainierten Modells
        """
        # Standardkategorien für Squats, falls nicht angegeben
        if error_categories is None:
            self.error_categories = [
                "correct",
                "knees_too_far_apart",
                "too_high",
                "wrong_going_up",
                "back_not_straight"
            ]
        else:
            self.error_categories = error_categories

        self.sequence_length = sequence_length
        self.model_path = model_path

        # Initialisieren des PoseExtractors
        self.pose_extractor = PoseExtractor(sequence_length=sequence_length)

    def build_model(self, input_shape):
        """
        Erstellt ein LSTM-Modell für die Klassifikation von Bewegungssequenzen.

        Args:
            input_shape: Form der Eingabedaten (Sequenzlänge, Anzahl Features)

        Returns:
            Kompiliertes Keras-Modell
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(len(self.error_categories), activation='softmax')
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
        print("Lade Trainingsdaten...")
        X, y, filenames = self.pose_extractor.load_training_data(
            video_dir,
            self.error_categories
        )

        if len(X) == 0:
            print("Keine Trainingsdaten gefunden!")
            return False

        print(f"Geladen: {len(X)} Videos")

        # Aufteilen in Trainings- und Validierungsdaten
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=42
        )

        # Input Shape: (Sequenzlänge, Anzahl Features pro Frame)
        input_shape = (X_train.shape[1], X_train.shape[2])

        # Modell erstellen
        model = self.build_model(input_shape)

        # Callbacks für Training
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(self.model_path, save_best_only=True)
        ]

        # Modell trainieren
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

        # Evaluieren
        loss, accuracy = model.evaluate(X_val, y_val)
        print(f"Validierungsgenauigkeit: {accuracy * 100:.2f}%")

        return True

    def get_model_path(self):
        """Gibt den Pfad zum gespeicherten Modell zurück"""
        return self.model_path

    def get_error_categories(self):
        """Gibt die Fehlerkategorien zurück"""
        return self.error_categories