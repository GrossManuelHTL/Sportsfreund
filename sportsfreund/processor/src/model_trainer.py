import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
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

        model.save(self.model_path)
        print(f"Modell gespeichert unter {self.model_path}")

        loss, accuracy = model.evaluate(X_val, y_val)
        print(f"Validierungsgenauigkeit: {accuracy * 100:.2f}%")

        return True

    def get_model_path(self):
        """Gibt den Pfad zum gespeicherten Modell zurück"""
        return self.model_path