"""
Sequenz-Labeling-Modell mit LSTM oder Transformer.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Bidirectional, Dropout,
    LayerNormalization, MultiHeadAttention, Conv1D,
    GlobalAveragePooling1D, Masking
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from typing import Tuple, List, Dict, Any, Optional

from .config import (
    MODEL_DIR,
    SEQUENCE_LENGTH,
    NUM_KEYPOINTS,
    NUM_COORDINATES,
    NUM_CLASSES,
    MODEL_TYPE,
    HIDDEN_UNITS,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS
)

class SequenceLabelingModel:
    """
    Klasse für das Training und die Inferenz eines Sequenz-Labeling-Modells für Sportübungen.

    Das Modell kann entweder ein LSTM oder ein Transformer sein und wird verwendet, um
    den Zustand der Übungsausführung (Pause, Wiederholung läuft, Wiederholung beendet)
    für jeden Frame zu klassifizieren.
    """

    def __init__(self, model_name: str = "exercise_model", use_saved_model: bool = True):
        """
        Initialisiert das Modell.

        Args:
            model_name: Name des Modells für das Speichern/Laden
            use_saved_model: Ob ein gespeichertes Modell geladen werden soll, falls verfügbar
        """
        self.model_name = model_name
        self.model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")

        # Definiere Eingabe- und Ausgabedimensionen
        self.input_shape = (SEQUENCE_LENGTH, NUM_KEYPOINTS * NUM_COORDINATES)
        self.num_classes = NUM_CLASSES

        # Erstelle oder lade das Modell
        if use_saved_model and os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Modell geladen von: {self.model_path}")
        else:
            self.model = self._build_model(MODEL_TYPE)

    def _build_lstm_model(self) -> Model:
        """
        Erstellt ein LSTM-basiertes Modell für Sequenz-Labeling.

        Returns:
            Ein kompiliertes Keras-Modell
        """
        model = Sequential([
            Input(shape=self.input_shape),
            Masking(mask_value=0.0),  # Maskierung für variable Sequenzlängen
            Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)),
            Dropout(0.3),
            Dense(HIDDEN_UNITS, activation='relu'),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _build_transformer_model(self) -> Model:
        """
        Erstellt ein Transformer-basiertes Modell für Sequenz-Labeling.

        Returns:
            Ein kompiliertes Keras-Modell
        """
        # Definiere Eingabe
        inputs = Input(shape=self.input_shape)

        # Maskierung für variable Sequenzlängen
        x = Masking(mask_value=0.0)(inputs)

        # Positionale Codierung mit Conv1D
        x = Conv1D(filters=HIDDEN_UNITS, kernel_size=3, padding='same')(x)

        # Transformer-Block
        for _ in range(2):
            # Multi-Head Attention
            attn_output = MultiHeadAttention(
                num_heads=8, key_dim=HIDDEN_UNITS // 8
            )(x, x)

            # Add & Norm
            x = LayerNormalization()(x + attn_output)

            # Feedforward & Residual
            ff_output = Sequential([
                Dense(HIDDEN_UNITS * 4, activation='relu'),
                Dropout(0.2),
                Dense(HIDDEN_UNITS)
            ])(x)

            # Add & Norm
            x = LayerNormalization()(x + ff_output)

        # Ausgabeschicht
        outputs = Dense(self.num_classes, activation='softmax')(x)

        # Erstelle das Modell
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _build_model(self, model_type: str) -> Model:
        """
        Erstellt das Modell basierend auf dem angegebenen Typ.

        Args:
            model_type: Art des zu erstellenden Modells ('LSTM' oder 'TRANSFORMER')

        Returns:
            Ein kompiliertes Keras-Modell
        """
        if model_type.upper() == 'LSTM':
            return self._build_lstm_model()
        elif model_type.upper() == 'TRANSFORMER':
            return self._build_transformer_model()
        else:
            raise ValueError(f"Unbekannter Modelltyp: {model_type}. Wähle 'LSTM' oder 'TRANSFORMER'.")

    def preprocess_data(self,
                        pose_sequences: List[np.ndarray],
                        labels: Optional[List[int]] = None
                        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Vorverarbeitung der Pose-Sequenzen und Labels.

        Args:
            pose_sequences: Liste von Pose-Sequenzen (jeweils [seq_len, num_keypoints, 3])
            labels: Liste von Labels für jede Sequenz (optional für Inferenz)

        Returns:
            Tuple mit:
                - Vorverarbeitete Pose-Sequenzen
                - Vorverarbeitete Labels (oder None, wenn keine Labels übergeben wurden)
        """
        processed_sequences = []

        for sequence in pose_sequences:
            # Stelle sicher, dass jede Sequenz SEQUENCE_LENGTH Frames hat
            if len(sequence) < SEQUENCE_LENGTH:
                # Auffüllen mit Nullen, wenn die Sequenz zu kurz ist
                padding = np.zeros((SEQUENCE_LENGTH - len(sequence), NUM_KEYPOINTS, NUM_COORDINATES))
                sequence = np.concatenate([sequence, padding], axis=0)
            elif len(sequence) > SEQUENCE_LENGTH:
                # Abschneiden, wenn die Sequenz zu lang ist
                sequence = sequence[:SEQUENCE_LENGTH]

            # Umformen in das erwartete Format [SEQUENCE_LENGTH, NUM_KEYPOINTS * NUM_COORDINATES]
            flat_sequence = sequence.reshape(SEQUENCE_LENGTH, -1)
            processed_sequences.append(flat_sequence)

        # Konvertieren zu NumPy-Array
        X = np.array(processed_sequences, dtype=np.float32)

        if labels is not None:
            # One-Hot-Codierung der Labels
            y = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)
            return X, y
        else:
            return X, None

    def train(self,
             train_sequences: List[np.ndarray],
             train_labels: List[int],
             val_sequences: Optional[List[np.ndarray]] = None,
             val_labels: Optional[List[int]] = None,
             class_weights: Optional[Dict[int, float]] = None) -> tf.keras.callbacks.History:
        """
        Trainiert das Modell.

        Args:
            train_sequences: Trainingsdaten - Liste von Pose-Sequenzen
            train_labels: Trainingslabels für jede Sequenz
            val_sequences: Validierungsdaten (optional)
            val_labels: Validierungslabels (optional)
            class_weights: Klassen-Gewichte für unausgewogene Daten (optional)

        Returns:
            Trainingsverlauf
        """
        # Vorverarbeitung der Trainingsdaten
        X_train, y_train = self.preprocess_data(train_sequences, train_labels)

        # Vorverarbeitung der Validierungsdaten, falls vorhanden
        validation_data = None
        if val_sequences is not None and val_labels is not None:
            X_val, y_val = self.preprocess_data(val_sequences, val_labels)
            validation_data = (X_val, y_val)

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=self.model_path,
                monitor='val_accuracy' if validation_data else 'accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Training
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights
        )

        # Speichere das Modell
        self.model.save(self.model_path)
        print(f"Modell gespeichert unter: {self.model_path}")

        return history

    def predict_sequence(self, sequence: np.ndarray) -> List[int]:
        """
        Führt die Vorhersage für eine einzelne Sequenz durch.

        Args:
            sequence: Eine einzelne Pose-Sequenz mit Form [seq_len, num_keypoints, 3]

        Returns:
            Liste der vorhergesagten Klassen für jeden Frame
        """
        # Vorverarbeitung
        X, _ = self.preprocess_data([sequence])

        # Vorhersage
        predictions = self.model.predict(X)[0]

        # Konvertiere zu Klassenindizes
        predicted_classes = np.argmax(predictions, axis=-1)

        return predicted_classes.tolist()

    def predict_batch(self, sequences: List[np.ndarray]) -> List[List[int]]:
        """
        Führt die Vorhersage für einen Batch von Sequenzen durch.

        Args:
            sequences: Liste von Pose-Sequenzen

        Returns:
            Liste der vorhergesagten Klassen für jeden Frame in jeder Sequenz
        """
        # Vorverarbeitung
        X, _ = self.preprocess_data(sequences)

        # Vorhersage
        predictions = self.model.predict(X)

        # Konvertiere zu Klassenindizes
        predicted_classes = np.argmax(predictions, axis=-1)

        return [seq_predictions.tolist() for seq_predictions in predicted_classes]

    def evaluate(self,
                test_sequences: List[np.ndarray],
                test_labels: List[int]) -> Dict[str, float]:
        """
        Evaluiert das Modell auf Testdaten.

        Args:
            test_sequences: Testdaten - Liste von Pose-Sequenzen
            test_labels: Testlabels für jede Sequenz

        Returns:
            Dictionary mit Evaluierungsmetriken
        """
        # Vorverarbeitung
        X_test, y_test = self.preprocess_data(test_sequences, test_labels)

        # Evaluierung
        metrics = self.model.evaluate(X_test, y_test)

        # Erstelle ein Dictionary mit den Metriken
        metrics_dict = dict(zip(self.model.metrics_names, metrics))

        return metrics_dict
