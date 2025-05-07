import os
import argparse
from model_trainer import ExerciseModelTrainer
from exercise_analyzer import ExerciseAnalyzer


def main():
    # Argument-Parser für Kommandozeilenoptionen
    parser = argparse.ArgumentParser(description='Fitness-Übungs-Analyse mit KI')
    parser.add_argument('--mode', type=str, default='analyze',
                        choices=['train', 'analyze', 'both'],
                        help='Betriebsmodus: train, analyze oder both')
    parser.add_argument('--training_dir', type=str, default='trainingdata',
                        help='Verzeichnis mit Trainingsvideos')
    parser.add_argument('--test_video', type=str, default=None,
                        help='Zu analysierendes Video')
    parser.add_argument('--model_path', type=str, default='squat_model.h5',
                        help='Pfad zum Modell')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualisierung anzeigen')

    args = parser.parse_args()

    # Fehlerkategorien definieren (kann je nach Übung angepasst werden)
    error_categories = [
        "correct",
        "knees_too_far_apart",
        "back_not_straight",
        "too_high",
        "wrong_going_up"
    ]

    # Modell trainieren, falls gewünscht
    if args.mode in ['train', 'both']:
        print(f"=== Starte Training mit Videos aus {args.training_dir} ===")
        trainer = ExerciseModelTrainer(
            error_categories=error_categories,
            model_path=args.model_path
        )
        success = trainer.train_model(args.training_dir)

        if not success:
            print("Training fehlgeschlagen!")
            if args.mode == 'both':
                print("Analyse wird übersprungen.")
                return

    # Video analysieren, falls gewünscht
    if args.mode in ['analyze', 'both']:
        if args.test_video is None:
            print("Fehler: Kein Testvideo angegeben!")
            print("Verwende --test_video, um ein Video zu analysieren.")
            return

        if not os.path.exists(args.model_path):
            print(f"Fehler: Modell {args.model_path} nicht gefunden!")
            return

        print(f"=== Analysiere Video {args.test_video} ===")
        analyzer = ExerciseAnalyzer(
            model_path=args.model_path,
            error_categories=error_categories
        )

        feedback = analyzer.analyze_video(args.test_video, show_visualization=args.visualize)

        if feedback:
            print("\n=== Feedback zur Übungsausführung ===")
            print(f"Erkannte Kategorie: {feedback['predicted_category']} (Konfidenz: {feedback['confidence']:.2f})")
            print(f"Feedback: {feedback['text']}")

            print("\nDetaillierte Wahrscheinlichkeiten:")
            for category, prob in feedback['all_probabilities'].items():
                print(f"  - {category}: {prob:.4f}")
        else:
            print("Analyse fehlgeschlagen!")


# Beispiel für die Verwendung als importiertes Modul
def analyze_single_video(video_path, model_path, visualize=False):
    """
    Hilfsfunktion zur Analyse eines einzelnen Videos von anderen Skripten aus.

    Args:
        video_path: Pfad zum Übungsvideo
        model_path: Pfad zum trainierten Modell
        visualize: Ob die Visualisierung angezeigt werden soll

    Returns:
        Dictionary mit Feedback-Informationen oder None bei Fehler
    """
    error_categories = [
        "correct",
        "knees_too_far_apart",
        "back_not_straight",
        "too_high",
        "wrong_going_up"
    ]

    analyzer = ExerciseAnalyzer(
        model_path=model_path,
        error_categories=error_categories
    )

    return analyzer.analyze_video(video_path, show_visualization=visualize)


if __name__ == "__main__":
    main()