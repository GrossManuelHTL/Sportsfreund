import os
import argparse
import json
from exercise_manager import ExerciseManager
from model_trainer import ExerciseModelTrainer
from exercise_analyzer import ExerciseAnalyzer


def list_exercises(manager):
    """Listet alle verfügbaren Übungen auf"""
    exercises = manager.get_available_exercises()

    if not exercises:
        print("Keine Übungen vorhanden! Füge zuerst eine Übung hinzu.")
        return

    print("\n=== Verfügbare Übungen ===")
    for idx, exercise in enumerate(exercises, 1):
        print(f"{idx}. {exercise['name']} ({exercise['id']})")
        print(f"   {exercise['description']}")
    print()


def train_model(trainer, manager, exercise_name, epochs):
    """Trainiert das Modell für eine bestimmte Übung"""
    exercise = manager.get_exercise_config(exercise_name)

    if not exercise:
        print(f"Übung mit Name '{exercise_name}' nicht gefunden.")
        return

    print(f"Starte Training für Übung '{exercise['exercise_name']}'...")
    trainer.train_model(exercise, epochs=epochs)
    print(f"Training abgeschlossen. Modell gespeichert als '{exercise['model_name']}'.")


def analyze_exercise(analyzer, manager, exercise_id, video_path=None):
    """Analysiert eine Übungsausführung mit dem trainierten Modell"""
    exercise = manager.get_exercise_config(exercise_id)

    if not exercise:
        print(f"Übung mit ID '{exercise_id}' nicht gefunden.")
        return

    # Prüfen, ob ein Modell existiert
    model_path = os.path.join("models", exercise["model_name"])
    if not os.path.exists(model_path):
        print(f"Kein trainiertes Modell für '{exercise['name']}' gefunden.")
        print("Bitte trainiere zuerst ein Modell mit dem 'train' Befehl.")
        return

    # Wenn kein Videopfad angegeben, Webcam verwenden
    if video_path:
        result = analyzer.analyze_video(exercise_id, video_path)
    else:
        result = analyzer.analyze_live(exercise_id)

    # Ergebnisse anzeigen
    print("\n=== Analyse Ergebnisse ===")
    print(f"Übung: {exercise['name']}")

    for category_id, probability in result.items():

        category = next((c for c in exercise["categories"] if c["id"] == category_id), None)
        if category:
            print(f"{category['name']}: {probability:.2f}")

    main_category_id = max(result, key=result.get)
    main_category = next((c for c in exercise["categories"] if c["id"] == main_category_id), None)

    if main_category:
        print("\nFEEDBACK:")
        print(main_category["feedback"])

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("exercises", exist_ok=True)

    manager = ExerciseManager()

    # Kommandozeilenparser einrichten
    parser = argparse.ArgumentParser(description="Übungsbewertungs-System")
    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Befehle")

    # Übungen auflisten
    list_parser = subparsers.add_parser("list", help="Verfügbare Übungen anzeigen")

    # Neue Übung erstellen
    create_parser = subparsers.add_parser("create", help="Neue Übung erstellen")

    # Trainingsdaten aufzeichnen
    record_parser = subparsers.add_parser("record", help="Trainingsdaten aufzeichnen")
    record_parser.add_argument("--exercise_name", help="Name der Übung")
    record_parser.add_argument("--category_id", help="ID der Kategorie")
    record_parser.add_argument("--samples", type=int, default=10,
                               help="Anzahl der aufzuzeichnenden Videos (Standard: 10)")

    # Modell trainieren
    train_parser = subparsers.add_parser("train", help="Modell für eine Übung trainieren")
    train_parser.add_argument("--exercise_name", help="Name der Übung")
    train_parser.add_argument("--epochs", type=int, default=50,
                              help="Anzahl der Trainings-Epochen (Standard: 50)")

    # Übung analysieren
    analyze_parser = subparsers.add_parser("analyze", help="Übungsausführung analysieren")
    analyze_parser.add_argument("--exercise_name", help="ID der Übung")
    analyze_parser.add_argument("--video", help="Pfad zum Übungsvideo (optional, sonst Webcam)")

    # Befehle parsen und ausführen
    args = parser.parse_args()

    if args.command == "list":
        list_exercises(manager)

    elif args.command == "train":
        trainer = ExerciseModelTrainer(args.exercise_name)
        train_model(trainer, manager, args.exercise_name, args.epochs)

    elif args.command == "analyze":
        analyzer = ExerciseAnalyzer("exercises/" + args.exercise_name + "/config.json")
        analyze_exercise(analyzer, manager, args.exercise_name, args.video)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()