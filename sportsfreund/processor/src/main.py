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
    exercises_dir = os.path.join("exercises", exercise_name)

    print("Train model method")
    print(exercises_dir)

    if not exercise:
        print(f"Übung mit Name '{exercise_name}' nicht gefunden.")
        return

    # Modell trainieren
    print(f"Starte Training für Übung '{exercise['exercise_name']}'...")

    if "phases" in exercise:
        # Trainiere beide Modelle - das Hauptmodell und das Phasenmodell
        print("Übung hat Phasen definiert. Trainiere Hauptmodell und Phasenmodell...")
        main_result, phase_result = trainer.train_all_models(exercises_dir, epochs=epochs)

        if main_result and phase_result:
            print(f"Training für beide Modelle abgeschlossen.")
        else:
            print("Training teilweise fehlgeschlagen. Bitte Trainingsdaten überprüfen.")
    else:
        # Nur das Hauptmodell trainieren
        success = trainer.train_model(exercises_dir, epochs=epochs)

        if success:
            print(f"Training abgeschlossen. Modell gespeichert als '{exercise['model_name']}'.")
        else:
            print("Training fehlgeschlagen. Bitte Trainingsdaten überprüfen.")


def format_repetition(rep_number, repetition):
    """Formatiert Informationen zu einer Wiederholung für die Konsolenausgabe"""
    feedback = repetition.get("feedback", {})
    phases = repetition.get("phases", [])

    phase_str = " -> ".join([phase["phase"] for phase in phases])

    output = []
    output.append(f"\n  ⟢ Wiederholung {rep_number}: {phase_str} ⟣")

    if feedback:
        category = feedback.get("category", "")
        text = feedback.get("text", "")
        confidence = feedback.get("confidence", 0) * 100

        output.append(f"    ▸ Bewertung: {category} ({confidence:.1f}%)")
        output.append(f"    ▸ Feedback: {text}")

    return "\n".join(output)


def print_analysis_result(result):
    """Gibt das Analyseergebnis formatiert in der Konsole aus"""
    if not result:
        print("Keine Analyseergebnisse verfügbar.")
        return

    exercise_name = result.get("exercise", "Unbekannte Übung")

    # Header
    print("\n" + "=" * 60)
    print(f"  ÜBUNGSANALYSE: {exercise_name.upper()}")
    print("=" * 60)

    # Gesamtfeedback zum Video
    if "text" in result:
        print(f"\nGesamtfeedback: {result['text']}")

    # Wiederholungen
    if "repetitions" in result:
        rep_data = result["repetitions"]
        rep_count = rep_data.get("count", 0)

        print(f"\nERKANNTE WIEDERHOLUNGEN: {rep_count}")
        print("-" * 60)

        if rep_count > 0:
            for i, rep in enumerate(rep_data.get("repetitions", []), 1):
                print(format_repetition(i, rep))
        else:
            print("\n  Keine Wiederholungen erkannt. Möglicherweise zu kurzes Video oder unvollständige Übung.")

        print("\n" + "-" * 60)

    # Zusammenfassung
    print("\nZUSAMMENFASSUNG:")
    if "predicted_category" in result and "confidence" in result:
        category = result.get("predicted_category", "")
        confidence = result.get("confidence", 0) * 100
        print(f"  ▸ Gesamtbewertung: {category} ({confidence:.1f}%)")

    # Auffälligkeiten bei den Wiederholungen zusammenfassen
    if "repetitions" in result and rep_data.get("count", 0) > 0:
        categories = {}
        for rep in rep_data.get("repetitions", []):
            feedback = rep.get("feedback", {})
            category = feedback.get("category", "")
            if category:
                if category in categories:
                    categories[category] += 1
                else:
                    categories[category] = 1

        print("  ▸ Wiederholungsmuster:")
        for category, count in categories.items():
            print(f"     - {count}x {category}")

    print("\n" + "=" * 60)


def analyze_exercise(analyzer, manager, exercise_id, video_path=None, show_visualization=False):
    """Analysiert eine Übungsausführung mit dem trainierten Modell"""
    exercise = manager.get_exercise_config(exercise_id)

    if not exercise:
        print(f"Übung mit ID '{exercise_id}' nicht gefunden.")
        return

    model_path = os.path.join("models", exercise["model_name"])
    if not os.path.exists(model_path):
        print(f"Kein trainiertes Modell für '{exercise['exercise_name']}' gefunden.")
        print("Bitte trainiere zuerst ein Modell mit dem 'train' Befehl.")
        return

    if video_path:
        result = analyzer.analyze_video(video_path, show_visualization=show_visualization)
    else:
        # Für Live-Analyse
        result = analyzer.analyze_live()

    if not result:
        print("Analyse fehlgeschlagen.")
        return

    # Ergebnisse ausgeben
    print_analysis_result(result)


def main():
    """Hauptfunktion der Anwendung"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("exercises", exist_ok=True)

    manager = ExerciseManager()

    parser = argparse.ArgumentParser(description="Übungsbewertungs-System")
    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Befehle")

    list_parser = subparsers.add_parser("list", help="Verfügbare Übungen anzeigen")

    create_parser = subparsers.add_parser("create", help="Neue Übung erstellen")

    record_parser = subparsers.add_parser("record", help="Trainingsdaten aufzeichnen")
    record_parser.add_argument("--exercise_name", help="Name der Übung")
    record_parser.add_argument("category_id", help="ID der Kategorie")
    record_parser.add_argument("--samples", type=int, default=10,
                               help="Anzahl der aufzuzeichnenden Videos (Standard: 10)")

    train_parser = subparsers.add_parser("train", help="Modell für eine Übung trainieren")
    train_parser.add_argument("--exercise_name", help="Name der Übung")
    train_parser.add_argument("--epochs", type=int, default=50,
                              help="Anzahl der Trainings-Epochen (Standard: 50)")

    analyze_parser = subparsers.add_parser("analyze", help="Übung analysieren")
    analyze_parser.add_argument("--exercise_name", required=True, help="Name der Übung")
    analyze_parser.add_argument("--video", required=True, help="Pfad zum Übungsvideo")
    analyze_parser.add_argument("--visualization", action="store_true",
                                help="Visualisierung der Pose-Erkennung anzeigen")

    args = parser.parse_args()

    if args.command == "list":
        list_exercises(manager)
    elif args.command == "train":
        trainer = ExerciseModelTrainer(args.exercise_name)
        train_model(trainer, manager, args.exercise_name, args.epochs)
    elif args.command == "analyze":
        analyzer = ExerciseAnalyzer(args.exercise_name)
        analyze_exercise(analyzer, manager, args.exercise_name, args.video, args.visualization)
    elif args.command == "create":
        # TODO: Implementierung für das Erstellen einer Übung
        print("Diese Funktion ist noch nicht implementiert.")
    elif args.command == "record":
        # TODO: Implementierung für das Aufzeichnen von Trainingsdaten
        print("Diese Funktion ist noch nicht implementiert.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
