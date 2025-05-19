#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
from exercise_analyzer import ExerciseAnalyzer
from model_trainer import ExerciseModelTrainer
from exercise_manager import ExerciseManager


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


def main():
    parser = argparse.ArgumentParser(description="Analysiert ein Übungsvideo und zählt Wiederholungen")
    parser.add_argument("exercise", help="Name der Übung")
    parser.add_argument("video", help="Pfad zum Analysevideo")
    parser.add_argument("--train", action="store_true", help="Modelle vor der Analyse trainieren")
    parser.add_argument("--visualization", action="store_true", help="Visualisierung der Pose-Erkennung anzeigen")

    args = parser.parse_args()

    exercise_name = args.exercise
    video_path = args.video

    manager = ExerciseManager()
    exercises = manager.get_available_exercises()
    exercise_exists = False

    for exercise in exercises:
        if exercise["id"] == exercise_name:
            exercise_exists = True
            break

    if not exercise_exists:
        print(f"Fehler: Übung '{exercise_name}' nicht gefunden.")
        print("Verfügbare Übungen:")
        for ex in exercises:
            print(f"  - {ex['id']}: {ex['name']}")
        return

    # Wenn --train angegeben, zuerst Modelle trainieren
    if args.train:
        print(f"Training der Modelle für {exercise_name} wird durchgeführt...")
        trainer = ExerciseModelTrainer(exercise_name)

        exercise_dir = os.path.join("exercises", exercise_name)
        main_result, phase_result = trainer.train_all_models(exercise_dir)

        if not main_result or not phase_result:
            print("Training fehlgeschlagen. Bitte überprüfe die Trainingsdaten.")
            return

        print("Training abgeschlossen.")

    # Video analysieren
    analyzer = ExerciseAnalyzer(exercise_name)
    result = analyzer.analyze_video(video_path, show_visualization=args.visualization)

    if result:
        print_analysis_result(result)
    else:
        print("Analyse fehlgeschlagen.")


if __name__ == "__main__":
    main()
