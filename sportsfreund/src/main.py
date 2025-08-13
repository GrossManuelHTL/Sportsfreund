"""
Main entry point for the Exercise Analysis System
Provides command-line interface for training and analysis
"""
import sys
import argparse
from pathlib import Path
from exercise_system import ExerciseAnalysisSystem, FeedbackGenerator
from rep_counter import RepCounter


def main():
    parser = argparse.ArgumentParser(description='Exercise Analysis System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Training command
    train_parser = subparsers.add_parser('train', help='Train models for an exercise')
    train_parser.add_argument('--config', help='Path to training configuration file')

    rep_count_parser = subparsers.add_parser('repcount', help='Rep count for an exercise')
    rep_count_parser.add_argument('--video', help='Path to video file for rep counting', default='testvideos/squat_single.mp4')
    rep_count_parser.add_argument('--exercise', help='Exercise type (squat, pushup, etc.)', default='squat')

    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a video')
    analyze_parser.add_argument('--video', help='Path to video file')
    analyze_parser.add_argument('--exercise', help='Exercise type (squat, pushup, etc.)')
    analyze_parser.add_argument('--no-models', action='store_true', help='Disable ML models')
    analyze_parser.add_argument('--visualize', action='store_true', help='Show real-time visualization')
    analyze_parser.add_argument('--output', help='Save detailed results to JSON file')

    # List command
    list_parser = subparsers.add_parser('list', help='List available exercises')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    system = ExerciseAnalysisSystem()

    try:
        if args.command == 'train':
            train_models(system, args.config)
        elif args.command == 'analyze':
            analyze_video(system, args)
        elif args.command == 'list':
            list_exercises(system)
        elif args.command == 'repcount':
            repcounter = RepCounter(args.exercise)
            repcounter.count_video(args.video)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def train_models(system: ExerciseAnalysisSystem, config_path: str):
    """Train models based on configuration"""
    print(f"Training models from config: {config_path}")

    try:
        results = system.train_exercise_models(config_path)

        print("\n=== Training Results ===")
        for model_type, result in results.items():
            if 'error' in result:
                print(f"{model_type}: FAILED - {result['error']}")
            else:
                accuracy = result.get('accuracy', 0)
                samples = result.get('total_samples', 0)
                print(f"{model_type}: SUCCESS - {accuracy:.3f} accuracy ({samples} samples)")

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

def analyze_video(system: ExerciseAnalysisSystem, args):
    """Analyze a video file"""
    video_path = args.video
    exercise_type = args.exercise
    use_models = not args.no_models

    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    if exercise_type not in system.get_available_exercises():
        print(f"Error: Exercise type '{exercise_type}' not supported.")
        print(f"Available exercises: {', '.join(system.get_available_exercises())}")
        sys.exit(1)

    print(f"Analyzing {exercise_type} video: {video_path}")
    if not use_models:
        print("Note: ML models disabled, using rule-based analysis only")

    try:
        # Perform analysis
        results = system.analyze_video(
            video_path=video_path,
            exercise_type=exercise_type,
            use_models=use_models,
            show_visualization=args.visualize
        )

        # Generate and display feedback
        feedback = FeedbackGenerator.generate_detailed_feedback(results)
        print("\n" + feedback)

        # Save detailed results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {args.output}")

    except Exception as e:
        print(f"Analysis failed: {e}")
        sys.exit(1)

def list_exercises(system: ExerciseAnalysisSystem):
    """List available exercise types"""
    exercises = system.get_available_exercises()
    print("Available exercise types:")
    for exercise in exercises:
        model_status = check_model_status(exercise)
        print(f"  • {exercise} {model_status}")

def check_model_status(exercise_name: str) -> str:
    """Check if trained models exist for an exercise"""
    model_dir = Path(f"models/{exercise_name}")

    rep_model = model_dir / "rep_model.pkl"
    form_model = model_dir / "form_model.pkl"

    if rep_model.exists() and form_model.exists():
        return "(trained models available)"
    elif rep_model.exists() or form_model.exists():
        return "(partially trained)"
    else:
        return "(no trained models)"

if __name__ == "__main__":
    print("🏋️ Exercise Analysis System")
    print("=" * 50)

    if len(sys.argv) == 1:
        print("Quick start examples:")
        print("  python main.py train configs/squat_training.json")
        print("  python main.py analyze video.mp4 squat")
        print("  python main.py analyze video.mp4 squat --visualize")
        print("  python main.py list")
        print("\nFor detailed help: python main.py --help")
    else:
        main()
