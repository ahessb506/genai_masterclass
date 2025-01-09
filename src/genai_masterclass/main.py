#!/usr/bin/env python
import sys
import warnings
import opentelemetry.trace
from pathlib import Path

# Suppress the TracerProvider warning globally
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                      message="Overriding of current TracerProvider is not allowed")

from genai_masterclass.crew import MasterclassCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """Run the masterclass crew."""
    try:
        # Ensure warning is suppressed in this context
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, 
                                  message="Overriding of current TracerProvider is not allowed")
            
            print("Starting GenAI Masterclass creation process...")
            
            crew = MasterclassCrew()
            print(f"Output directory: {crew.output_path}")
            
            result = crew.get_crew().kickoff()
            
            # Verify outputs were created
            output_files = [
                'professor_guide.md',
                'presentation_slides.md',
                'student_handout.md'
            ]
            
            print("\nChecking output files:")
            for filename in output_files:
                file_path = crew.output_path / filename
                if file_path.exists():
                    print(f"✓ {filename} was created successfully")
                    # Optional: print file size
                    print(f"  Size: {file_path.stat().st_size} bytes")
                else:
                    print(f"✗ {filename} was not created")
            
            print("\nProcess completed!")
            return 0
            
    except Exception as e:
        print(f"\nAn error occurred while running the crew: {str(e)}")
        return 1


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'topic': 'Generative AI for Beginners',
        'audience': 'Non-technical professionals',
        'duration': 'One-day masterclass',
        'focus': 'Practical AI tools and productivity enhancement'
    }
    try:
        MasterclassCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        MasterclassCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'topic': 'Generative AI for Beginners',
        'audience': 'Non-technical professionals',
        'duration': 'One-day masterclass',
        'focus': 'Practical AI tools and productivity enhancement'
    }
    try:
        MasterclassCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    sys.exit(run())
