#!/usr/bin/env python
import sys
import warnings
import opentelemetry.trace

from genai_masterclass.crew import MasterclassCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """Run the masterclass crew."""
    # Suppress the tracer provider warning
    warnings.filterwarnings("ignore", category=RuntimeWarning, 
                          message="Overriding of current TracerProvider is not allowed")
    
    try:
        print("Starting GenAI Masterclass material creation...")
        crew = MasterclassCrew()
        result = crew.get_crew().kickoff()
        print("\nMasterclass materials have been created successfully!")
        return 0
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        return 1
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
    run()
