from pathlib import Path
import yaml
import os
import warnings
from crewai import Agent, Task, Crew, Process

class MasterclassCrew:
    """Crew for developing a GenAI masterclass for non-technical audiences"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    masterclass_config = 'config/masterclass_concept.yaml'

    def __init__(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning, 
                              message="Overriding of current TracerProvider is not allowed")
        
        self.base_path = Path(__file__).parent
        self.output_path = self.base_path / 'outputs'
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self._load_masterclass_concept()
        self._load_tasks_config()
        self._load_agents_config()

    def _load_masterclass_concept(self):
        config_path = self.base_path / self.masterclass_config
        with open(config_path, 'r') as f:
            self.masterclass_concept = yaml.safe_load(f)

    def _load_tasks_config(self):
        config_path = self.base_path / self.tasks_config
        with open(config_path, 'r') as f:
            self.tasks = yaml.safe_load(f)
            print(f"Loaded tasks: {list(self.tasks.keys())}")

    def _load_agents_config(self):
        config_path = self.base_path / self.agents_config
        with open(config_path, 'r') as f:
            self.agents = yaml.safe_load(f)

    def content_developer(self) -> Agent:
        return Agent(
            role=self.agents['content_developer']['role'],
            goal=self.agents['content_developer']['goal'],
            backstory=self.agents['content_developer']['backstory'],
            verbose=True
        )

    def feedback_agent(self) -> Agent:
        return Agent(
            role=self.agents['feedback_agent']['role'],
            goal=self.agents['feedback_agent']['goal'],
            backstory=self.agents['feedback_agent']['backstory'],
            verbose=True
        )

    def materials_creator(self) -> Agent:
        return Agent(
            role=self.agents['materials_creator']['role'],
            goal=self.agents['materials_creator']['goal'],
            backstory=self.agents['materials_creator']['backstory'],
            verbose=True
        )

    def get_crew(self) -> Crew:
        try:
            # Step 1: Create Initial Outline
            print("\nCreating initial outline...")
            initial_outline_task = Task(
                description=self.tasks['create_initial_outline']['description'].format(
                    masterclass_concept=self.masterclass_concept['concept']  # Make sure to access the correct key
                ),
                expected_output=self.tasks['create_initial_outline']['expected_output'],
                agent=self.content_developer()
            )
            
            initial_crew = Crew(
                agents=[self.content_developer()],
                tasks=[initial_outline_task],
                process=Process.sequential,
                verbose=True
            )
            
            initial_result = initial_crew.kickoff()
            initial_outline = self._get_result_content(initial_result)
            self._save_output('initial_outline.md', initial_outline)
            print("✓ Initial outline created")

            # Step 2: Review Initial Outline
            print("\nReviewing initial outline...")
            review_task = Task(
                description=self.tasks['review_initial_outline']['description'].format(
                    initial_outline=initial_outline
                ),
                expected_output=self.tasks['review_initial_outline']['expected_output'],
                agent=self.feedback_agent()
            )
            
            review_crew = Crew(
                agents=[self.feedback_agent()],
                tasks=[review_task],
                process=Process.sequential,
                verbose=True
            )
            
            review_result = review_crew.kickoff()
            outline_review = self._get_result_content(review_result)
            self._save_output('outline_review.md', outline_review)
            print("✓ Outline review completed")

            # Step 3: Create Final Outline
            print("\nCreating final outline...")
            final_outline_task = Task(
                description=self.tasks['create_final_outline']['description'].format(
                    initial_outline=initial_outline,
                    outline_review=outline_review,
                    masterclass_concept=self.masterclass_concept['concept']
                ),
                expected_output=self.tasks['create_final_outline']['expected_output'],
                agent=self.content_developer()
            )
            
            final_crew = Crew(
                agents=[self.content_developer()],
                tasks=[final_outline_task],
                process=Process.sequential,
                verbose=True
            )
            
            final_result = final_crew.kickoff()
            self.approved_outline = self._get_result_content(final_result)
            self._save_output('final_outline.md', self.approved_outline)
            print("✓ Final outline created and approved")

            # Now that we have the approved outline, create the professor's guide
            print("\nGenerating professor's guide...")
            if not hasattr(self, 'approved_outline'):
                raise ValueError("No approved outline available. Please ensure the outline is created first.")
                
            guide_task = Task(
                description=self.tasks['create_professor_guide']['description'].format(
                    approved_outline=self.approved_outline
                ),
                expected_output=self.tasks['create_professor_guide']['expected_output'],
                agent=self.content_developer()
            )
            
            guide_crew = Crew(
                agents=[self.content_developer()],
                tasks=[guide_task],
                process=Process.sequential,
                verbose=True
            )
            
            guide_result = guide_crew.kickoff()
            guide_content = self._get_result_content(guide_result)
            self._save_output('professor_guide.md', guide_content)
            print("✓ Professor's guide created")

            return guide_crew

        except Exception as e:
            print(f"\nError during task execution: {str(e)}")
            print(f"Tasks loaded: {list(self.tasks.keys())}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise

    def _get_result_content(self, result):
        """Helper method to extract content from crew results"""
        try:
            if hasattr(result, 'raw_output'):
                return result.raw_output
            elif hasattr(result, 'outputs'):
                return result.outputs[0] if result.outputs else str(result)
            elif isinstance(result, (list, tuple)):
                return result[0]
            else:
                return str(result)
        except Exception as e:
            print(f"Warning: Error extracting content: {e}")
            return str(result)

    def _save_output(self, filename: str, content: str):
        """Helper method to save output to file"""
        output_path = self.output_path / filename
        with open(output_path, 'w') as f:
            f.write(content)