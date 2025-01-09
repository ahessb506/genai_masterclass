from pathlib import Path
import yaml
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew, after_kickoff

@CrewBase
class MasterclassCrew:
    """Crew for developing a GenAI masterclass for non-technical audiences"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    masterclass_config = 'config/masterclass_concept.yaml'

    def __init__(self):
        self.base_path = Path(__file__).parent
        super().__init__()
        self._load_masterclass_concept()

    def _load_masterclass_concept(self):
        """Load masterclass concept from config file."""
        config_path = self.base_path / self.masterclass_config
        with open(config_path, 'r') as f:
            self.masterclass_concept = yaml.safe_load(f)

    @agent
    def content_developer(self) -> Agent:
        return Agent(config=self.agents_config['content_developer'], verbose=True)

    @agent
    def feedback_agent(self) -> Agent:
        return Agent(config=self.agents_config['feedback_agent'], verbose=True)

    @agent
    def materials_creator(self) -> Agent:
        return Agent(config=self.agents_config['materials_creator'], verbose=True)

    @task
    def create_initial_outline(self) -> Task:
        return Task(
            description=f"Based on this masterclass concept:\n\n{yaml.dump(self.masterclass_concept, sort_keys=False)}\n\nCreate a detailed course outline that includes:\n- Clear learning objectives\n- Topic breakdown\n- Time allocation\n- Key points\n- Activities",
            expected_output="A detailed course outline in Markdown format",
            agent=self.content_developer()
        )

    @task
    def review_initial_outline(self) -> Task:
        return Task(
            description="Review the initial course outline and provide feedback",
            expected_output="Detailed review and suggestions in Markdown format",
            agent=self.feedback_agent(),
            context=[self.create_initial_outline()]
        )

    @task
    def create_final_outline(self) -> Task:
        return Task(
            description="Create the final course outline incorporating the feedback",
            expected_output="Final course outline in Markdown format",
            agent=self.content_developer(),
            context=[self.create_initial_outline(), self.review_initial_outline()]
        )

    @task
    def create_professor_guide(self) -> Task:
        return Task(
            description="Create a comprehensive teaching guide based on the final outline",
            expected_output="A comprehensive professor's guide in Markdown format",
            agent=self.content_developer(),
            context=[self.create_final_outline()]
        )

    @task
    def create_presentation_slides(self) -> Task:
        return Task(
            description="Create presentation slides based on the final outline",
            expected_output="Slide-by-slide content in Markdown format",
            agent=self.materials_creator(),
            context=[self.create_final_outline()]
        )

    @task
    def create_student_handout(self) -> Task:
        return Task(
            description="Create a concise reference guide based on the final outline",
            expected_output="A concise summary document in Markdown format",
            agent=self.materials_creator(),
            context=[self.create_final_outline()]
        )

    @task
    def review_all_materials(self) -> Task:
        return Task(
            description="Review all materials for consistency and alignment",
            expected_output="Review report with suggested improvements",
            agent=self.feedback_agent(),
            context=[
                self.create_final_outline(),
                self.create_professor_guide(),
                self.create_presentation_slides(),
                self.create_student_handout()
            ]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.content_developer(),
                self.feedback_agent(),
                self.materials_creator()
            ],
            tasks=[
                self.create_initial_outline(),
                self.review_initial_outline(),
                self.create_final_outline(),
                self.create_professor_guide(),
                self.create_presentation_slides(),
                self.create_student_handout(),
                self.review_all_materials()
            ],
            process=Process.sequential,
            verbose=True
        )

    @after_kickoff
    def save_outputs(self, result):
        """Save task outputs to files after crew completion."""
        output_mapping = {
            'create_initial_outline': 'outputs/initial_outline.md',
            'review_initial_outline': 'outputs/outline_review.md',
            'create_final_outline': 'outputs/final_outline.md',
            'create_professor_guide': 'outputs/professor_guide.md',
            'create_presentation_slides': 'outputs/slides/presentation_content.md',
            'create_student_handout': 'outputs/student_handout.md',
            'review_all_materials': 'outputs/review/review_report.md'
        }

        for task_name, output_path in output_mapping.items():
            task = getattr(self, task_name)()
            if task.output:
                output_path = self.base_path / output_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(task.output.raw)

        return result