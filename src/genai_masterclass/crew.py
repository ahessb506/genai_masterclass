from pathlib import Path
import yaml
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task

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
    def revise_outline_with_human_feedback(self, feedback: str) -> Task:
        """Create a task for revising the outline based on human feedback only."""
        return Task(
            description=f"Revise the outline based on this human feedback:\n\n{feedback}\n\nMake sure to address all the points raised in the feedback.",
            expected_output="Revised course outline in Markdown format",
            agent=self.content_developer(),
            context=[self.create_initial_outline()]
        )

    def get_crew(self) -> Crew:
        """Create and return the crew with iterative outline review."""
        # Step 1: Create initial outline and get AI review
        initial_crew = Crew(
            agents=[self.content_developer(), self.feedback_agent()],
            tasks=[
                self.create_initial_outline(),
                self.review_initial_outline()
            ],
            process=Process.sequential,
            verbose=True
        )

        print("\nCreating initial outline and getting AI review...")
        initial_results = initial_crew.kickoff()
        
        try:
            initial_outline = initial_results.get_task_output(self.create_initial_outline())
            ai_feedback = initial_results.get_task_output(self.review_initial_outline())
            
            print("\nInitial AI Feedback received. Incorporating feedback...")
            
            # Step 2: Have content developer incorporate AI feedback
            revision_crew = Crew(
                agents=[self.content_developer()],
                tasks=[
                    Task(
                        description=f"Review and incorporate this AI feedback into the outline:\n\nOriginal Outline:\n{initial_outline}\n\nAI Feedback:\n{ai_feedback}",
                        expected_output="Revised course outline in Markdown format",
                        agent=self.content_developer()
                    )
                ],
                process=Process.sequential,
                verbose=True
            )
            
            revision_result = revision_crew.kickoff()
            current_outline = revision_result.get_task_output(revision_crew.tasks[0])
            
        except Exception as e:
            print(f"Note: Error accessing results: {e}")
            current_outline = str(initial_results)
        
        # Step 3: Begin human feedback loop
        while True:
            print("\nCurrent outline after incorporating AI feedback:")
            print(current_outline)
            
            user_input = input("\nType 'approved' to continue with the current outline, or provide additional feedback for revision: ")
            
            if user_input.lower() == 'approved':
                break
            
            print("\nRevising outline based on your feedback...")
            # Create a new crew for human-feedback revision
            revision_crew = Crew(
                agents=[self.content_developer()],
                tasks=[self.revise_outline_with_human_feedback(user_input)],
                process=Process.sequential,
                verbose=True
            )
            
            revision_result = revision_crew.kickoff()
            try:
                current_outline = revision_result.get_task_output(revision_crew.tasks[0])
            except Exception as e:
                print(f"Note: Error accessing revision result: {e}")
                current_outline = str(revision_result)

        print("\nOutline approved. Creating final materials...")
        
        # Store the approved outline for use in final materials
        self.approved_outline = current_outline
        
        # Create final materials once outline is approved
        final_tasks = [
            self.create_professor_guide(),
            self.create_presentation_slides(),
            self.create_student_handout(),
            self.review_all_materials()
        ]

        return Crew(
            agents=[
                self.content_developer(),
                self.feedback_agent(),
                self.materials_creator()
            ],
            tasks=final_tasks,
            process=Process.sequential,
            verbose=True
        )

    @task
    def create_professor_guide(self) -> Task:
        return Task(
            description=f"Based on this approved outline:\n\n{self.approved_outline}\n\nCreate a comprehensive teaching guide that includes:\n- Detailed explanations for each section\n- Teaching instructions and timing\n- Activities and guidelines\n- Common questions and answers\n- Tips for engagement",
            expected_output="A comprehensive professor's guide in Markdown format",
            agent=self.content_developer()
        )

    @task
    def create_presentation_slides(self) -> Task:
        return Task(
            description=f"Based on this approved outline:\n\n{self.approved_outline}\n\nCreate presentation slides that:\n- Follow the established progression\n- Use concise, non-technical language\n- Include visual elements and interaction points\n- Highlight key concepts and examples",
            expected_output="Slide-by-slide content in Markdown format",
            agent=self.materials_creator()
        )

    @task
    def create_student_handout(self) -> Task:
        return Task(
            description=f"Based on this approved outline:\n\n{self.approved_outline}\n\nCreate a concise reference guide that includes:\n- Key concepts and definitions\n- Practical tips and best practices\n- Common pitfalls to avoid\n- Resources for further learning",
            expected_output="A concise summary document in Markdown format",
            agent=self.materials_creator()
        )

    @task
    def review_all_materials(self) -> Task:
        return Task(
            description=f"Based on this approved outline:\n\n{self.approved_outline}\n\nReview all materials to ensure:\n- Alignment with the approved outline\n- Consistency across all documents\n- Appropriate level for the target audience\n- Effective progression of concepts",
            expected_output="Review report with suggested improvements",
            agent=self.feedback_agent()
        )