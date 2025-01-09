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
            # Access results directly from the list
            initial_outline = initial_results[0]
            ai_feedback = initial_results[1]
            
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
            current_outline = revision_result[0]
            
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
                current_outline = revision_result[0]
            except Exception as e:
                print(f"Note: Error accessing revision result: {e}")
                current_outline = str(revision_result)

        print("\nOutline approved. Creating final materials...")
        
        # Store the approved outline for use in final materials
        self.approved_outline = current_outline
        
        # Create final materials with specific instructions
        final_tasks = [
            Task(
                description=f"""Create a comprehensive professor's guide based on this approved outline:

{self.approved_outline}

Your task is to create a detailed teaching guide that includes:
1. Detailed explanations for each section of the course
2. Step-by-step teaching instructions with timing
3. Specific activities and exercises with implementation guidelines
4. Anticipated student questions and prepared answers
5. Tips for maintaining engagement and handling discussions
6. Additional resources and references for each topic
7. Assessment criteria and grading guidelines""",
                expected_output="A comprehensive professor's guide in Markdown format",
                agent=self.content_developer()
            ),
            Task(
                description=f"""Create presentation slides based on this approved outline:

{self.approved_outline}

Your task is to create engaging presentation content that:
1. Follows the outline's progression logically
2. Uses clear, non-technical language appropriate for the audience
3. Includes specific points for visual elements and graphics
4. Highlights key concepts with examples and analogies
5. Incorporates interactive elements and discussion points
6. Provides clear transitions between topics
7. Includes speaker notes with delivery suggestions""",
                expected_output="Slide-by-slide content in Markdown format",
                agent=self.materials_creator()
            ),
            Task(
                description=f"""Create a student handout based on this approved outline:

{self.approved_outline}

Your task is to create a concise reference guide that:
1. Summarizes key concepts and definitions
2. Provides practical tips and best practices
3. Lists common pitfalls and how to avoid them
4. Includes hands-on exercises and practice problems
5. Offers resources for further learning
6. Contains a glossary of important terms
7. Includes space for notes and reflections""",
                expected_output="A concise student handout in Markdown format",
                agent=self.materials_creator()
            ),
            Task(
                description=f"""Review all the created materials in the context of this approved outline:

{self.approved_outline}

Your task is to perform a comprehensive review ensuring:
1. Perfect alignment with the approved outline
2. Consistency in terminology and concepts across all materials
3. Appropriate difficulty level for the target audience
4. Logical progression of concepts and learning objectives
5. Effectiveness of exercises and activities
6. Clarity and completeness of explanations
7. Proper coverage of all topics
8. Engagement level of materials

Provide specific feedback for improvements if needed.""",
                expected_output="Detailed review report with specific recommendations",
                agent=self.feedback_agent()
            )
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