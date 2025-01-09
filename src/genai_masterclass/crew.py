from pathlib import Path
import yaml
import os
import warnings
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task

# Suppress the TracerProvider warning globally
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                      message="Overriding of current TracerProvider is not allowed")

@CrewBase
class MasterclassCrew:
    """Crew for developing a GenAI masterclass for non-technical audiences"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    masterclass_config = 'config/masterclass_concept.yaml'

    def __init__(self):
        # Ensure warning is suppressed
        warnings.filterwarnings("ignore", category=RuntimeWarning, 
                              message="Overriding of current TracerProvider is not allowed")
        
        self.base_path = Path(__file__).parent
        self.output_path = self.base_path / 'outputs'
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
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

    def _save_output(self, filename: str, content: str):
        """Save content to a file in the outputs directory."""
        try:
            file_path = self.output_path / filename
            print(f"Attempting to save to: {file_path}")
            
            # Ensure content is a string
            content_str = str(content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content_str)
            
            print(f"Successfully saved output to: {file_path}")
            
            # Verify file was created
            if file_path.exists():
                print(f"Verified file exists: {file_path}")
            else:
                print(f"Warning: File was not created: {file_path}")
                
        except Exception as e:
            print(f"Error saving output to {filename}: {str(e)}")

    def _get_task_output(self, result, task_index):
        """Safely extract output from CrewOutput object."""
        try:
            if hasattr(result, 'outputs'):
                output = result.outputs[task_index]
            elif hasattr(result, 'values'):
                output = list(result.values())[task_index]
            elif isinstance(result, (list, tuple)):
                output = result[task_index]
            else:
                output = str(result)
            
            print(f"Successfully extracted output for task {task_index}")
            return output
            
        except Exception as e:
            print(f"Error accessing task {task_index} output: {e}")
            return str(result)

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
            
            # Save initial outputs
            self._save_output('initial_outline.md', initial_outline)
            self._save_output('ai_feedback.md', ai_feedback)
            
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
            self._save_output('revised_outline.md', current_outline)
            
        except Exception as e:
            print(f"Note: Error accessing results: {e}")
            current_outline = str(initial_results)
        
        # Step 3: Begin human feedback loop
        iteration = 1
        while True:
            print("\nCurrent outline after incorporating feedback:")
            print(current_outline)
            
            user_input = input("\nType 'approved' to continue with the current outline, or provide additional feedback for revision: ")
            
            if user_input.lower() == 'approved':
                self._save_output('final_outline.md', current_outline)
                break
            
            # Save human feedback
            self._save_output(f'human_feedback_{iteration}.md', user_input)
            
            print("\nRevising outline based on your feedback...")
            revision_crew = Crew(
                agents=[self.content_developer()],
                tasks=[self.revise_outline_with_human_feedback(user_input)],
                process=Process.sequential,
                verbose=True
            )
            
            revision_result = revision_crew.kickoff()
            try:
                current_outline = revision_result[0]
                self._save_output(f'revised_outline_{iteration}.md', current_outline)
                iteration += 1
            except Exception as e:
                print(f"Note: Error accessing revision result: {e}")
                current_outline = str(revision_result)

        print("\nOutline approved. Creating final materials...")
        self.approved_outline = current_outline
        
        # Create final materials with agent-specific format instructions
        final_tasks = [
            Task(
                description="""Create a comprehensive professor's guide for this masterclass.

APPROVED COURSE OUTLINE:
{}

REQUIRED CONTENT:
Create a detailed teaching guide document that includes:
1. Detailed explanations for each section of the course
2. Step-by-step teaching instructions with timing
3. Specific activities and exercises with implementation guidelines
4. Anticipated student questions and prepared answers
5. Tips for maintaining engagement and handling discussions
6. Additional resources and references for each topic
7. Assessment criteria and grading guidelines

Please provide your response in this exact format:

Thought: I will create a comprehensive professor's guide based on the approved outline.
Final Answer: 

# Professor's Guide for [Course Title]

[Rest of the guide content following the requirements above]""".format(self.approved_outline),
                expected_output="A comprehensive professor's guide in Markdown format",
                agent=self.content_developer()
            ),
            Task(
                description="""Create presentation slides for this masterclass.

APPROVED COURSE OUTLINE:
{}

REQUIRED CONTENT:
Create a complete slide deck content that includes:
1. Title slide and introduction
2. Clear slides for each section of the outline
3. Key points and concepts for each topic
4. Examples and analogies to explain complex ideas
5. Interactive elements and discussion prompts
6. Visual element descriptions
7. Speaker notes for each slide

Please provide your response in this exact format:

Thought: I will create a complete slide deck based on the approved outline.
Final Answer: 

# Presentation Slides

## Slide 1: Title
Content: [Slide content]
Speaker Notes: [Notes for presenter]

[Continue with remaining slides]""".format(self.approved_outline),
                expected_output="Slide-by-slide content in Markdown format",
                agent=self.materials_creator()
            ),
            Task(
                description="""Create a student handout for this masterclass.

APPROVED COURSE OUTLINE:
{}

REQUIRED CONTENT:
Create a comprehensive student reference guide that includes:
1. Executive summary of the course
2. Key concepts and definitions for each section
3. Practical tips and best practices
4. Common pitfalls and how to avoid them
5. Hands-on exercises and practice problems
6. Resources for further learning
7. Glossary of important terms
8. Note-taking sections

Please provide your response in this exact format:

Thought: I will create a comprehensive student handout based on the approved outline.
Final Answer: 

# Student Handout

## Course Summary
[Executive summary]

[Continue with remaining sections]""".format(self.approved_outline),
                expected_output="A concise student handout in Markdown format",
                agent=self.materials_creator()
            )
        ]

        final_crew = Crew(
            agents=[
                self.content_developer(),
                self.materials_creator()
            ],
            tasks=final_tasks,
            process=Process.sequential,
            verbose=True
        )

        # Execute final tasks and save outputs
        print("\nExecuting final tasks...")
        final_results = final_crew.kickoff()
        
        print("\nSaving final outputs...")
        output_files = [
            ('professor_guide.md', 0),
            ('presentation_slides.md', 1),
            ('student_handout.md', 2)
        ]
        
        for filename, index in output_files:
            print(f"\nProcessing {filename}...")
            content = self._get_task_output(final_results, index)
            if content:
                print(f"Content extracted for {filename}, saving...")
                self._save_output(filename, content)
            else:
                print(f"No content extracted for {filename}")

        return final_crew