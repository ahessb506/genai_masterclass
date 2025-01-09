from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew

@CrewBase
class MasterclassCrew:
    """Crew for developing a one-day GenAI masterclass for non-technical audiences"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def content_developer(self) -> Agent:
        return Agent(config=self.agents_config['content_developer'], verbose=True)

    @agent
    def exercise_designer(self) -> Agent:
        return Agent(config=self.agents_config['exercise_designer'], verbose=True)

    @agent
    def materials_creator(self) -> Agent:
        return Agent(config=self.agents_config['materials_creator'], verbose=True)

    @agent
    def engagement_specialist(self) -> Agent:
        return Agent(config=self.agents_config['engagement_specialist'], verbose=True)

    @agent
    def project_manager(self) -> Agent:
        return Agent(config=self.agents_config['project_manager'], verbose=True)

    @agent
    def feedback_agent(self) -> Agent:
        return Agent(config=self.agents_config['feedback_agent'], verbose=True)

    @task
    def develop_course_outline(self) -> Task:
        return Task(config=self.tasks_config['develop_course_outline'])

    @task
    def create_fundamental_content(self) -> Task:
        return Task(config=self.tasks_config['create_fundamental_content'])

    @task
    def design_practical_exercises(self) -> Task:
        return Task(config=self.tasks_config['design_practical_exercises'])

    @task
    def develop_interactive_activities(self) -> Task:
        return Task(config=self.tasks_config['develop_interactive_activities'])

    @task
    def create_supporting_materials(self) -> Task:
        return Task(config=self.tasks_config['create_supporting_materials'])

    @task
    def design_assessment_tools(self) -> Task:
        return Task(config=self.tasks_config['design_assessment_tools'])

    @task
    def manage_project_delivery(self) -> Task:
        return Task(config=self.tasks_config['manage_project_delivery'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )