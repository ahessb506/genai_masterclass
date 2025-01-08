from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew

@CrewBase
class MasterclassCrew:
    """Crew for developing a 3-hour GenAI masterclass"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def content_architect(self) -> Agent:
        return Agent(config=self.agents_config['content_architect'], verbose=True)

    @agent
    def summary_writer(self) -> Agent:
        return Agent(config=self.agents_config['summary_writer'], verbose=True)

    @agent
    def quiz_activity_designer(self) -> Agent:
        return Agent(config=self.agents_config['quiz_activity_designer'], verbose=True)

    @agent
    def feedback_agent(self) -> Agent:
        return Agent(config=self.agents_config['feedback_agent'], verbose=True)

    @agent
    def coordinator(self) -> Agent:
        return Agent(config=self.agents_config['coordinator'], verbose=True)

    @task
    def design_lesson_plan(self) -> Task:
        return Task(config=self.tasks_config['design_lesson_plan'])

    @task
    def create_slide_content(self) -> Task:
        return Task(config=self.tasks_config['create_slide_content'])

    @task
    def draft_reminder_document(self) -> Task:
        return Task(config=self.tasks_config['draft_reminder_document'])

    @task
    def develop_quizzes_activities(self) -> Task:
        return Task(config=self.tasks_config['develop_quizzes_activities'])

    @task
    def review_materials(self) -> Task:
        return Task(config=self.tasks_config['review_materials'])

    @task
    def coordinate_project(self) -> Task:
        return Task(config=self.tasks_config['coordinate_project'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )