from pathlib import Path
import yaml
import os
import warnings
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
import litellm

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
        
        # Load environment variables
        load_dotenv()
        
        # Enable debug mode
        litellm.set_verbose = True
        
        # Simplify LLM configuration - remove extra parameters
        self.llm = LLM(
            model="claude-3-sonnet-20240229"  # Only specify the model
        )
        
        # Load YAML configurations
        self.config_path = Path(__file__).parent / "config"

    def _load_masterclass_concept(self):
        config_path = self.base_path / self.masterclass_config
        with open(config_path, 'r') as f:
            self.masterclass_concept = yaml.safe_load(f)
            # Set default language if not specified
            if 'language' not in self.masterclass_concept['concept']:
                self.masterclass_concept['concept']['language'] = 'spanish'

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

    def get_agents(self):
        """Create agents from YAML configuration"""
        with open(self.config_path / "agents.yaml", "r") as f:
            agents_config = yaml.safe_load(f)
            
        if isinstance(agents_config, dict):
            agents_list = agents_config.values()
        else:
            agents_list = agents_config
            
        return [
            Agent(
                role=agent_config['role'],
                goal=agent_config['goal'],
                backstory=agent_config['backstory'],
                llm=self.llm
            )
            for agent_config in agents_list
        ]

    def get_tasks(self):
        """Create tasks from YAML configuration"""
        with open(self.config_path / "tasks.yaml", "r") as f:
            tasks_config = yaml.safe_load(f)
            
        if isinstance(tasks_config, dict):
            tasks_list = tasks_config.values()
        else:
            tasks_list = tasks_config
            
        agents = self.get_agents()
        return [
            Task(
                description=task_config['description'],
                expected_output=task_config['expected_output'],
                agent=agents[0]
            )
            for task_config in tasks_list
        ]

    def get_crew(self):
        """Create and return the crew"""
        return Crew(
            agents=self.get_agents(),
            tasks=self.get_tasks(),
            llm=self.llm
        )

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