import os
import asyncio
from openai import AsyncOpenAI
from agents import Agent, Runner, RunConfig
from agents.models.openai_provider import OpenAIProvider

from dotenv import load_dotenv
load_dotenv(override=True)

# Définissez votre clé API OpenRouter ici. Remplacez "VOTRE_CLE_API_OPENROUTER_ICI" par votre clé réelle.
os.environ["OPENROUTER_API_KEY"] = "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"
# Définissez également l'URL de base pour OpenRouter afin que la bibliothèque agents la prenne en compte.
os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"

# 1. Check for the OpenRouter API key
if "OPENROUTER_API_KEY" not in os.environ:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

# 2. Create a custom AsyncOpenAI client for OpenRouter
openrouter_client = AsyncOpenAI(
    base_url=os.environ["OPENROUTER_API_BASE"],
    api_key=os.environ["OPENROUTER_API_KEY"],
    
)

# 3. Wrap the client in an OpenAIProvider
openrouter_provider = OpenAIProvider(openai_client=openrouter_client)


# Define the Task Generator agent
task_generator = Agent(
    name="Task Generator",
    instructions="""You help users break down their specific LLM powered AI Agent goal into small, achievable tasks.
    For any goal, analyze it and create a structured plan with specific actionable steps.
    Each task should be concrete, time-bound when possible, and manageable.
    Organize tasks in a logical sequence with dependencies clearly marked.
    Never answer anything unrelated to AI Agents.""",
    # Specify the model for OpenRouter
    model="openai/gpt-3.5-turbo",
)


# Define a function to run the agent with the correct provider
async def generate_tasks(goal):
    # 4. Pass the provider to the runner via RunConfig
    result = await Runner.run(
        task_generator, 
        goal, 
        run_config=RunConfig(model_provider=openrouter_provider)
    )
    return result.final_output


# Example usage
async def main():
    user_goal = "Start a small online business selling handmade jewelry"
    tasks = await generate_tasks(user_goal)
    print("--- Generated Task Plan ---")
    print(tasks)

# 3. Output the agent's answer
if __name__ == "__main__":
    asyncio.run(main())
