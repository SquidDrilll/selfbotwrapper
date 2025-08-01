# agentic.py
import os
import json
import re
import textwrap
from discord.ext import commands
from selfbot import SelfBot

from agno.agent import Agent
from agno.memory.v2.db.redis import RedisMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.groq import Groq
from agno.storage.redis import RedisStorage
from agno.tools.daytona import DaytonaTools

REGISTRY_FILE = "agentic_tools.json"

class AgenticLayer:
    def __init__(self, bot: SelfBot):
        self.bot = bot
        self.registry: Dict[str, str] = {}
        self.load_registry()
        self.agent = self._build_agent()

        @bot.event
        async def on_message(message):
            if message.author.id != bot.bot.user.id:
                return
            if message.content.startswith("."):
                text = message.content[1:].strip()
                if text:
                    if "write and register a tool" in text.lower():
                        await self._create_tool(message, text)
                    else:
                        response = await self.agent.arun(text)
                        for chunk in (response.content[i:i+1900] for i in range(0, len(response.content), 1900)):
                            await message.channel.send(chunk)

    async def _create_tool(self, message, text):
        tool_description = text.split("write and register a tool that", 1)[1].strip()
        if not tool_description:
            await message.channel.send("Please provide a description for the tool.")
            return

        tool_code = await self.agent.arun(f"Write a single async Python function named 'run' that {tool_description}.")
        if not tool_code.content:
            await message.channel.send("Failed to generate the tool code.")
            return

        function_name = re.findall(r"def\s+(\w+)\s*\(", tool_code.content)
        if not function_name:
            await message.channel.send("Failed to extract the function name from the generated code.")
            return
        function_name = function_name[0]

        self.registry[function_name] = tool_code.content
        self.save_registry()
        self.bot.bot.add_command(commands.Command(self._callable(function_name), name=function_name))
        await message.channel.send(f"✅ Tool `!{function_name}` registered. You can now use `!{function_name} <args>`.")

    def _callable(self, name: str):
        code = self.registry[name]
        loc = {}
        exec(textwrap.dedent(code), loc)
        return loc["run"]

    def load_registry(self):
        if os.path.isfile(REGISTRY_FILE):
            with open(REGISTRY_FILE) as f:
                self.registry = json.load(f)

    def save_registry(self):
        with open(REGISTRY_FILE, "w") as f:
            json.dump(self.registry, f, indent=2)

    def _build_agent(self):
        api_keys = {
            "groq": os.environ["GROQ_API_KEY"],
            "daytona": os.environ["DAYTONA_API_KEY"],
            "redis": os.environ["REDIS_PASSWORD"],
        }

        instructions = Path("instructions.txt").read_text()

        memory_db = RedisMemoryDb(
            prefix="agno_memory",
            host="aware-sawfly-6267.upstash.io",
            port=6379,
            password=api_keys["redis"],
            ssl=True,
        )
        storage = RedisStorage(
            prefix="agno_storage",
            host="aware-sawfly-6267.upstash.io",
            port=6379,
            password=api_keys["redis"],
            ssl=True,
        )

        tools = [
            DaytonaTools(api_key=api_keys["daytona"]),
        ]

        return Agent(
            name="Mist",
            model=Groq(id="moonshotai/kimi-k2-instruct", api_key=api_keys["groq"]),
            memory=Memory(db=memory_db),
            storage=storage,
            session_id="hero",
            tools=tools,
            description="A helpful personal assistant chatbot that can search the web, generate images, and remember user preferences.",
            instructions=instructions,
            add_history_to_messages=True,
            num_history_runs=3,
            markdown=True,
            show_tool_calls=True,
        )
