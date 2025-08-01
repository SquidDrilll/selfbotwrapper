# agentic.py
import os
import json
import re
import textwrap
import aiohttp
import asyncio
from pathlib import Path
from typing import Dict
from discord.ext import commands
from selfbot import SelfBot

from agno.agent import Agent
from agno.memory.v2.db.redis import RedisMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.groq import Groq
from agno.storage.redis import RedisStorage
from agno.tools.calculator import CalculatorTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.file import FileTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.spider import SpiderTools

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
                if not text:
                    return

                # 1. CREATE / REGISTER NEW TOOL
                if text.lower().startswith("create tool"):
                    tool_desc = text[11:].strip()
                    await self._create_tool(message, tool_desc)
                # 2. LIST CURRENT TOOLS
                elif text.lower() == "list tools":
                    tools = ", ".join(self.registry.keys()) or "none"
                    await message.channel.send(f"**Tools:** {tools}")
                # 3. RUN ANY OTHER COMMAND
                else:
                    response = await self.agent.arun(text)
                    for chunk in (response.content[i:i+1900] for i in range(0, len(response.content), 1900)):
                        await message.channel.send(chunk)

    async def _create_tool(self, message, tool_desc):
        attempt = 0
        while True:
            attempt += 1
            try:
                # Ask agent for code
                prompt = (
                    f"Write a single async Python function named 'run' that {tool_desc}. "
                    "It must return a string and require **no API keys** and **no external packages**. "
                    "End with `return str(...)`."
                )
                tool_code = await self.agent.arun(prompt)
                code = tool_code.content.strip()
                name = re.findall(r"def\s+(\w+)\s*\(", code)[0]
                code = textwrap.dedent(code)

                # Compile test
                loc = {}
                exec(code, loc)
                func = loc[name]

                # Live test against httpbin
                async with aiohttp.ClientSession() as s:
                    res = await func(s, query="test")
                    if "error" in str(res).lower():
                        raise RuntimeError(res)

                # Success → register & break
                self.registry[name] = code
                self.save_registry()
                self.bot.bot.add_command(commands.Command(self._callable(name), name=name))
                await message.channel.send(f"✅ Tool `!{name}` is live.")
                break

            except Exception as e:
                await asyncio.sleep(1)  # small pause

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

        return Agent(
            name="Mist",
            model=Groq(id="moonshotai/kimi-k2-instruct", api_key=api_keys["groq"]),
            memory=Memory(db=memory_db),
            storage=storage,
            session_id="hero",
            tools=[CalculatorTools(), DuckDuckGoTools(), GoogleSearchTools(), FileTools(), SpiderTools()],
            description="A helpful personal assistant chatbot.",
            instructions=instructions,
            add_history_to_messages=True,
            num_history_runs=3,
            markdown=True,
            show_tool_calls=True,
        )
