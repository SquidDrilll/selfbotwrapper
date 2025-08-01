# agentic.py
import os
import json
import re
import textwrap
import aiohttp
from pathlib import Path
from typing import Dict
from discord.ext import commands
from selfbot import SelfBot

# ---------- Agno imports ----------
from agno.agent import Agent
from agno.memory.v2.db.redis import RedisMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.groq import Groq
from agno.storage.redis import RedisStorage
from agno.tools.calculator import CalculatorTools
from agno.tools.crawl4ai import Crawl4aiTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.file import FileTools
from agno.tools.firecrawl import FirecrawlTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.replicate import ReplicateTools
from agno.tools.spider import SpiderTools
from agno.tools.website import WebsiteTools
from agno.tools.yfinance import YFinanceTools

REGISTRY_FILE = "agentic_tools.json"

class AgenticLayer:
    def __init__(self, bot: SelfBot):
        self.bot = bot
        self.registry: Dict[str, str] = {}
        self.load_registry()
        self.agent = self._build_agent()

        # expose auto-generated tools
        for name in self.registry:
            self.bot.bot.add_command(commands.Command(self._callable(name), name=name))

        # ---------- UNIVERSAL CHAT HANDLER ----------
        @bot.command(name="*", invoke_without_command=True)
        async def universal(ctx, *, text: str):
            if ctx.author.id != bot.bot.user.id:
                return
            response = await self.agent.arun(text)
            for chunk in (response.content[i:i+1900] for i in range(0, len(response.content), 1900)):
                await ctx.send(chunk)

    # ---------- persistence ----------
    def load_registry(self):
        if os.path.isfile(REGISTRY_FILE):
            with open(REGISTRY_FILE) as f:
                self.registry = json.load(f)

    def save_registry(self):
        with open(REGISTRY_FILE, "w") as f:
            json.dump(self.registry, f, indent=2)

    # ---------- build the full Agno agent ----------
    def _build_agent(self) -> Agent:
        api_keys = {
            "groq": os.environ["GROQ_API_KEY"],
            "firecrawl": os.environ["FIRECRAWL_API_KEY"],
            "replicate": os.environ["REPLICATE_API_KEY"],
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
            DuckDuckGoTools(),
            GoogleSearchTools(),
            CalculatorTools(
                add=True, subtract=True, multiply=True, divide=True,
                exponentiate=True, factorial=True, is_prime=True, square_root=True
            ),
            FileTools(),
            WebsiteTools(),
            SpiderTools(),
            FirecrawlTools(api_key=api_keys["firecrawl"]),
            Crawl4aiTools(),
            YFinanceTools(
                stock_price=True, analyst_recommendations=True,
                company_news=True, stock_fundamentals=True, company_info=True
            ),
            ReplicateTools(model="luma/photon-flash", api_key=api_keys["replicate"]),
        ]

        return Agent(
            name="Mist",
            model=Groq(id="moonshotai/kimi-k2-instruct", api_key=api_keys["groq"]),
            memory=Memory(db=memory_db),
            storage=storage,
            session_id="mist",
            tools=tools,
            instructions=instructions,
            add_history_to_messages=True,
            num_history_runs=3,
            markdown=True,
            show_tool_calls=True,
        )

    # ---------- meta commands ----------
    async def add_tool(self, ctx, *, spec: str):
        """!add an tool that will ..."""
        if ctx.author.id != self.bot.bot.user.id:
            return
        response = await self.agent.arun(spec)
        code = response.content.strip()
        name = re.findall(r"def\s+(\w+)\s*\(", code)[0]
        if name != "run":
            code = code.replace(name, "run", 1)
            name = "run"
        self.registry[name] = code
        self.save_registry()
        self.bot.bot.add_command(commands.Command(self._callable(name), name=name))
        await ctx.send(f"✅ Tool `!{name}` live!", delete_after=6)

    async def gen_image(self, ctx, *, prompt: str):
        """!gen an image of ..."""
        url = f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '%20')}?width=512&height=512"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as r:
                if r.status == 200:
                    await ctx.send(f"{ctx.author.mention} {url}")
                else:
                    await ctx.send("❌ Pollinations borked.", delete_after=4)

    # ---------- dynamic wrapper ----------
    def _callable(self, name: str):
        code = self.registry[name]
        loc = {}
        exec(textwrap.dedent(code), loc)
        return loc["run"]
