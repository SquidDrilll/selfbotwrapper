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

class AgenticLayer:
    def __init__(self, bot: SelfBot):
        self.bot = bot
        self.agent = self._build_agent()

        @bot.event
        async def on_message(message):
            if message.author.id != bot.bot.user.id:
                return
            if message.content.startswith("!"):
                text = message.content[1:].strip()
                if text:
                    response = await self.agent.arun(text)
                    for chunk in (response.content[i:i+1900] for i in range(0, len(response.content), 1900)):
                        await message.channel.send(chunk)

    def _build_agent(self):
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
            session_id="hero",
            tools=tools,
            instructions=instructions,
            add_history_to_messages=True,
            num_history_runs=3,
            markdown=True,
            show_tool_calls=True,
        )
