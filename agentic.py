
# agentic.py
import os
import json
import re
import textwrap
from pathlib import Path
from typing import Dict
from discord.ext import commands
from selfbot import SelfBot

from agno.agent import Agent
from agno.models.groq import Groq
from agno.memory.v2.db.redis import RedisMemoryDb
from agno.memory.v2.memory import Memory
from agno.storage.redis import RedisStorage
from agno.tools.calculator import CalculatorTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.file import FileTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.spider import SpiderTools
from agno.tools.yfinance import YFinanceTools

REGISTRY_FILE = "tools.json"
CHAT_HISTORY = "chat_history.json"

class AgenticLayer:
    def __init__(self, bot: SelfBot):
        self.bot = bot
        self.registry = self._load_json(REGISTRY_FILE, {})
        self.chats = self._load_json(CHAT_HISTORY, {})
        self.agent = self._build_agent()

        @bot.event
        async def on_message(message):
            if message.author.id != bot.bot.user.id:
                return
            text = message.content.lstrip(".")
            if not text:
                return

            cid = str(message.channel.id)
            self.chats.setdefault(cid, []).append({
                "author": str(message.author),
                "msg": message.content,
                "ts": message.created_at.isoformat()
            })
            self._save_json(CHAT_HISTORY, self.chats)

            try:
                reply = (await self.agent.arun(text)).content.strip()[:1900]
                await message.channel.send(reply)
            except Exception as e:
                await message.channel.send("Error: " + str(e)[:1900])

    def _load_json(self, file: str, default):
        return json.loads(Path(file).read_text()) if Path(file).exists() else default

    def _save_json(self, file: str, obj):
        Path(file).write_text(json.dumps(obj, indent=2))

        def _build_agent(self):
        instructions = Path("instructions.txt").read_text()

        redis_host = "usable-marmot-6518.upstash.io"
        redis_port = 6379
        redis_password = os.getenv("UPSTASH_REDIS_PASSWORD")

        memory_db = RedisMemoryDb(
            prefix="agno_memory",
            host=redis_host,
            port=redis_port,
            password=redis_password,
            ssl=True,
        )
        storage = RedisStorage(
            prefix="agno_storage",
            host=redis_host,
            port=redis_port,
            password=redis_password,
            ssl=True,
        )

        return Agent(
            name="Mist",
            model=Groq(id="moonshotai/kimi-k2-instruct", api_key=os.getenv("GROQ_API_KEY")),
            memory=Memory(db=memory_db),
            storage=storage,
            session_id="hero",
            tools=[
                DuckDuckGoTools(),
                GoogleSearchTools(),
                CalculatorTools(),
                FileTools(),
                SpiderTools(),
                YFinanceTools(
                    stock_price=True,
                    analyst_recommendations=True,
                    company_news=True,
                    stock_fundamentals=True,
                    company_info=True,
                ),
            ],
            instructions=instructions,
            add_history_to_messages=True,
            num_history_runs=1000,
            markdown=True,
            show_tool_calls=True,
            debug_mode=False,
        )
