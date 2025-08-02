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

REGISTRY_FILE   = "tools.json"
CHAT_HISTORY    = "chat_history.json"

class AgenticLayer:
    def __init__(self, bot: SelfBot):
        self.bot      = bot
        self.registry = self._load_json(REGISTRY_FILE, {})   # auto-tools
        self.chats    = self._load_json(CHAT_HISTORY, {})    # DM / group history
        self.agent    = self._build_agent()

        # expose auto-generated tools
        for name in self.registry:
            self.bot.bot.add_command(commands.Command(self._callable(name), name=name))

        @bot.event
        async def on_message(message):
            if message.author.id != bot.bot.user.id:
                return

            text = message.content.lstrip(".")
            if not text:
                return

            cid = str(message.channel.id)
            # record chat
            self.chats.setdefault(cid, []).append({
                "author": str(message.author),
                "msg":    message.content,
                "ts":     message.created_at.isoformat()
            })
            self._save_json(CHAT_HISTORY, self.chats)

            try:
                # 1. dynamic tool creation
                if text.lower().startswith("create tool"):
                    await self._create_tool(message, text[11:].strip())
                    return

                # 2. list tools
                if text.lower() == "list tools":
                    tools = ", ".join(self.registry.keys()) or "none"
                    await message.channel.send(f"Tools: {tools}")
                    return

                # 3. single detailed reply
                reply = (await self.agent.arun(text)).content.strip()[:1900]
                await message.channel.send(reply)

            except Exception as e:
                await message.channel.send("Error: " + str(e)[:1900])

    # ---------- auto-tool engine ----------
    async def _create_tool(self, message, desc):
        while True:
            try:
                prompt = (
                    f"Write an async Python function named 'run' that {desc}. "
                    "Signature: async def run(ctx, *, query='') -> str. "
                    "Use aiohttp or standard libs only."
                )
                code_raw = await self.agent.arun(prompt)
                code = textwrap.dedent(code_raw.content).strip()
                name = re.findall(r"def\s+(\w+)\s*\(", code)[0]

                # sandbox test
                test_src = code + '\nimport asyncio; asyncio.run(run(None, query="test"))'
                payload  = {"language": "python", "source": test_src}
                async with aiohttp.ClientSession() as s:
                    r = await s.post("https://emkc.org/api/v1/piston/execute", json=payload)
                    res = await r.json()
                    if res.get("run", {}).get("code") == 0:
                        break
            except Exception:
                await asyncio.sleep(1)

            self.registry[name] = code
            self._save_json(REGISTRY_FILE, self.registry)
            self.bot.bot.add_command(commands.Command(self._callable(name), name=name))
            await message.channel.send(f"✅ Tool `!{name}` ready.")

    def _callable(self, name: str):
        code = self.registry[name]
        loc = {}
        exec(textwrap.dedent(code), loc)
        return loc[name]

    def _load_json(self, file: str, default):
        if Path(file).exists():
            with open(file, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    def _save_json(self, file: str, obj):
        with open(file, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def _build_agent(self):
        instructions = Path("instructions.txt").read_text()

        redis_url = os.getenv("UPSTASH_REDIS_URL")
        if redis_url:
            memory_db = RedisMemoryDb(
                prefix="agno_memory",
                url=redis_url,
                ssl=True,
            )
            storage = RedisStorage(
                prefix="agno_storage",
                url=redis_url,
                ssl=True,
            )
        else:
            memory_db = None
            storage = None

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
                ReplicateTools(model="luma/photon-flash", api_key=os.getenv("REPLICATE_API_KEY")),
            ],
            instructions=instructions,
            add_history_to_messages=True,
            num_history_runs=1000,
            markdown=True,
            show_tool_calls=True,
            debug_mode=False,
        )

if __name__ == "__main__":
    bot = SelfBot(token=os.getenv("DISCORD_TOKEN"), prefix=".")
    AgenticLayer(bot)
    bot.run()
