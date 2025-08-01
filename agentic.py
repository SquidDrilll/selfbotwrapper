# agentic.py
import os
import json
import re
import textwrap
from discord.ext import commands
from selfbot import SelfBot

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.calculator import CalculatorTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.file import FileTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.spider import SpiderTools
from agno.tools.yfinance import YFinanceTools
from agno.memory.v2.db.redis import RedisMemoryDb
from agno.storage.redis import RedisStorage

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

            # Record chat (DM / group)
            cid = str(message.channel.id)
            self.chats.setdefault(cid, []).append({
                "author": str(message.author),
                "msg": message.content,
                "ts": message.created_at.isoformat()
            })
            self._save_json(CHAT_HISTORY, self.chats)

            try:
                # 1. Create tool
                if text.lower().startswith("create tool"):
                    await self._create_tool(message, text[11:].strip())
                    return

                # 2. List tools
                if text.lower() == "list tools":
                    tools = ", ".join(self.registry.keys()) or "none"
                    await message.channel.send(f"Tools: {tools}")
                    return

                # 3. Chat / run tool
                response = await self.agent.arun(text)
                for chunk in (response.content[i:i+1900] for i in range(0, len(response.content), 1900)):
                    await message.channel.send(chunk)

            except Exception as e:
                print(f"Error processing message: {e}")  # Debugging log

    async def _create_tool(self, message, desc):
        while True:
            try:
                prompt = (
                    f"Write a single async Python function named 'run' that {desc}. "
                    "It must accept (ctx, *, query) and return str. "
                    "Use only aiohttp / requests / standard libs. "
                    "End with return str(...)."
                )
                code_raw = await self.agent.arun(prompt)
                code = textwrap.dedent(code_raw.content).strip()
                name = re.findall(r"def\s+(\w+)\s*\(", code)[0]

                # Local compile test
                loc = {}
                exec(code, loc)
                func = loc[name]

                # Remote sandbox test
                test_src = code + '\nimport asyncio; asyncio.run(run(None, query="test"))'
                payload = {"language": "python", "source": test_src}
                async with aiohttp.ClientSession() as s:
                    r = await s.post("https://emkc.org/api/v1/piston/execute", json=payload)
                    res = await r.json()
                    if res.get("run", {}).get("code") == 0:
                        break
            except Exception as e:
                print(f"Error creating tool: {e}")  # Debugging log
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

        # Redis setup with error handling
        try:
            redis_pass = os.getenv("REDIS_PASSWORD")
            if redis_pass:
                memory = Memory(
                    db=RedisMemoryDb(
                        prefix="agno_memory",
                        host="usable-marmot-6518.upstash.io",
                        port=6379,
                        password=redis_pass,
                        ssl=True,
                    )
                )
                storage = RedisStorage(
                    prefix="agno_storage",
                    host="usable-marmot-6518.upstash.io",
                    port=6379,
                    password=redis_pass,
                    ssl=True,
                )
            else:
                memory = None
                storage = None
        except Exception as e:
            print(f"Error setting up Redis: {e}")
            memory = None
            storage = None

        return Agent(
            name="Mist",
            model=Groq(id="moonshotai/kimi-k2-instruct", api_key=os.getenv("GROQ_API_KEY")),
            memory=memory,
            storage=storage,
            session_id="hero",
            tools=[
                CalculatorTools(),
                DuckDuckGoTools(),
                FileTools(),
                GoogleSearchTools(),
                SpiderTools(),
                YFinanceTools(
                    stock_price=True,
                    analyst_recommendations=True,
                    company_news=True,
                    stock_fundamentals=True,
                    company_info=True,
                ),
            ],
            description="A self-bot that remembers every chat and creates tools on demand.",
            instructions=instructions,
            add_history_to_messages=True,
            num_history_runs=1000,
            markdown=True,
            show_tool_calls=True,
            debug_mode=False,
        )
