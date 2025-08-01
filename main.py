# main.py
import os
from dotenv import load_dotenv
from selfbot import SelfBot
from tldr import setup_tldr
from agentic import AgenticLayer

load_dotenv()

bot = SelfBot(
    token=os.getenv("DISCORD_TOKEN"),
    prefix="!",
)

setup_tldr(bot)
AgenticLayer(bot)   # ← new layer

if __name__ == "__main__":
    bot.run()
