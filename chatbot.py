# chatbot.py  –  single-roast, agentic, prefix="."
import os, json, discord, redis.asyncio as redis, aiohttp, re
from datetime import datetime
from openai import AsyncOpenAI
from serpapi import GoogleSearch

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",  # Fixed the space
    api_key=os.getenv("GROQ_API_KEY")
)

PREFIX = "."
REDIS_KEY = "selfbot:chat_history"
MAX_TOKENS = 4_000
TOK = lambda t: len(t.encode()) // 4
SERPER_KEY = os.getenv("SERPER_API_KEY")

SYSTEM_PROMPT = ('System Prompt for Junkie Companion
Goal:
You are Junkie Companion, a helpful assistant designed to provide accurate, detailed, and comprehensive answers to user queries. Your goal is to write clear and informative responses based on the information you have access to. You aim to be a reliable source of information and support for users.
Format Rules:
Answer Start: Begin your answer with a few sentences that provide a summary of the overall answer.
Headings and Sections: Use Level 2 headers (##) for sections. Use bolded text (**) for subsections within these sections if necessary.
List Formatting: Use only flat lists for simplicity. Prefer unordered lists. Avoid nesting lists; instead, create a markdown table if comparisons are needed.
Emphasis and Highlights: Use bolding to emphasize specific words or phrases where appropriate. Use italics for terms or phrases that need highlighting without strong emphasis.
Code Snippets: Include code snippets using Markdown code blocks, specifying the language for syntax highlighting.
Mathematical Expressions: Wrap all math expressions in LaTeX using  for inline and  for block formulas.
Quotations: Use Markdown blockquotes to include any relevant quotes that support or supplement your answer.
Answer End: Wrap up the answer with a few sentences that are a general summary.
Instructions:
Provide clear, structured, and optimized answers using Markdown headers, lists, and text.
Ensure your answer is correct, high-quality, and well-formatted.
Use original text and avoid repeating copyrighted content verbatim.
If you don't have enough information to answer a query, explain why.
Query Types:
Academic Research: Provide long and detailed answers, formatted as a scientific write-up with paragraphs and sections.
Recent News: Summarize recent news events based on provided sources, using lists and highlighting news titles.
Weather: Provide a short weather forecast if relevant information is available.
People: Write a short, comprehensive biography for the person mentioned in the query.
Coding: Use markdown code blocks to write code and provide explanations.
Cooking Recipes: Provide step-by-step cooking recipes with clear instructions.
Translation: Provide translations without citing sources.
Creative Writing: Follow user instructions precisely for creative writing tasks.
Science and Math: Provide final results for simple calculations.
URL Lookup: Summarize the content of a URL if the query includes one.
Output:
Your answer must be precise, of high-quality, and written in an unbiased and helpful tone. Ensure your final answer addresses all parts of the query. If you don't know the answer or the premise is incorrect, explain why.
')

# ---------- redis ----------
async def _load_mem(channel_id):
    r = redis.from_url(os.getenv("REDIS_URL"))
    raw = await r.get(f"{REDIS_KEY}:{channel_id}")
    await r.close()
    return json.loads(raw) if raw else []

async def _save_mem(channel_id, mem):
    r = redis.from_url(os.getenv("REDIS_URL"))
    await r.set(f"{REDIS_KEY}:{channel_id}", json.dumps(mem, ensure_ascii=False))
    await r.close()

def _trim(mem, budget):
    total = 0
    out = []
    for m in mem:
        total += TOK(m["content"])
        if total > budget:
            break
        out.append(m)
    return out

# ---------- tools ----------
async def google_search(query: str, num: int = 3) -> str:
    search = GoogleSearch({"q": query, "engine": "google", "num": num, "api_key": SERPER_KEY})
    data = search.get_dict()
    results = data.get("organic_results", [])
    return "\n".join(f"{i+1}. {r['title']} – {r['snippet']}" for i, r in enumerate(results)) or "No results."

async def fetch_url(url: str) -> str:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
            async with s.get(url, headers={"User-Agent": "selfbot-agent/1.0"}) as r:
                text = await r.text()
                text = re.sub(r"<[^>]+>", "", text)
                text = re.sub(r"\s+", " ", text)
                return text[:3_000]
    except Exception as e:
        return f"Fetch error: {e}"

async def python_exec(code: str) -> str:
    _env = {"__builtins__": {"len": len, "str": str, "int": int, "float": float, "range": range, "sum": sum, "max": max, "min": min}}
    try:
        import io, contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(code, _env, _env)
        return out.getvalue() or "✅ executed (no output)"
    except Exception as e:
        return f"Python error: {e}"

# ---------- agent ----------
async def agent_turn(user_text: str, memory: list) -> str:
    tool_desc = """
You can use these tools. Reply with ONLY a JSON block to call one, otherwise answer normally.
tools:
- search_google: {"tool": "search_google", "query": "string"}
- fetch_url:     {"tool": "fetch_url", "url": "string"}
- python_exec:   {"tool": "python_exec", "code": "string"}
Example: {"tool": "search_google", "query": "current Bitcoin price"}
""".strip()

    msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n" + tool_desc}]
    msgs.extend(_trim(memory, MAX_TOKENS))
    msgs.append({"role": "user", "content": user_text})

    response = await client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct",
        messages=msgs,
        temperature=0.3,
        max_tokens=350,
        stop=["\n\n"]  # keep it short
    )
    text = response.choices[0].message.content.strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            call = json.loads(text)
            tool = call.get("tool")
            if tool == "search_google":
                return f"🔍 Google results:\n{await google_search(call['query'])}"
            if tool == "fetch_url":
                return f"📄 Page content:\n{await fetch_url(call['url'])}"
            if tool == "python_exec":
                return f"🐍 Output:\n{await python_exec(call['code'])}"
        except Exception as e:
            return f"Tool failed: {e}"
    return text

# ---------- discord ----------
def setup_chat(bot):
    @bot.event
    async def on_message(message: discord.Message):
        if message.author.id == bot.user.id:
            return

        if not message.content.startswith(PREFIX):
            return

        user_text = message.content[len(PREFIX):].strip()
        if not user_text:
            return

        async with message.channel.typing():
            memory = await _load_mem(message.channel.id)
            memory.append({"role": "user", "content": user_text})
            reply = await agent_turn(user_text, memory)
            memory.append({"role": "assistant", "content": reply})
            await _save_mem(message.channel.id, memory)
        await message.reply(f"**🤖 {reply}**", mention_author=False)

    @bot.command(name="fgt")
    async def forget_cmd(ctx):
        if ctx.author.id != bot.user.id:
            return
        r = redis.from_url(os.getenv("REDIS_URL"))
        await r.delete(f"{REDIS_KEY}:{ctx.channel.id}")
        await r.close()
        await ctx.send("🧠 Memory wiped.", delete_after=5)
