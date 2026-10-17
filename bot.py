import os
import re
import logging
import subprocess
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pdfbot")

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
MODEL = os.getenv("MODEL", "claude-sonnet-5-5")
PDF_DIR = Path(os.getenv("PDF_DIR", "pdfs"))
PDF_REPO_URL = os.getenv("PDF_REPO_URL")  # e.g. https://github.com/you/your-pdfs.git
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # only needed for private repos
TOP_K = int(os.getenv("TOP_K", "6"))

# Comma-separated Telegram user IDs allowed to use the bot. Leave empty to allow
# everyone (not recommended: strangers could run up your API bill).
ALLOWED = {
    int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()
}

# Reads ANTHROPIC_API_KEY from the environment automatically.
client = anthropic.AsyncAnthropic()

SYSTEM_PROMPT = (
    "You answer questions using only the documents provided in the user's message. "
    "If the answer is not in the documents, say you couldn't find it. "
    "Mention which document(s) you used. Keep answers concise; they are read in a chat app."
)


def sync_pdfs() -> None:
    """Clone or update the GitHub repo containing the PDFs."""
    if not PDF_REPO_URL:
        PDF_DIR.mkdir(exist_ok=True)
        return
    url = PDF_REPO_URL
    if GITHUB_TOKEN and url.startswith("https://"):
        url = url.replace("https://", f"https://{GITHUB_TOKEN}@", 1)
    if (PDF_DIR / ".git").exists():
        subprocess.run(["git", "-C", str(PDF_DIR), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", url, str(PDF_DIR)], check=True)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_documents() -> tuple[list[dict], BM25Okapi]:
    docs = []
    for path in sorted(PDF_DIR.rglob("*.pdf")):
        try:
            reader = PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        except Exception as e:
            log.warning("Could not read %s: %s", path, e)
            continue
        if not text:
            log.warning("No extractable text in %s (scanned? needs OCR)", path)
            continue
        docs.append({"name": path.name, "text": text})
    if not docs:
        raise SystemExit(f"No readable PDFs found in {PDF_DIR}")
    log.info("Loaded %d PDFs", len(docs))
    index = BM25Okapi([tokenize(d["text"]) for d in docs])
    return docs, index


sync_pdfs()
DOCS, INDEX = load_documents()


def retrieve(query: str) -> list[dict]:
    scores = INDEX.get_scores(tokenize(query))
    ranked = sorted(range(len(DOCS)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    return [DOCS[i] for i in ranked if scores[i] > 0] or [DOCS[i] for i in ranked[:3]]


def allowed(update: Update) -> bool:
    return not ALLOWED or (update.effective_user and update.effective_user.id in ALLOWED)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await update.message.reply_text("Hi! Ask me a question about the documents.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    query = update.message.text
    await context.bot.send_chat_action(update.effective_chat.id, "typing")

    docs = retrieve(query)
    context_block = "\n\n".join(
        f'<document name="{d["name"]}">\n{d["text"]}\n</document>' for d in docs
    )
    try:
        resp = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"{context_block}\n\nQuestion: {query}",
                }
            ],
        )
        answer = resp.content[0].text
    except Exception:
        log.exception("LLM call failed")
        answer = "Sorry, something went wrong. Try again in a moment."

    # Telegram messages max out at 4096 characters.
    for i in range(0, len(answer), 4000):
        await update.message.reply_text(answer[i : i + 4000])


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
