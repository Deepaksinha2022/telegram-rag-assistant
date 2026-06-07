from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
import chromadb
from rag_chroma import ask_question
import os
from ingest import ingest_pdf


# Telegram Bot Token
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")

# Active document collection
ACTIVE_COLLECTION = "The-Metamorphosis-Franz-Kafka"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I am your AI RAG Assistant."
    )

async def use_collection(update, context):

    global ACTIVE_COLLECTION

    try:
        collection_name = " ".join(
            context.args
        )

        ACTIVE_COLLECTION = collection_name

        await update.message.reply_text(
            f"Active collection changed to:\n{ACTIVE_COLLECTION}"
        )

    except Exception as e:

        await update.message.reply_text(
            f"Error: {str(e)}"
        )

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_message = update.message.text

    try:
        print("Collection:", ACTIVE_COLLECTION)
        print("User Question:", user_message)
        answer = ask_question(
            ACTIVE_COLLECTION,
            user_message
        )

        await update.message.reply_text(answer)

    except Exception as e:

        await update.message.reply_text(
            f"Error: {str(e)}"
        )

async def collections(update, context):
    client = chromadb.PersistentClient(
    path="./chroma_db"
)

    collections = client.list_collections()

    message = "Available Collections:\n\n"

    for collection in collections:
        message += f"- {collection.name}\n"

    await update.message.reply_text(message)

# Create application
app = Application.builder().token(TOKEN).build()

# Handlers
app.add_handler(CommandHandler("start", start))
app.add_handler(
    MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        echo
    )
)
app.add_handler(
    CommandHandler(
        "collections",
        collections
    )
)

app.add_handler(
    CommandHandler(
        "use",
        use_collection
    )
)

async def handle_document(update, context):

    document = update.message.document

    file = await context.bot.get_file(
        document.file_id
    )

    file_path = os.path.join(
        "uploads",
        document.file_name
    )

    await file.download_to_drive(
        file_path
    )
    await update.message.reply_text(
    "Processing PDF..."
)
    ingest_pdf(file_path)

    await update.message.reply_text(
    f"""
PDF saved and indexed successfully.

Collection:
{document.file_name.replace('.pdf', '')}
"""
)

app.add_handler(
    MessageHandler(
        filters.Document.PDF,
        handle_document
    )
)


print("Bot is running...")

app.run_polling()