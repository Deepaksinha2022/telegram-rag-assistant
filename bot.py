from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import google.generativeai as genai


TOKEN = "TOKEN"
GEMINI_API_KEY = "gemini_api_key"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I am your AI Agent.")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text

    response = model.generate_content(user_message)

    await update.message.reply_text(response.text)

app = Application.builder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

print("Bot is running...")
app.run_polling()