import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from query_data import PROMPT_TEMPLATE
from langchain.prompts import ChatPromptTemplate

load_dotenv()


TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# Configura logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hola! Envíame una pregunta y te mostraré el prompt que se enviaría al modelo.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text

    # Simula el uso de query_rag para obtener el prompt real (sin llamar modelo)
    try:
        # Llama a query_rag pero solo genera el prompt, sin ejecutar el modelo
        # Copiamos la lógica relevante de query_rag para obtener el contexto y el prompt
        import chromadb
        from chromadb.config import Settings
        from get_embedding_function import get_embedding_function
        from langchain_chroma import Chroma
        from transformers import AutoTokenizer

        CHROMA_PATH = "chroma"
        embedding_function = get_embedding_function()
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        db = Chroma(
            client=chroma_client,
            collection_name="my_collection",
            embedding_function=embedding_function
        )
        num_docs = 3
        results = db.similarity_search_with_score(user_message, k=num_docs)
        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=user_message)
        await update.message.reply_text(f"Prompt generado (simulando --use-local-db):\n\n{prompt}")
    except Exception as e:
        await update.message.reply_text(f"Error generando el prompt real: {str(e)}")

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Bot corriendo. Presiona Ctrl+C para detener.")
    app.run_polling()
