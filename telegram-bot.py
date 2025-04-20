import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from query_data import PROMPT_TEMPLATE
from langchain.prompts import ChatPromptTemplate
import datetime

load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# Configura logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Estructura para manejar múltiples conversaciones
conversation_history = {}

# Clase para manejar el estado de una conversación


class Conversation:
    def __init__(self):
        # Lista de mensajes (cada mensaje tendrá 'role', 'content')
        self.messages = []
        self.context = None  # Contexto actual de la conversación
        self.last_response = None  # Última respuesta generada
        self.last_query_time = None  # Timestamp de la última interacción

    def add_message(self, role, content):
        """Agrega un mensaje a la conversación"""
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.datetime.now()
        })
        self.last_query_time = datetime.datetime.now()

    def get_recent_messages(self, limit=10):
        """Obtiene los últimos mensajes"""
        return self.messages[-limit:]

    def clear_history(self):
        """Limpia el historial de mensajes"""
        self.messages = []
        self.context = None
        self.last_response = None

    def __str__(self):
        """Retorna una representación legible de la conversación"""
        messages_str = "\n".join([
            f"[{msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] {msg['role']}: {msg['content']}"
            for msg in self.messages
        ])
        return f"""
                    Estado de la conversación:
                    - Última consulta: {self.last_query_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_query_time else 'Nunca'}
                    - Mensajes:
                    {messages_str}
                    - Contexto actual: {'Sí' if self.context else 'No'}
                    - Última respuesta: {'Sí' if self.last_response else 'No'}
                """


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hola! Envíame una pregunta y te mostraré el prompt que se enviaría al modelo.')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_message = update.message.text

    # Inicializar conversación si no existe
    if user_id not in conversation_history:
        conversation_history[user_id] = Conversation()

    # Agregar mensaje del usuario al historial
    conversation_history[user_id].add_message('user', user_message)

    try:
        # Llama a query_rag pero solo genera el prompt, sin ejecutar el modelo
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
        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=context_text, question=user_message)

        # Guardar contexto y respuesta en la conversación
        conversation_history[user_id].context = context_text
        conversation_history[user_id].last_response = prompt
        conversation_history[user_id].add_message('assistant', prompt)

        print("Estado de todas las conversaciones:")
        for user_id, conv in conversation_history.items():
            print(f"Usuario {user_id}:")
            print(conv)

        await update.message.reply_text(f"Prompt generado (simulando --use-local-db):\n\n{prompt}")
    except Exception as e:
        await update.message.reply_text(f"Error generando el prompt real: {str(e)}")

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(
        filters.TEXT & (~filters.COMMAND), handle_message))

    print("Bot corriendo. Presiona Ctrl+C para detener.")
    app.run_polling()
