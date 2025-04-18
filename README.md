# RAG (Retrieval-Augmented Generation) System

Este proyecto implementa un sistema RAG que permite consultar documentos PDF utilizando diferentes modelos de embedding y lenguaje.

## Características

- Procesamiento de documentos PDF
- Múltiples opciones de modelos de embedding:
  - all-MiniLM-L12-v2 (ligero y eficiente)
  - BAAI/bge-m3 (multilingüe y potente)
  - hkunlp/instructor-large (con capacidad de instrucciones)
- Soporte para diferentes modelos de lenguaje:
  - Modelos FLAN-T5 locales (small, base, large, xl)
  - API externa para generación de respuestas
- Sistema de chunking inteligente con superposición
- Base de datos vectorial con Chroma

## Requisitos

```bash
pip install langchain-community langchain-huggingface chromadb sentence-transformers transformers torch requests
```

## Estructura del Proyecto

- `populate_database.py`: Procesa los documentos PDF y los almacena en la base de datos
- `query_data.py`: Maneja las consultas y genera respuestas
- `get_embedding_function.py`: Gestiona los diferentes modelos de embedding

## Uso

### 1. Preparar los Documentos

Coloca tus archivos PDF en el directorio `data/`.

### 2. Poblar la Base de Datos

```bash
# Crear nueva base de datos multilingual por default
python populate_database.py --reset

# Actualizar base de datos existente
python populate_database.py
```

### 3. Realizar Consultas

```bash
# Consulta básica
python query_data.py "Tu pregunta aquí"

# Opciones adicionales
python query_data.py "Tu pregunta" \
    --model xl \           # Modelo a utilizar (small, base, large, xl)
    --docs 3 \            # Número de documentos para contexto (1-5)
    --use-local \         # Usar modelo local en lugar de API
    --max-tokens 2048     # Límite de tokens para la respuesta (solo aplica API)
```

## Configuración de Modelos

### Modelos de Embedding Disponibles

1. **minil12** (all-MiniLM-L12-v2)
   - Ligero y eficiente
   - Buen balance entre rendimiento y recursos

2. **multilingual** (BAAI/bge-m3)
   - Soporte multilingüe
   - Mayor potencia y precisión
   - Configurado para CPU por defecto

3. **instruction** (hkunlp/instructor-large)
   - Permite instrucciones personalizadas
   - Mayor consumo de recursos
   - Optimizado para documentos en español

### Modelos de Lenguaje

- **Locales** (FLAN-T5):
  - small: 512 tokens máx.
  - base: 768 tokens máx.
  - large: 1024 tokens máx.
  - xl: 2048 tokens máx.

- **API Externa**:
  - Límite configurable de tokens
  - Requiere API key válida

## Notas Importantes

- El sistema ajusta automáticamente el contexto si se excede el límite de tokens
- Los documentos se dividen en chunks de 800 caracteres con 80 caracteres de superposición
- La base de datos se puede resetear o actualizar según necesidad
- Se recomienda usar el modelo de embedding según el caso de uso:
  - minil12: para casos generales y recursos limitados
  - multilingual: para documentos multilingües
  - instruction: para casos que requieren instrucciones específicas

# FLASK API CONFIGURATION
## Inside env
To run your web application, you’ll first tell Flask where to find the application (api.py) with the FLASK_APP environment variable:

```bash
export FLASK_APP=api
```

Then run it in development mode with the FLASK_ENV environment variable:
```bash
export FLASK_ENV=development
```

Run api
```bash
flask run
```

# BOT DE TELEGRAM: `telegram-bot.py`

Este archivo implementa un bot de Telegram que permite probar el sistema RAG desde un chat de Telegram. Cuando envías una pregunta al bot, este genera el prompt real que se enviaría al modelo, usando el contexto recuperado de la base de datos local (simula la opción `--use-local-db`). El bot **no ejecuta el modelo ni llama a la API**, solo muestra el prompt construido.

## ¿Cómo funciona?
- Usa la función de recuperación de contexto del sistema (`Chroma`, embeddings, etc.) para buscar los documentos más relevantes según tu pregunta.
- Construye el prompt con el contexto real y la pregunta, igual que lo haría el sistema antes de llamar a un modelo LLM.
- Te responde en Telegram con el prompt generado (puedes copiarlo y usarlo para pruebas o inspección).

## Requisitos
- Tener la base de datos local (`chroma/`) ya poblada con tus documentos.
- Tener instaladas las dependencias del proyecto, incluyendo `python-telegram-bot`, `transformers`, `chromadb`, `langchain`, etc.
- Un token de bot de Telegram válido (puedes obtenerlo de @BotFather).

## Configuración
1. Edita la variable `TELEGRAM_TOKEN` en `telegram-bot.py` y pon ahí tu token de bot.
2. Asegúrate de que la base de datos local está creada y contiene tus documentos.

## Uso

```bash
python telegram-bot.py
```

- Inicia el bot y abre un chat con él en Telegram.
- Escribe cualquier pregunta; el bot responderá con el prompt real generado a partir de tus documentos.

## Notas
- El bot **no responde con la respuesta del modelo**, solo con el prompt y contexto que se enviarían.
- Útil para depuración, pruebas de contexto y verificación de recuperación de documentos.
- Si quieres cambiar el número de documentos de contexto, edita el parámetro `num_docs` en la función `handle_message`.