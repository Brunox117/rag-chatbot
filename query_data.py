import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import requests
import json
import chromadb
from chromadb.config import Settings
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# Definimos los modelos disponibles con sus límites de tokens
AVAILABLE_MODELS = {
    "small": {
        "name": "google/flan-t5-small",
        "max_tokens": 512,
    },
    "base": {
        "name": "google/flan-t5-base",
        "max_tokens": 768,
    },
    "large": {
        "name": "google/flan-t5-large",
        "max_tokens": 1024,
    },
    "xl": {
        "name": "google/flan-t5-xl",
        "max_tokens": 2048,
    },
}

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--model", type=str, choices=["small", "base", "large", "xl", "xxl"],
                        default="large", help="Modelo a utilizar (small, base, large, xl, xxl).")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Número máximo de tokens a utilizar (solo aplica cuando se usa la API externa).")
    parser.add_argument("--docs", type=int, default=3,
                        help="Número de documentos a utilizar como contexto (1-5).")
    parser.add_argument("--use-local", action="store_true",
                        help="Usar modelo local FLAN-T5 en lugar de API externa.")
    parser.add_argument("--use-local-db", action="store_true",
                        help="Usa la base de datos del proyecto, no la de docker.")
    args = parser.parse_args()

    query_rag(args.query_text, args.model, args.docs,
              args.use_local, args.max_tokens, args.use_local_db)


def generate_local_response(prompt: str, model, tokenizer):
    """Genera una respuesta usando el modelo local FLAN-T5."""
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    response = pipe(prompt)[0]["generated_text"]

    # Calculamos tokens para mantener consistencia con la API externa
    input_tokens = len(tokenizer.encode(prompt))
    output_tokens = len(tokenizer.encode(response))

    return {
        "response": response,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "cost": 0  # Los modelos locales no tienen costo por uso
    }


def query_rag(query_text: str, model_size: str = "large", num_docs: int = 2,
              use_local: bool = False, max_tokens: int = 2048, use_local_db: bool = False):
    # Prepare the DB.
    print(f"\nConsultando: '{query_text}'")
    print("Cargando embedding function y base de datos en caso de no usar el default all-MiniLM-L12-v2 se debe especificar el modelo de embedding en la lína 92...")
    embedding_function = get_embedding_function()

    # Modificar la conexión según el parámetro use_local_db
    if use_local_db:
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )
    else:
        chroma_client = chromadb.HttpClient(
            host="localhost",
            port=8000
        )

    try:
        # Obtener la colección
        collection = chroma_client.get_collection(name="my_collection")

        # Inicializar Chroma de LangChain con el cliente HTTP
        db = Chroma(
            client=chroma_client,
            collection_name="my_collection",
            embedding_function=embedding_function
        )

        # Search the DB.
        print("Buscando documentos similares...")
        # Usar el número de documentos especificado
        results = db.similarity_search_with_score(query_text, k=num_docs)

        print(f"\nSe encontraron {len(results)} documentos relevantes:")
        for i, (doc, score) in enumerate(results):
            print(
                f"\n[{i+1}] Documento: {doc.metadata.get('id', 'Unknown')} (Similaridad: {1-score:.4f})")
            print(f"    {doc.page_content[:150]}...")

        # Obtener información del modelo seleccionado
        model_info = AVAILABLE_MODELS[model_size]
        model_name = model_info["name"]
        # Obtener el límite de tokens según si es local o API
        if use_local:
            token_limit = model_info["max_tokens"]
        else:
            token_limit = max_tokens
        print(f"\nUsando modelo: {model_name}")

        if use_local:
            print(f"Límite de tokens del modelo local: {token_limit}")
        else:
            print(f"Límite de tokens configurado para API: {token_limit}")

        # Inicializar el modelo y tokenizer antes para poder verificar el tamaño
        print("Cargando modelo de lenguaje...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"USE-LOCAL: {use_local}")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name) if use_local else None

        # Asegurarnos de que el contexto no es demasiado largo
        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc, _score in results])
        question_tokens = len(tokenizer.encode(query_text))
        context_tokens = len(tokenizer.encode(context_text))
        template_tokens = len(tokenizer.encode(
            PROMPT_TEMPLATE.replace("{context}", "").replace("{question}", "")))
        total_tokens = context_tokens + question_tokens + template_tokens

        print(
            f"\nEstimación de tokens: {total_tokens} (máximo permitido: {max_tokens})")
        print(f"- Contexto: {context_tokens} tokens")
        print(f"- Pregunta: {question_tokens} tokens")
        print(f"- Plantilla: {template_tokens} tokens")

        # Dejamos un margen de seguridad del 3%
        if total_tokens > max_tokens * 0.97:
            print("\n⚠️ El contexto es demasiado grande, truncando...")
            # Estrategia simple: usar menos documentos
            max_docs = max(1, num_docs - 1)
            context_text = "\n\n---\n\n".join(
                [doc.page_content for doc, _score in results[:max_docs]])
            context_tokens = len(tokenizer.encode(context_text))
            total_tokens = context_tokens + question_tokens + template_tokens
            print(f"Nuevos tokens totales después de truncar: {total_tokens}")

            # Si aún es demasiado grande, truncamos aún más
            if total_tokens > max_tokens * 0.97 and max_docs > 1:
                max_docs = 1
                context_text = "\n\n---\n\n".join(
                    [doc.page_content for doc, _score in results[:max_docs]])
                context_tokens = len(tokenizer.encode(context_text))
                total_tokens = context_tokens + question_tokens + template_tokens
                print(
                    f"Nuevos tokens totales después de truncar más: {total_tokens}")

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=context_text, question=query_text)
        print("\n" + "="*50 + " PROMPT " + "="*50)
        print(prompt)
        print("="*109 + "\n")

        if use_local:
            print("\nUsando modelo local FLAN-T5...")
            response_text = generate_local_response(prompt, model, tokenizer)
        else:
            print("\nGenerando respuesta desde API externa...")
            response_text = call_external_api(prompt, token_limit)

        sources = [doc.metadata.get('id', 'Unknown')
                   for doc, _score in results]
        formatted_response = f"\nRespuesta: {response_text['response']}\n\nFuentes consultadas: {sources}\nTokens de entrada: {response_text['inputTokens']}\nTokens de salida: {response_text['outputTokens']}\nCosto: {response_text['cost']}"
        print(formatted_response)
        return response_text['response']

    except Exception as e:
        print(f"Error al acceder a la colección: {str(e)}")
        return None


def call_external_api(prompt):
    # Definir la URL de la API
    url = "https://labs-ai-proxy.acloud.guru/rest/openai/chatgpt-35/v1/chat/completions"

    # API Key que el usuario puede cambiar
    API_KEY = "44a33da0-84b3-4e21-848e-93fc1b9bdeb7"

    # Configurar headers
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Preparar el payload
    payload = {
        "prompt": prompt,
    }

    # Hacer la solicitud POST
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Lanzar excepción si hay error HTTP

        # Procesar la respuesta
        response_data = response.json()
        return response_data

    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la solicitud a la API: {e}")
        # Devolver un diccionario con valores predeterminados en caso de error
        return {
            "response": f"Error en la solicitud a la API: {str(e)}",
            "inputTokens": 0,
            "outputTokens": 0,
            "cost": 0
        }


if __name__ == "__main__":
    main()
