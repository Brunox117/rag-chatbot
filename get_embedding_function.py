from langchain_huggingface import HuggingFaceEmbeddings
import sys

MODEL_CONFIG = {
    # BUENO Y LIGERO
    "minil12": {
        "name": "all-MiniLM-L12-v2",
        "kwargs": {}
    },
    # MUCHOS IDIOMAS Y EL MÁS POTENTE
    "multilingual": {
        "name": "BAAI/bge-m3",
        "kwargs": {
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {
                "normalize_embeddings": True,
                "batch_size": 16
            }
        }
    },
    #SE LE PUEDEN DAR INSTRUCCIONES, PESADO
    "instruction": {
        "name": "hkunlp/instructor-large",
        "kwargs": {
            "instruction": "Represent the Spanish document for retrieval: ",
            "model_kwargs": {"device": "cpu"}
        }
    }
}


def get_embedding_function(model_name="minil12"):
    try:
        # Obtener configuración del modelo
        config = MODEL_CONFIG.get(model_name)

        if not config:
            # Si no está en la lista predefinida, usar el nombre directamente
            print(f"Usando modelo personalizado: {model_name}")
            return HuggingFaceEmbeddings(model_name=model_name)

        print(f"Usando modelo: {config['name']} - Tipo: {model_name.upper()}")

        # Cargar modelo con configuración específica
        return HuggingFaceEmbeddings(
            model_name=config["name"],
            **config["kwargs"]
        )

    except Exception as e:
        print(f"\n❌ Error al cargar el modelo {model_name}: {str(e)}")
        print("\nPosibles soluciones:")
        print("- Ejecuta: pip install sentence-transformers")
        print("- Verifica el nombre del modelo en HuggingFace")
        print(f"- Modelos disponibles: {', '.join(MODEL_CONFIG.keys())}")
        sys.exit(1)
