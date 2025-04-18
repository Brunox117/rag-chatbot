from flask import Flask, request, jsonify
from populate_database import main as populate_db
from query_data import query_rag
import threading

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/populate', methods=['POST'])
def populate_database():
    try:
        # Obtener parámetros del request
        data = request.get_json()
        reset = data.get('reset', False)
        use_local_db = data.get('use_local_db', False)
        
        # Ejecutar la población de la base de datos en un thread separado
        # para no bloquear la respuesta de la API
        def run_populate():
            import sys
            sys.argv = ['populate_database.py']
            if reset:
                sys.argv.append('--reset')
            if use_local_db:
                sys.argv.append('--use-local-db')
            populate_db()
        
        thread = threading.Thread(target=run_populate)
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Database population started'
        }), 202
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        
        # Extraer parámetros del request con valores por defecto
        query_text = data.get('query_text')
        if not query_text:
            return jsonify({
                'status': 'error',
                'message': 'query_text is required'
            }), 400
            
        model_size = data.get('model_size', 'large')
        num_docs = data.get('num_docs', 2)
        use_local = data.get('use_local', False)
        max_tokens = data.get('max_tokens', 2048)
        use_local_db = data.get('use_local_db', False)
        
        # Llamar a la función de consulta
        response = query_rag(
            query_text=query_text,
            model_size=model_size,
            num_docs=num_docs,
            use_local=use_local,
            max_tokens=max_tokens,
            use_local_db=use_local_db
        )
        
        return jsonify({
            'status': 'success',
            'response': response
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
