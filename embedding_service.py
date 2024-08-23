from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    # Get the JSON data from the request
    data = request.json
    sentence = data.get('sentence', '')
    
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400
    
    # Generate the embedding
    embedding = model.encode(sentence).tolist()  # Convert numpy array to list
    
    
    return jsonify({'embedding': embedding})

if __name__ == '__main__':

    app.run(port=5000)  # Run the Flask app on port 5000
