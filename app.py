from flask import Flask, render_template, request, jsonify
from inference import transliterate_with_dict 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transliterate', methods=['POST'])
def transliterate():
    data = request.json
    word = data.get('word', '')
    if not word.strip():
        return jsonify({'candidates': []})
    
    candidates = transliterate_with_dict(eng_input=word, n=5)  # get top 5 suggestions
    return jsonify({'candidates': candidates})

if __name__ == '__main__':
    app.run(debug=True)
