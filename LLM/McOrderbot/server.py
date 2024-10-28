from flask import Flask, request, jsonify, send_from_directory
from model import OrderModule, gpt4o
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 모든 라우트에 CORS 허용
order_module = OrderModule(gpt4o)

# @app.route('/')
# def serve_html():
#     return send_from_directory('static', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({"error": "메시지가 없습니다"}), 400
            
        response = order_module.handle_prompt(user_message)

        return jsonify({"response": response})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
