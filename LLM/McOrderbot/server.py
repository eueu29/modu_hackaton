from flask import Flask, request, jsonify, send_from_directory
from model import OrderModule, gpt4o
from flask_cors import CORS
import json

app = Flask(__name__, static_folder='static')
CORS(app)
order_module = OrderModule(gpt4o)

@app.route('/')
def serve_html():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/cart', methods=['GET'])
def cart_view():
    try:
        # cart_message = order_module.cart.get_order_message() if order_module.cart.cart else ""
        cart_message = order_module.cart.get_order_message()
        return jsonify({
            "cart": cart_message
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({"error": "메시지가 없습니다"}), 400
            
        response = order_module.handle_prompt(user_message)
        parsed_response = json.loads(response)
        message = parsed_response["message"]
        print(f"message:{message}")
        order_finish = parsed_response["order_finish"]
        print(f"order_finish:{order_finish}")
        order = parsed_response["order"]
        print(f"order:{order}")
        cart_summary = parsed_response["cart_summary"]
        print(f"cart_summary:{cart_summary}")

        return jsonify({
            "message": message,
            "order_finish": order_finish,
            "order": order,
            "cart_summary": cart_summary
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
