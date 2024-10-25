from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from Flutter.llm_st import OrderModule
import os
from dotenv import load_dotenv

load_dotenv()

# gpt4o 모델을 전역 변수로 선언
gpt_model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.3)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 모든 도메인에 대해 요청 허용

# OrderModule 인스턴스 생성
order_module = OrderModule(gpt_model)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')

    # 클라이언트가 보낸 메시지를 서버 콘솔에 출력
    print(f"Received message from client: {user_message}")

    try:
        # 주문 모듈에서 사용자 메시지 처리
        order_response = order_module.execute_order(user_message)  # LLM의 응답을 받아옵니다.

        # 응답 데이터 구성
        response_data = {
            "response": order_response  # LLM의 응답을 클라이언트에게 전달합니다.
        }

        # 응답 데이터를 콘솔에 출력 (디버깅용)
        print(f"Response data to be sent: {response_data}")

        # Flask에서 응답을 JSON 형식으로 반환
        return jsonify(response_data)

    except Exception as e:
        # 오류가 발생한 경우 서버 콘솔에 오류 출력 및 오류 응답 반환
        print(f"Error occurred: {e}")
        return jsonify({"response": "An error occurred while processing your request.", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
