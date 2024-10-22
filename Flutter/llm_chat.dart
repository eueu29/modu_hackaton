import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_tts/flutter_tts.dart'; // TTS 라이브러리 추가

class LLMChatScreen extends StatefulWidget {
  final Function(String) onMessageSent;

  const LLMChatScreen({Key? key, required this.onMessageSent}) : super(key: key);

  @override
  LLMChatScreenState createState() => LLMChatScreenState();
}

class LLMChatScreenState extends State<LLMChatScreen> {
  final TextEditingController _controller = TextEditingController();
  final List<Map<String, String>> _messages = [];
  final String serverUrl = 'http://127.0.0.1:5000/api/chat'; // Flask 서버 URL
  FlutterTts flutterTts = FlutterTts(); // TTS 객체 생성

  @override
  void initState() {
    super.initState();
    _initializeTTS(); // TTS 초기화
    _messages.add({
      "sender": "system",
      "message": "안녕하세요, 무엇을 도와드릴까요?"
    });
  }

  Future<void> _initializeTTS() async {
    await flutterTts.setLanguage("ko-KR"); // 한국어 설정 (영어는 en-US)
    await flutterTts.setPitch(1.0); // 음성 높낮이 설정
    await flutterTts.setSpeechRate(1.0); // 말하는 속도 설정
  }

  // TTS로 텍스트 변환 및 음성 재생 (서버 응답에만 사용)
  Future<void> _speak(String text) async {
    await flutterTts.speak(text);
  }

  Future<void> _getLLMResponse(String message) async {
    final url = Uri.parse(serverUrl);
    final headers = {'Content-Type': 'application/json'};
    final body = jsonEncode({'message': message});

    try {
      final response = await http.post(url, headers: headers, body: body);
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final responseMessage = data['response']['메시지']; // '메시지' 필드 확인

        setState(() {
          _messages.add({
            "sender": "LLM", // 서버에서 온 응답임을 명시
            "message": responseMessage
          });
        });

        // 오직 서버에서 온 메시지만 TTS로 변환
        _speak(responseMessage);
      } else {
        setState(() {
          _messages.add({
            "sender": "LLM",
            "message": "서버에서 응답을 가져오는 중 오류 발생. 상태 코드: ${response.statusCode}"
          });
        });
      }
    } catch (e) {
      setState(() {
        _messages.add({
          "sender": "LLM",
          "message": "서버에 연결 실패: $e"
        });
      });
    }
  }

  // 사용자의 입력 메시지 처리 (TTS 호출 없음)
  void _sendMessage() {
    if (_controller.text.isEmpty) return;

    setState(() {
      _messages.add({"sender": "User", "message": _controller.text});
    });

    widget.onMessageSent(_controller.text); // 부모 위젯의 콜백 함수 호출
    _getLLMResponse(_controller.text); // LLM에 메시지 전송 (서버 응답만 TTS로 변환됨)
    _controller.clear(); // 입력란 초기화
  }

  // 음성 인식 결과를 처리하는 메서드 추가 (LLM 응답에만 TTS 사용)
  void receiveMessageFromSTT(String message) {
    setState(() {
      _messages.add({
        "sender": "User",
        "message": message
      });
    });
    _getLLMResponse(message); // 음성 인식 결과를 LLM에 전송
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: ListView.builder(
            itemCount: _messages.length,
            itemBuilder: (context, index) {
              final isUserMessage = _messages[index]["sender"] == "User";
              return Align(
                alignment: isUserMessage
                    ? Alignment.centerRight
                    : Alignment.centerLeft,
                child: Container(
                  margin: EdgeInsets.symmetric(vertical: 5, horizontal: 10),
                  padding: EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: isUserMessage ? Colors.blue[100] : Colors.grey[300],
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    _messages[index]["message"] ?? "",
                    style: TextStyle(
                      color: isUserMessage ? Colors.black : Colors.black87,
                    ),
                  ),
                ),
              );
            },
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _controller,
                  decoration: InputDecoration(
                    hintText: "메시지를 입력하세요...",
                    border: OutlineInputBorder(),
                  ),
                ),
              ),
              IconButton(
                icon: Icon(Icons.send),
                onPressed: _sendMessage, // 사용자의 입력은 TTS 없이 처리됨
              ),
            ],
          ),
        ),
      ],
    );
  }
}
