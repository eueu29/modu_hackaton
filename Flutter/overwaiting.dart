import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart'; // 음성 인식 추가
import 'package:flutter_tts/flutter_tts.dart'; // TTS 추가
import 'llm_chat.dart'; // LLMChatScreen을 사용하기 위한 import
import 'dart:async'; // 타이머 사용

class OrderWaitingScreen extends StatefulWidget {
  @override
  _OrderWaitingScreenState createState() => _OrderWaitingScreenState();
}

class _OrderWaitingScreenState extends State<OrderWaitingScreen> {
  SpeechToText _speechToText = SpeechToText();
  FlutterTts flutterTts = FlutterTts(); // TTS 객체 생성
  bool _isListening = false;
  bool _speechEnabled = false; // 음성 인식 활성화 상태 확인
  Timer? _timer; // 타이머 추가
  final GlobalKey<LLMChatScreenState> _chatKey = GlobalKey<LLMChatScreenState>(); // GlobalKey로 상태 제어

  @override
  void initState() {
    super.initState();
    _initializeSpeech();
    _initializeTTS(); // TTS 초기화
  }

  Future<void> _initializeSpeech() async {
    _speechEnabled = await _speechToText.initialize();
    if (!_speechEnabled) {
      print("음성 인식 초기화 실패");
    } else {
      print("음성 인식 초기화 성공");
    }
  }

  Future<void> _initializeTTS() async {
    // TTS 초기화
    await flutterTts.setLanguage("ko-KR"); // 한국어 설정 (영어는 en-US)
    await flutterTts.setPitch(1.2); // 음성 높낮이 설정
    await flutterTts.setSpeechRate(1.0); // 말하는 속도 설정
  }

  // TTS로 텍스트 변환 및 음성 재생 (서버 응답에만 사용)
  Future<void> _speak(String text) async {
    await flutterTts.speak(text);
  }

  void _startListening() async {
    if (_speechEnabled) {
      await _speechToText.listen(onResult: (result) {
        print("인식된 단어: ${result.recognizedWords}, 최종 결과: ${result.finalResult}");

        if (result.finalResult) {
          setState(() {
            if (result.hasConfidenceRating && result.confidence > 0.5) {
              _chatKey.currentState?.receiveMessageFromSTT(result.recognizedWords);
              // TTS로 음성 인식 결과를 읽지 않고, 서버 응답만 읽도록 변경
            }
          });
        }
      });

      setState(() {
        _isListening = true;
      });

      _timer = Timer(Duration(seconds: 3), () {
        _stopListening();
        print("타이머 만료로 음성 인식 종료");
      });
    } else {
      print("음성 인식이 활성화되지 않음");
    }
  }

  void _stopListening() async {
    await _speechToText.stop();
    _timer?.cancel(); // 타이머 취소
    setState(() {
      _isListening = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('주문 대기 화면'),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: <Widget>[
              // LLMChatScreen을 GlobalKey로 감싸서 상태에 접근 가능하게 함
              Container(
                height: 200,
                child: LLMChatScreen(
                  key: _chatKey, // GlobalKey 설정
                  onMessageSent: (message, {bool isUserMessage = false}) {
                    // 사용자가 입력한 메시지인 경우 TTS로 변환하지 않음
                    if (!isUserMessage) {
                      print("서버로부터 응답 메시지: $message");
                      _speak(message); // 서버의 응답을 TTS로 출력
                    }
                  },
                ),
              ),
              SizedBox(height: 20),

              // 장바구니 메시지와 버튼을 나란히 배치
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  // 장바구니 메시지
                  Flexible(
                    flex: 2,
                    child: Container(
                      height: 200,
                      padding: EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: Color(0xFFB22626),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Center(
                        child: Text(
                          '비어있습니다.',
                          style: TextStyle(color: Colors.white, fontSize: 24),
                        ),
                      ),
                    ),
                  ),
                  SizedBox(width: 20),

                  // 버튼 섹션
                  Flexible(
                    flex: 1,
                    child: Container(
                      height: 200,
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: <Widget>[
                          // 장바구니 & 결제 버튼
                          Expanded(
                            child: ElevatedButton(
                              onPressed: () {
                                print("장바구니 & 결제 버튼 클릭됨");
                              },
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.blueAccent,
                                padding: EdgeInsets.symmetric(vertical: 15),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(30),
                                ),
                                elevation: 5,
                              ),
                              child: Text(
                                '장바구니',
                                style: TextStyle(fontSize: 16, color: Colors.white, fontWeight: FontWeight.bold),
                              ),
                            ),
                          ),
                          SizedBox(width: 10),

                          // 전체 메뉴 보기 버튼
                          Expanded(
                            child: ElevatedButton(
                              onPressed: () {
                                print("전체 메뉴 보기 버튼 클릭됨");
                              },
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.blueAccent,
                                padding: EdgeInsets.symmetric(vertical: 15),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(30),
                                ),
                                elevation: 5,
                              ),
                              child: Text(
                                '전체 메뉴 보기',
                                style: TextStyle(fontSize: 16, color: Colors.white, fontWeight: FontWeight.bold),
                              ),
                            ),
                          ),
                          SizedBox(width: 10),

                          // 말하기 버튼
                          Expanded(
                            child: ElevatedButton(
                              onPressed: () {
                                if (_isListening) {
                                  _stopListening();
                                } else {
                                  _startListening();
                                }
                              },
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.blueAccent,
                                padding: EdgeInsets.symmetric(vertical: 15),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(30),
                                ),
                                elevation: 5,
                              ),
                              child: Text(
                                _isListening ? '정지' : '말하기',
                                style: TextStyle(fontSize: 16, color: Colors.white, fontWeight: FontWeight.bold),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
