import 'package:flutter/material.dart';
import 'orderwaiting.dart'; // OrderWaitingScreen을 사용하기 위한 import

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '주문 앱',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MainScreen(), // 시작 화면 설정
    );
  }
}

// 메인 화면
class MainScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('모두의 점원'),
      ),
      body: Center(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            // 터치해서 주문하기 버튼
            GestureDetector(
              onTap: () {
                // 터치해서 주문하기 기능 추가
                print("터치해서 주문하기 버튼 클릭됨");
              },
              child: Container(
                width: 300,
                height: 150,
                margin: EdgeInsets.symmetric(vertical: 10),
                decoration: BoxDecoration(
                  color: Colors.red,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Center(
                  child: Text(
                    '터치해서 주문하기!',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 25,
                    ),
                  ),
                ),
              ),
            ),
            // 말로 주문하기 버튼
            SizedBox(width: 30),
            GestureDetector(
              onTap: () {
                // 주문 대기 화면으로 이동
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => OrderWaitingScreen()),
                );
              },
              child: Container(
                width: 300,
                height: 150,
                margin: EdgeInsets.symmetric(vertical: 10),
                decoration: BoxDecoration(
                  color: Colors.red,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Center(
                  child: Text(
                    '말로 주문하기!',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 25,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
