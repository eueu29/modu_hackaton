import 'package:flutter/material.dart';
import 'package:final_proj_flutter/util/colors.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
// import 'util/theme.dart';

void main() {
  runApp(const ModuApp());
}

class ModuApp extends StatelessWidget {
  const ModuApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ScreenUtilInit(
      designSize: const Size(360, 690),
      minTextAdapt: true,
      builder: (context, child) {
        return MaterialApp(
          theme: ThemeData(
            primaryColor: moduRed,
            textTheme: appTextTheme(),
          ),
          debugShowCheckedModeBanner: false,
          home: const MenuPage(),
        );
      },
    );
  }
}

class MenuPage extends StatelessWidget {
  const MenuPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          Flexible(
            flex: 2,
            child: Container(
              color: Colors.transparent,
              width: double.infinity,
              child: const Expanded(
                child: LLMresponse(
                  text:
                      'LLM 응답을 받을 위치. 예시) 주문하고 싶은 메뉴가 정해져 있다면 메뉴명을 말씀해주세요. 메뉴 검색을 원하실 경우, 최대한 구체적인 취향을 알려주세요! 치킨이 들어간 햄버거는 6가지입니다. 맥치킨, 맥치킨 모짜렐라, 맥크리스피 클래식 버거, 맥크리스피 디럭스 버거, 맥스리스피 스리라차 마요, 맥스파이시 상하이버거가 있습니다.',
                ),
              ),
            ),
          ),
          const Expanded(
            flex: 7,
            child: Stack(
              children: [
                ResponsiveMenu(),
                Positioned(
                  left: 0,
                  bottom: 0,
                  child: ButtonBox(),
                ),
              ],
            ),
          ),
          Flexible(
            flex: 3,
            child: Container(
              width: double.infinity,
              decoration: const BoxDecoration(
                color: moduRed,
                borderRadius: BorderRadius.only(
                  topRight: Radius.circular(60),
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black26,
                    blurRadius: 20,
                    offset: Offset(20, -4),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class BasicButton extends StatelessWidget {
  final String? label;
  final TextStyle? textStyle;

  const BasicButton({super.key, this.label, this.textStyle});

  @override
  Widget build(BuildContext context) {
    double buttonSize =
        ScreenUtil().orientation == Orientation.portrait ? 70.h : 40.w;

    return Container(
      width: buttonSize,
      height: buttonSize,
      decoration: BoxDecoration(
        color: moduButton,
        borderRadius: BorderRadius.circular(12),
        boxShadow: const [
          BoxShadow(
            color: Colors.black26,
            blurRadius: 10,
            offset: Offset(-1, 3),
          ),
        ],
      ),
      child: Center(
        child: Text(
          label ?? 'Button',
          style: textStyle ?? const TextStyle(color: moduPencilBlack),
        ),
      ),
    );
  }
}

class ButtonBox extends StatelessWidget {
  const ButtonBox({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: const BoxDecoration(
        color: moduButtonSetOnGrey,
        borderRadius: BorderRadius.only(
          topRight: Radius.circular(16),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black26,
            blurRadius: 10,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          BasicButton(
              label: '말하기',
              textStyle: Theme.of(context).textTheme.headlineLarge),
          const SizedBox(width: 24),
          BasicButton(
              label: '메뉴 다 보기',
              textStyle: Theme.of(context).textTheme.headlineLarge),
          const SizedBox(width: 24),
          BasicButton(
              label: '장바구니 결제',
              textStyle: Theme.of(context).textTheme.headlineLarge),
        ],
      ),
    );
  }
}

class LLMresponse extends StatelessWidget {
  final String text;

  const LLMresponse({super.key, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 300,
      height: 200,
      padding: const EdgeInsets.symmetric(horizontal: 30),
      child: ClipRRect(
        child: SingleChildScrollView(
          child: Text(
            text,
            style: Theme.of(context).textTheme.headlineLarge,
          ),
        ),
      ),
    );
  }
}

class ResponsiveMenu extends StatelessWidget {
  const ResponsiveMenu({super.key});

  @override
  Widget build(BuildContext context) {
    ScrollController scrollController = ScrollController();

    return Column(
      children: [
        const SizedBox(
          height: 70,
          child: TopMenuScroll(),
        ),
        Expanded(
          child: Scrollbar(
            controller: scrollController,
            thumbVisibility: true,
            child: OrientationBuilder(
              builder: (context, orientation) {
                return GridView.builder(
                  controller: scrollController,
                  padding: const EdgeInsets.all(16),
                  gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: orientation == Orientation.portrait ? 2 : 3,
                    crossAxisSpacing: 16,
                    mainAxisSpacing: 16,
                    childAspectRatio: 0.8,
                  ),
                  itemCount: 6,
                  itemBuilder: (context, index) {
                    return const MenuItemCard();
                  },
                );
              },
            ),
          ),
        ),
      ],
    );
  }
}

class MenuItemCard extends StatelessWidget {
  const MenuItemCard({super.key});

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 1,
      child: Container(
        decoration: BoxDecoration(
          color: Colors.grey[300],
          borderRadius: BorderRadius.circular(12),
          boxShadow: const [
            BoxShadow(
              color: Colors.black26,
              blurRadius: 4,
              offset: Offset(2, 4),
            ),
          ],
        ),
        child: const Center(
          child: Text(
            '메뉴',
            style: TextStyle(fontSize: 16),
          ),
        ),
      ),
    );
  }
}

class TopMenuButton extends StatelessWidget {
  final String label;

  const TopMenuButton({
    super.key,
    required this.label,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.symmetric(horizontal: 8.w),
      child: Container(
        padding: EdgeInsets.symmetric(horizontal: 20.w, vertical: 10.h),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          boxShadow: const [
            BoxShadow(
              color: Colors.black12,
              blurRadius: 4,
              offset: Offset(2, 2),
            ),
          ],
        ),
        child: Text(
          label,
          style: TextStyle(
            color: moduRed,
            fontSize: 16.sp,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }
}

class TopMenuScroll extends StatelessWidget {
  const TopMenuScroll({super.key});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 70.h,
      child: const SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(
          children: [
            TopMenuButton(label: '추천 메뉴'),
            TopMenuButton(label: '행사 메뉴'),
            TopMenuButton(label: '버거 세트'),
            TopMenuButton(label: '버거'),
            TopMenuButton(label: '사이드'),
            TopMenuButton(label: '음료'),
            TopMenuButton(label: '디저트'),
          ],
        ),
      ),
    );
  }
}
