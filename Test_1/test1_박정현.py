# 상품 정보를 저장할 Product 클래스를 정의한다.
# 맴버 변수 : prdNo, prdName, prdPrice, prdYear, prdMaker
# 변수 타입은 데이터 용도에 맞게 지정
# 매개변수가 있는 생성자 : 객체 생성 시 값을 전달 받아서 멤버 변수 값 초기화
# 데이터 출력 : __str__ 사용
# main 모듈에서 ProductTest 클래스로부터 객체 생성


class Product:
    def __init__(self, prdNo, prdName, prdPrice, prdYear, prdMaker):
        self.prdNo = prdNo
        self.prdName = prdName
        self.prdPrice = prdPrice
        self.prdYear = prdYear
        self.prdMaker = prdMaker

    def __str__(self):
        return "{}    {}    {}    {}    {}".format(
            self.prdNo, self.prdName, self.prdPrice, self.prdYear, self.prdMaker
        )


class ProductTest:
    def __init__(self):
        print("상품번호     상품명     가격     연도     상품제조사")
        print("--------------------------------------------------------------")

    def printProduct(self, prd):
        print(prd)


if __name__ == "__main__":
    pt = ProductTest()
    pt.printProduct(Product(100, "노트북", 1200000, 2021, "삼성"))
    pt.printProduct(Product(200, "모니터", 300000, 2021, "LG"))
    pt.printProduct(Product(400, "마우스", 30000, 2020, "로지텍"))
