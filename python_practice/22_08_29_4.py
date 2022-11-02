# Q. pandas 버전 확인

# Q. 다음의 list, ndarray, dict를  pandas의 Series 객체로 생성하시오i
# import numpy as np
# mylist = list('abcedfghijklmnopqrstuvwxyz')
# myarr = np.arange(26)
# mydict = dict(zip(mylist, myarr))


# Q.  시리즈의 인덱스를 데이터 프레임의 열로 변환하시오
# 입력> mylist = list('abcedfghijklmnopqrstuvwxyz')
#          myarr = np.arange(26)
#          mydict = dict(zip(mylist, myarr))
#          ser = pd.Series(mydict)


# Q. ser1과 ser2를 결합하여 데이터 프레임을 생성하시오
# import numpy as np
# ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
# ser2 = pd.Series(np.arange(26))

# Q. Series에 '알파벳'이라고 부르는 이름을 지정하시오
# 입력> ser = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))


# Q. Series ser1에서 Series ser1에 있는 항목을 제거하시오
# 입력> ser1 = pd.Series([1, 2, 3, 4, 5])
#           ser2 = pd.Series([4, 5, 6, 7, 8])


# Q. ser1과 ser2에서 공통적이지 않은 모든 항목을 가져오시오
# 입력> ser1 = pd.Series([1, 2, 3, 4, 5])
#           ser2 = pd.Series([4, 5, 6, 7, 8])

# Q. 숫자 계열의 최소값, 25번째 백분위수, 중앙값, 75번째 및 최대값을 출력하시오
# 입력> ser = pd.Series(np.random.normal(10, 5, 25))


# Q. 시리즈의 고유한 항목의 수를 출력하시오
# 입력> ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))


# Q. 가장 빈번한 상위 2 개 값 만 그대로 유지하고 다른 모든 값을 '기타'로 대체핫오
# 입력 > np.random.RandomState(100)
#           ser = pd.Series(np.random.randint(1, 5, [12]))

# Q. 시리즈를 같은 크기의 10개의 동일한 십분위수로 묶고 값을 bin 이름으로 바꾸시오 
# 입력> ser = pd.Series(np.random.random(20))


# Q. 시리즈 ser에서 pos 목록의   위치에서 항목을 추출하시오
# 입력> ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
#          pos = [0, 4, 8, 14, 20] 

# Q.  ser1 및 ser2를 수직 및 수평으로 연결하여 데이터 프레임을 생성하시오
# 입력> ser1 = pd.Series(range(5))
#          ser2 = pd.Series(list('abcde'))

# Q.  ser2항목의 ser1위치를 목록으로 가져오시오
# 입력> ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])
#          ser2 = pd.Series([1, 3, 10, 13])


# Q.  truth항목에서 pred항목의 차의 평균 제곱 오차를 계산하시오
# 입력> truth = pd.Series(range(10))
#         pred = pd.Series(range(10)) + np.random.random(10)


# Q. ser의 각 단어에서 각 단어의 첫 번째 문자를 대문자로 변경합니다.
# 입력> ser = pd.Series(['how', 'to', 'kick', 'ass?'])



# Q. ser의 각 단어에서 각 단어의 문자 수를 계산하시오
# 입력> ser = pd.Series(['how', 'to', 'kick', 'ass?'])


# Q. ser2항목 간의 차이(차분)와  차이의 차이(차분의 차분)를 계산하시오
# 입력> ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35])
# 출력> [nan, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 8.0]
#          [nan, nan, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0]

# Q.  날짜 문자열을 시계열로 변환하시오
# 입력> ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
# 출력> 0   2010-01-01 00:00:00
#          1   2011-02-02 00:00:00
#          2   2012-03-03 00:00:00
#          3   2013-04-04 00:00:00
#          4   2014-05-05 00:00:00
#          5   2015-06-06 12:20:00
#          dtype: datetime64[ns]

# Q.  날짜 문자열에서 요일, 주 번호, 요일 및 요일을 가져오시오
# 입력>  ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
# 출력> Date:  [1, 2, 3, 4, 5, 6]
#          Week number:  [53, 5, 9, 14, 19, 23]
#          Day num of year:  [1, 33, 63, 94, 125, 157]
#          Day of week:  ['Friday', 'Wednesday', 'Saturday', 'Thursday', 'Monday', 'Saturday']


# Q.  연도 - 월 문자열을 매월 4 일에 해당하는 날짜로 변환하시오
# 입력>   ser = pd.Series(['Jan 2010', 'Feb 2011', 'Mar 2012'])
# 출력> 0   2010-01-04
#          1   2011-02-04
#          2   2012-03-04
#          dtype: datetime64[ns]

# Q.   시리즈에서 적어도 2 개의 모음이 포함 된 단어를 필터링하시오
# 입력> ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])
# 출력> 0     Apple
#          1    Orange
#          4     Money
#         dtype: object


# Q. 시리즈에서 유효한 이메일을 필터링하시오
# 입력> emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@t.co', 'narendra@modi.com'])
#         pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'
# 출력> 1    rameses@egypt.com
#          2            matt@t.co
#          3    narendra@modi.com
#          dtype: object


# Q. weights시리즈를 fruit시리즈로 그룹화된 시리즈의 평균을 구하시오
# 입력> fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
#          weights = pd.Series(np.linspace(1, 10, 10))
#          print(weight.tolist())
#          print(fruit.tolist())
# 출력> apple     6.0
#          banana    4.0
#          carrot    5.8
#          dtype: float64 


# Q. 두 시리즈 사이의 유클리드 거리를 계산하시오
# 입력> p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#          q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
# 출력> 18.165

# Q. 숫자 시리즈에서 피크의 위치 (양쪽의 작은 값으로 둘러싸인 값)를 가져오시오
# 입력> ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])
# 출력> array([1, 5, 7])

# Q. 문자열에서 누락 된 공백을 가장 빈번한 문자로 대체하시오
# 입력> my_str = 'dbc deb abed gade'
# 출력> 'dbccdebcabedcgade'  # least frequent is 'c'

# Q. '2000-01-01'부터 시작하여 난수를 값으로 지정한 후 주말 10회(토요일)에 타임시리즈를 만드시오
# 출력> 2000-01-01    4
#          2000-01-08    1
#          2000-01-15    8
#          2000-01-22    4
#          2000-01-29    4
#          2000-02-05    2
#          2000-02-12    4
#          2000-02-19    9
#          2000-02-26    6
#          2000-03-04    6

# Q. 누락 된 모든 날짜가 이전 누락되지 않은 날짜의 값으로 표시되도록 시계열을 채우시오 
# 입력> ser = pd.Series([1,10,3,np.nan], index=pd.to_datetime(['2000-01-01', '2000-01-03', '2000-01-06', '2000-01-08']))
# 출력> 2000-01-01     1.0
#          2000-01-02     1.0
#          2000-01-03    10.0
#          2000-01-04    10.0
#          2000-01-05    10.0
#          2000-01-06     3.0
#          2000-01-07     3.0
#          2000-01-08     NaN

# Q.  첫 10개의 숫자 계열의 자기 상관 관계를 계산하여 가장 큰 상관 관계를 가지고 있는지 알아보시오
# 입력> ser = pd.Series(np.arange(20) + np.random.normal(1, 10, 20))
# 출력> [0.29999999999999999, -0.11, -0.17000000000000001, 0.46000000000000002, 0.28000000000000003, -0.040000000000000001, -0.37, 0.41999999999999998, 0.47999999999999998, 0.17999999999999999]
# Lag having highest correlation:  9

# Q. csv 파일 BostonHousing 데이터 세트의 50번째 행마다 데이터 프레임으로 가져옵니다.
 

# Q. csv 파일 BostonHousing 데이터 세트의  medv(중간 주택 값) 열을 변경하여 25< 값이 '낮음'이 되고 > 25가 '높음'이 되도록 합니다. 
 

# Q. 다음 시리즈를  strides로 데이터 프레임 행으로 생성하시오
# 입력> L = pd.Series(range(15))
# 출력> array([[ 0,  1,  2,  3],
#                   [ 2,  3,  4,  5],
#                   [ 4,  5,  6,  7],
#                   [ 6,  7,  8,  9],
#                   [ 8,  9, 10, 11],
#                   [10, 11, 12, 13]])

# Q. csv 파일 BostonHousing 데이터 세트의  'crim' 및 'medv' 열을 데이터 프레임으로 가져옵니다.
 

# Q. Cars93 데이터 세트의 각 열에 대한 행, 열, 데이터 유형 및 요약 통계의 수를 가져옵니다.  
# 입력>df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
 


# Q. Price가 가장 높은  제조업체, 모델 및 유형  셀의 행과 열 번호를 찾으시오
# 입력>df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
# 출력>

# Q. Type열의 이름을 CarType으로 바꾸고,  열 이름의 '.' 을 '_'로 바꾸시오
# 입력>df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
 

# Q. 데이터 프레임에 누락 된 값이 있는지 확인하시오
# 입력> df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
 

# Q. df의 각 열에서 누락된 값의 수를 계산합니다. 누락된 값의 최대 수가 있는 열은 무엇입니까?
# 입력> df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
 

# Q. Min.Price열과 Max.Price열에서 누락된 값을 해당 열의 평균으로 바꿉니다.
# 입력> df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
 

# Q. df에서 apply메서드를 사용하여 Min.Price 누락된 값을 열의 평균으로 대체하고 Max.Price 누락된 값을 열의 중앙값으로 바꿉니다.
# 입력> df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
 

# Q. 데이터 프레임에서 첫 번째 a 열을 시리즈가 아닌 데이터 프레임으로 가져오시오
# 입력>  df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))
 

# Q. df에서  열 이름을 하드 코딩하지 않고 a열과 c열을 교환하는 일반 함수를 만듭니다.
# 열이름으로 내림차순 정렬하시오
# 입력> df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))
# 출력>

# Q. df의 화면 출력시  디스플레이 설정을 변경하여 최대 10 행과 10 열이 표시되도록 하시오
# 입력> df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
# 출력>

# Q. df로부터  'e-03'과 같은 과학적 표기법을 억제하고 십진수 뒤에 최대 4 개의 숫자를 인쇄하시오
# 입력> df = pd.DataFrame(np.random.random(4)**10, columns=['random'])
# 출력> random
#          0  0.0035
#          1  0.0000
#          2  0.0747
#          3  0.0000


# Q.  df의  'random' 열에 있는 값을 백분율로 형식화하시오
# 입력> df = pd.DataFrame(np.random.random(4), columns=['random'])
# 출력> random
#          0    68.97%
#          1    95.72%
#          2    15.91%
#          3    2.10%


# Q.  df에서 1 번째(행 0)부터 20번째 행마다    'Manufacturer', 'Model', 'Type' 열 데이터를 추출하시오
# 입력> df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
# 출력>  Manufacturer         Model      Type                  Min.Price  Max.Price
# Acura_Integra_Small           Acura      Integra    Small       12.9       18.8
# missing_Legend_Midsize      missing   Legend  Midsize      29.2       38.7
# Audi_90_Compact                Audi           90  Compact      25.9       32.3
# Audi_100_Midsize                Audi      100  Midsize            NaN       44.6
# BMW_535i_Midsize              BMW     535i  Midsize          NaN        NaN

# Q.   df에서 'a' 열의 다섯 번째로 큰 값의 행 위치를 찾으시오
# 입력> df = pd.DataFrame(np.random.randint(1, 30, 30).reshape(10,-1), columns=list('abc'))
 

# Q. ser 에서 평균보다 큰 두 번째로 큰 값의 위치를 찾으시오
# 입력> ser = pd.Series(np.random.randint(1, 100, 15)) 
# 출력> 


# Q. df에서 행 합이 100보다 큰 마지막 두 행을 가져오시오.
# 입력> df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
# 출력> 


# Q. ser  의 하위 5%ile 및 95%ile 초과 모든 값을 각각의 5번째 및 95번째 %ile 값으로 바꿉니다 .
# 입력> 
# 출력> 


# Q. ser = pd.Series(np.logspace(-2, 2, 30))
# 입력> 
# 출력>


# Q.
# 입력> 
# 출력>


# Q.
# 입력> 
# 출력>