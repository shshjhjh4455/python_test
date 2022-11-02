# Q. numpy를 np로 가져오고 버전 확인
import numpy as np

# Q. 0에서 9까지의 숫자로 구성된 1D 배열 생성
a = np.arange(10)

# Q. 모든 True의 3×3 numpy 배열 생성
b = np.ones((3,3))

# Q. 1D 배열에서 모든 홀수를 추출
# 입력 > arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# 출력 > array([1, 3, 5, 7, 9])
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr[arr%2==1])

# Q. 모든 홀수 arr를 -1 로 바꿉니다.
# 입력 > arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# 출력 > array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr[arr%2==1] = -1
print(arr)

# Q. 1D 배열을 2행이 있는 2D 배열로 변환
# 입력 > np.arange(10)
# 출력 > array([[0, 1, 2, 3, 4],
#                   [5, 6, 7, 8, 9]])
arr = np.arange(10)
print(arr.reshape(2,5))

# Q. 두 개의 어레이를 수직으로 쌓기
# 입력 > a = np.arange(10).reshape(2,-1)
#           b = np.repeat(1, 10).reshape(2,-1)
# 출력 > array([[0, 1, 2, 3, 4, 1, 2, 3, 4, 5],
#                   [5, 6, 7, 8, 9, 6, 7, 8, 9, 10]])
a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
#?




# 출력 >  array([[0, 1, 2, 3, 4],
#                    [5, 6, 7, 8, 9],
# 	     [1, 1, 1, 1, 1],
# 	     [1, 1, 1, 1, 1]])

a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
print(np.vstack((a,b)))

# Q. 두 개의 어레이를 수평으로 쌓기
# 입력 > a = np.arange(10).reshape(2,-1)
#           b = np.repeat(1, 10).reshape(2,-1)

# 출력 >  array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
#                     [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
print(np.hstack((a,b)))

# Q. 하드코딩 없이 아래 입력 배열만 사용 출력과 같은 패턴을 만드시오
# 입력 > a = np.array([1,2,3])
# 출력 > array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
a = np.array([1,2,3])
print(np.tile(a,3))

# Q 두 파이썬 numpy 배열 사이의 공통 항목  출력
# 입력 > a = np.array([1,2,3,2,3,4,3,4,5,6])
#          b = np.array([7,2,10,2,7,4,9,4,9,8])
# 출력 >  array([2, 4])
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a,b))


# Q. 배열 a에서 b 배열에 있는 모든 항목을 제거합니다.
# 입력 > a = np.array([1,2,3,4,5])
#           b = np.array([5,6,7,8,9])
# 출력 > array([1,2,3,4])
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
print(np.setdiff1d(a,b))

# Q. a, b두 배열의 요소가 일치하는 위치를 출력
# 입력 > a = np.array([1,2,3,2,3,4,3,4,5,6])
#           b = np.array([7,2,10,2,7,4,9,4,9,8])
# 출력 > (array([1, 3, 5, 7]),)
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.where(a==b))


# Q. a배열에서  5에서 10 사이의 모든 항목을 에서 가져옵니다 
# 입력 > a = np.array([2, 6, 1, 9, 10, 3, 27])
# 출력 > ( array ([ 6 , 9 , 10 ]),) 
a = np.array([2, 6, 1, 9, 10, 3, 27])
print(np.where(a>5, a, 0))

# Q. maxx두 개의 스칼라에서 작동하는 함수를 두 개의 배열에서 작동하도록 변환하시오
# 입력 > def maxx(x, y):
#     """Get the maximum of two items"""
#     if x >= y:
#         return x
#     else:
#         return y

# maxx(1, 5)  #호출
# 5             #출력


# 출력 > a = np.array([5, 7, 9, 8, 6, 4, 5])
#           b = np.array([6, 3, 4, 8, 9, 7, 1])
#           pair_max(a, b)     #함수 호출
#           array([ 6.,  7.,  9.,  8.,  9.,  7.,  5.])  #출력
def maxx(x, y):
    """Get the maximum of two items"""
    if x >= y:
        return x
    else:
        return y

def pair_max(a, b):
    return np.vectorize(maxx)(a, b)

a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
print(pair_max(a, b))



# Q.  2d numpy 배열에서 배열의 1열과 2열을 바꿉니다
# 입력 > arr = np.arange(9).reshape(3,3)
# 출력 > array([[0, 3, 6],
#                 [1, 4, 7],
#                 [2, 5, 8]])
arr = np.arange(9).reshape(3,3)
print(arr)
np.swapaxes(arr, 1, 0)
print(arr)


# Q.  2d numpy 배열에서 배열의 1행과 2행을  바꿉니다
# 입력 > arr = np.arange(9).reshape(3,3)
# 출력 > array([[0, 3, 6],
#                 [1, 4, 7],
#                 [2, 5, 8]])
arr = np.arange(9).reshape(3,3)
print(arr)
np.swapaxes(arr, 0, 1)
print(arr)



# Q. 2D 배열의 행을 뒤집습니다.
# 입력 >  arr = np.arange(9).reshape(3,3)
# 출력 > array([[2, 1, 0],
#                 [5, 4, 3],
#                 [8, 7, 6]])
arr = np.arange(9).reshape(3,3)
print(arr)
np.flip(arr, 0)
print(arr)

# Q. 2D 배열의 열을 뒤집습니다.
# 입력 >  arr = np.arange(9).reshape(3,3)
# 출력 > array([[8, 7, 6],
#                 [5, 4, 3],
#                 [2, 1, 0]])
arr = np.arange(9).reshape(3,3)
print(arr)
np.flip(arr, 1)
print(arr)


# Q. 5에서 10 사이의 임의의 십진수를 포함하는 5x3 모양의 2D 배열을 만듭니다.
# 입력 >  np.random.randint(5, 10, (5, 3))
# 출력 > array([[7, 8, 9],
#                 [6, 7, 8],
#                 [5, 6, 7],
#                 [4, 5, 6],
#                 [3, 4, 5]])
np.random.randint(5, 10, (5, 3))
print(np.random.randint(5, 10, (5, 3)))


# Q. 5에서 10 사이의 임의의 부동 소수점을 포함하는 5x3 모양의  2D 배열 만듭니다.
# 입력 >  np.random.rand(5, 3)
# 출력 > array([[0.85714286, 0.92857143, 0.96428571],
#                 [0.85714286, 0.92857143, 0.96428571],
#                 [0.85714286, 0.92857143, 0.96428571],
#                 [0.85714286, 0.92857143, 0.96428571],
#                 [0.85714286, 0.92857143, 0.96428571]])
np.random.rand(5, 3)
print(np.random.rand(5, 3))


# Q. numpy 배열의 소수점 이하 세 자리만 출력합니다.
# 입력 > rand_arr = np.random.random((5,3))
# 출력 > array([[ 0.898,  0.898,  0.898],
#                [ 0.898,  0.898,  0.898],
#                [ 0.898,  0.898,  0.898],
#                [ 0.898,  0.898,  0.898],
#                [ 0.898,  0.898,  0.898]])
rand_arr = np.random.random((5,3))
print(np.around(rand_arr, 3))

 