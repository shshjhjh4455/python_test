import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



#가격이 10000원인 주식이 있고, 이 주식의 일간 수익률은 평균 0%, 표준편차가 1%인 정규분포를 따른다고 가정하고 250일 동안의 주가를 무작위로 생성하시오.
s =[10000+int(np.random.normal(0,1)*100) for i in range(250)]
print(s)
plt.plot(s)
plt.show()


#주사위 100번 던서 나오는 숫자의 평균을 구하시오.
dice = np.random.randint(1,7,size=100)
print(dice)
print(np.mean(dice))
