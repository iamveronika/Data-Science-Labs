import numpy as np
import math as mt
import matplotlib.pyplot as plt


# Функція обчислень алгоритму а-в фільтру
def ABF (Yin, iter):
    # Початкові дані для запуску фільтра
    YoutAB = np.zeros((iter, 1))
    T0=1
    Yspeed_retro=(Yin[1, 0]-Yin[0, 0])/T0
    Yextra=Yin[0, 0]+Yspeed_retro
    alfa=2*(2*1-1)/(1*(1+1))
    beta=(6/1)*(1+1)
    YoutAB[0, 0]=Yin[0, 0]+alfa*(Yin[0, 0])

    # Рекурентний прохід по вимірам
    for i in range(1, iter):
        YoutAB[i,0]=Yextra+alfa*(Yin[i, 0]- Yextra)
        Yspeed=Yspeed_retro+(beta/T0)*(Yin[i, 0]- Yextra)
        Yspeed_retro = Yspeed
        Yextra = YoutAB[i,0] + Yspeed_retro
        alfa = (2 * (2 * i - 1)) / (i * (i + 1))
        beta = 6 /(i* (i + 1))
    print('Yin=', Yin, 'YoutAB=', YoutAB)
    return YoutAB

def half_div_method(a, b, f):
    if f(a) * f(b) >= 0:
        return 100
    x = (a + b) / 2
    while abs(f(x)) >= e:
        x = (a + b) / 2
        a, b = (a, x) if f(a) * f(x) < 0 else (x, b)
    return (a + b) / 2

def MNK(Yin, F):
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return Yout

# Початкові дані---------
dm = 0
dsig = 5
n = 100
square_mod_err = 10
nAV = int((n * square_mod_err) / 100)
SAV = np.zeros(nAV)
SV = np.zeros(n)
SV_AV = np.zeros(n)
S0 = np.zeros(n)
SV0 = np.zeros(n)
S = np.random.normal(dm, dsig, n)

for i in range(n):
    S0[i] = (0.0000005 * i * i)  # Квадратична модель реального процесу
    SV[i] = S0[i] + S[i]
    SV0[i] = abs(SV[i] - S0[i])  # Урахування тренду в оцінках статистичних хараткеристик
    SV_AV[i] = SV[i]

# Модель виміру (квадратичний закон) з нормальний шумом + АНОМАЛЬНІ ВИМІРИ
SSAV = np.random.normal(dm, (3 * dsig), nAV)  # Аномальна випадкова похибка з нормальним законом
for i in range(nAV):
    k = int(SAV[i])
    SV_AV[k] = S0[k] + SSAV[i]  # Аномальні вимірів з рівномірно розподіленими номерами

# Графіки тренда, вимірів з нормальним шумом
plt.plot(SV)
plt.plot(S0)
plt.ylabel('Динаміка продажів')
plt.show()
mSV0 = np.median(SV0)
dSV0 = np.var(SV0)
sigma = mt.sqrt(dSV0)
print('Статистичні характеристики виміряної вибірки без АВ')
print('Вибірка ', S)
print('Матиматичне сподівання ВВ3=', mSV0)
print('Дисперсія ВВ =', dSV0)
print('СКВ ВВ3=', sigma)

a, b = -0.97, 4.3
e = 0.001
func = lambda x: (x + 4) * (x - 1) ** 5 - A * (x + 4)

disp = sigma ** 2
finite_diff = np.diff(SV_AV, 2)

detected_AB = []
for i in range(int(finite_diff.size)):
    A = finite_diff[i] / disp
    res = half_div_method(a, b, func)
    if res == 100:
        detected_AB.append(i)

F = np.ones((n, 3))
Yin = np.zeros((n, 1))
for i in range(n):  # Формування структури вхідних матриць МНК
    Yin[i, 0] = float(S0[i])  # Формування матриці вхідних даних без аномілій
    F[i, 1] = float(i)
    F[i, 2] = float(i * i)  # Формування матриці вхідних даних без аномілій

for i in range(n):
    Yin[i, 0] = float(SV_AV[i])

Yout = MNK(Yin, F)

Yout00 = np.zeros((n))
for i in range(n):
    Yout00[i] = abs(Yout[i] - S0[i])

print('--------------------------- Статистичні характеристики згладженої вибірки  ----------------------------')
mYout00 = np.median(Yout00)
dYout00 = np.var(Yout00)
scvYout00 = mt.sqrt(dYout00)
print('---------------------------- Похибки аномальні ---------------------------------------------------------')
print('Вибірка ', Yout00)
print('Матиматичне сподівання ВВ=  ', mYout00)
print('Дисперсія ВВ =              ', dYout00)
print('СКВ ВВ=                     ', scvYout00)
print('-------------------------------------------------------------------------------------------------------')

# Гістограми вхідних похибок, МНК оцінок нормальних та аномальних
plt.hist(S, bins=20, alpha=0.5, label='SV0')
plt.hist(Yout00, bins=20, alpha=0.5, label='Yout00')
plt.show()

# Виклик альфа-бета фільтра
YoutABG = ABF(Yin, n)

# Графіки тренда, альфа-бета фыльтра - оцінок нормального та аномального шуму
plt.plot(Yin)
plt.plot(YoutABG)
plt.show()
