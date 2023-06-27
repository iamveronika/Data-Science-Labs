import numpy as np
import math as mt
import matplotlib.pyplot as plt

dm = 0
dsig = 5
n = 10000
square_mod_err = 10
nAV = int((n * square_mod_err) / 100)
SAV = np.zeros(nAV)
SV = np.zeros(n)
SV_AV = np.zeros(n)
S0 = np.zeros(n)
SV0 = np.zeros(n)
S = np.random.normal(dm, dsig, n)

for i in range(n):
    S0[i] = (0.0000005 * i * i)  # квадратична модель реального процесу
    SV[i] = S0[i] + S[i]
    SV0[i] = abs(SV[i] - S0[i])  # урахування тренду в оцінках статистичних хараткеристик
    SV_AV[i] = SV[i]

# модель виміру (квадратичний закон) з нормальний шумом + АНОМАЛЬНІ ВИМІРИ
SSAV = np.random.normal(dm, (3 * dsig), nAV)  # аномальна випадкова похибка з нормальним законом
for i in range(nAV):
    k = int(SAV[i])
    SV_AV[k] = S0[k] + SSAV[i]  # аномальні вимірів з рівномірно розподіленими номерами

# графіки тренда, вимірів з нормальним шумом
plt.plot(SV)
plt.plot(S0)
plt.ylabel('Динаміка продажів')
plt.show()
mSV0 = np.median(SV0)
dSV0 = np.var(SV0)
sigma = mt.sqrt(dSV0)
print('-------- Статистичні характеристики виміряної вибірки без АВ ----------')
print('Матиматичне сподівання ВВ3=', mSV0)
print('Дисперсія ВВ =', dSV0)
print('СКВ ВВ3=', sigma)

a = -0.97
b = 4.3
e = 0.001
func = lambda x: (x + 4) * (x - 1) ** 5 - A * (x + 4)


def half_div_method(a, b, f):
    if f(a) * f(b) >= 0:
        return 100
    x = (a + b) / 2
    while abs(f(x)) >= e:
        x = (a + b) / 2
        a, b = (a, x) if f(a) * f(x) < 0 else (x, b)
    return (a + b) / 2


disp = sigma ** 2
fin_diff = np.diff(SV_AV, 2)

detect_AB = []
for i in range(int(fin_diff.size)):
    A = fin_diff[i] / disp
    res = half_div_method(a, b, func)
    if res == 100:
        detect_AB.append(i)

F = np.ones((n, 3))
Yin = np.zeros((n, 1))
for i in range(n):  # формування структури вхідних матриць МНК
    Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних без аномілій
    F[i, 1] = float(i)
    F[i, 2] = float(i * i)  # формування матриці вхідних даних без аномілій


def MNK(Yin, F):
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return Yout


for i in range(n):
    Yin[i, 0] = float(SV_AV[i])

Yout = MNK(Yin, F)

Yout00 = np.zeros((n));
for i in range(n):
    Yout00[i] = abs(Yout[i] - S0[i])

print('--------------------------- Статистичні характеристики згладженої вибірки  ----------------------------')
mYout00 = np.median(Yout00)
dYout00 = np.var(Yout00)
scvYout00 = mt.sqrt(dYout00)
print('---------------------------- Похибки аномальні ---------------------------------------------------------')
print('Матиматичне сподівання ВВ=  ', mYout00)
print('Дисперсія ВВ =              ', dYout00)
print('СКВ ВВ=                     ', scvYout00)

# гістограми вхідних похибок, МНК оцінок нормальних та аномальних
plt.hist(S, bins=20, alpha=0.5, label='SV0')
plt.hist(Yout00, bins=20, alpha=0.5, label='Yout00')
plt.show()
