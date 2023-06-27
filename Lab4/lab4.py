import pandas as pd
import numpy as np


def to_float(f): # функція, що перетворює у дробовий тип елементи таблиці
    for i in range(len(f)):
        f[i][0] = float(str(d['Товар 9'][i]).replace(',', '.'))
        f[i][1] = float(str(d['Товар 8'][i]).replace(',', '.'))
        f[i][2] = float(str(d['Товар 7'][i]).replace(',', '.'))
        f[i][3] = float(str(d['Товар 6'][i]).replace(',', '.'))
        f[i][4] = float(str(d['Товар 5'][i]).replace(',', '.'))
        f[i][5] = float(str(d['Товар 4'][i]).replace(',', '.'))
        f[i][6] = float(str(d['Товар 3'][i]).replace(',', '.'))
        f[i][7] = float(str(d['Товар 2'][i]).replace(',', '.'))
        f[i][8] = float(str(d['Товар 1'][i]).replace(',', '.'))

def Voronin(G0):

    integro = np.zeros(numb_op_sist)
    sum_F = np.zeros(numb_op_sist)

    for i in range(len(F)):
        sum_F[i] = sum(F[i])

    for i in range(len(F)):
        for j in range(len(F)):
            F0[i][j] = F[i][j]/sum_F[i]

    for i in range(len(F)):
        integro[i] = G0[0] * (1 - F0[0][i]) ** (-1) + \
                     G0[1] * (1 - F0[1][i]) ** (-1) + \
                     G0[2] * (1 - F0[2][i]) ** (-1) + \
                     G0[3] * (1 - F0[3][i]) ** (-1) + \
                     G0[4] * (1 - F0[4][i]) ** (-1) + \
                     G0[5] * (1 - F0[5][i]) ** (-1) + \
                     G0[6] * (1 - F0[6][i]) ** (-1) + \
                     G0[7] * (1 - F0[7][i]) ** (-1) + \
                     G0[8] * (1 - F0[8][i]) ** (-1)

    min = 10000
    opt = 0
    for i in range(len(integro)):
        if min > integro[i]:
            min = integro[i]
            opt = i

    print('Integro', integro)
    print('Номер_оптимального_товару =', opt+1)

    return Voronin


numb_op_sist = 9 # к-ть обчислювальних систем

F = []
F0 = []
for i in range(numb_op_sist): # обнулення критеріальних масивів
    F.append(np.zeros(numb_op_sist))
    F0.append(np.zeros(numb_op_sist))

d = pd.read_excel('Pr1.xls') # парсинг вхідного файлу
print('d=', d) # вивід усого масиву файлу Pr1.xlsx
print('----------------STOLB-------------------')
print(d['Товар 5']) # вивід стовпця 'Товар 5'
print()


to_float(F)

for elem in F: # вивід елементів таблиці у дробовому типі
    print(elem)


# коефіціенти переваги критеріїв
G = np.ones(numb_op_sist) # заповнюємо 1, при бажанні, можна змінити
                          # за допомогою доступу до індексу
GNorm = sum(G)
G0 = []

for elem in G:
    G0.append(elem/GNorm)

print()
Voronin(G0)  # функція визначення номеру оптимальної Обчислювальної системи
