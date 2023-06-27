import numpy as np
import math as mt
import matplotlib.pyplot as plt


def normal(n, dm, dsigm):
    mass = ((np.random.randn(n)) * dsigm) + dm
    stats(mass)
    plt.xlabel('Нормальний розподіл')
    plt.hist(mass, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return mass


def exponential(n):
    mass = np.random.exponential(size = n)
    stats(mass)
    plt.xlabel('Експоненційний розподіл')
    plt.hist(mass, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return mass


def linear(n,mass,error):
    mass3 = np.zeros((n));
    mass0 = np.zeros((n))
    for i in range(n):
        mass0[i] = (error * i)
        mass3[i] = mass0[i] + mass[i]
    plt.plot(mass3)
    plt.plot(mass0)
    plt.xlabel('Експоненціний - Лінійний')
    plt.show()
    return mass3, mass0


def square(n, mass, error):
    mass_1 = np.zeros(n)
    mass_2 = np.zeros(n)

    for i in range(n):
        mass_2[i] = (error * (i * i))
        mass_1[i] = mass_2[i] + mass[i]
    plt.xlabel("Нормальний - Квадратичний")
    plt.plot(mass_1)
    plt.plot(mass_2)
    plt.show()
    return mass_1, mass_2


def stats(mass):
    median_mass = np.median(mass)
    print("Медіана -", median_mass)
    std_mass = np.var(mass)
    print("Дисперсія - " ,std_mass)
    skv_mass = mt.sqrt(std_mass)
    print("Середнє квадратичне відхилення - ",skv_mass)


def assessment(n, mass_1, mass_3, mass_0, mass, label):
    mass_4 = np.zeros(n)
    for i in range(n):
        mass_4[i] = (mass_3[i] - mass_0[i])
    plt.xlabel(label)
    plt.hist(mass, bins=20, alpha=0.5, label='mass')
    plt.hist(mass_1, bins=20, alpha=0.5, label='mass_1')
    plt.hist(mass_3, bins=20, alpha=0.5, label='mass_3')
    plt.hist(mass_4, bins=20, alpha=0.5, label='mass_4')
    plt.show()


n = 20000
dsigm = 5
dm = 5
error = 0.0000005
mass = np.random.randn(n)


normal_mass = normal(n, dm, dsigm)


exponential_mass = exponential(n)


square_n_normal, mass_2 = square(n, normal_mass, error)


plt.xlabel("Закон розподілу: Нормальний - Квадратичний")
plt.hist(normal_mass, bins=20, alpha=0.5, label='mass')
plt.hist(mass, bins=20, alpha=0.5, label='mass_1')
plt.hist(square_n_normal, bins=20, alpha=0.5, label='mass_3')
plt.show()
stats(square_n_normal)


assessment(n, mass, square_n_normal, mass_2, normal_mass, "Статистична характеристика: Нормальний - Квадратичний")


linear_n_exponential, mass_2 = linear(n, exponential_mass, error)


plt.xlabel("Закон розподілу: Експоненційний - Лінійний")
plt.hist(exponential_mass, bins=20, alpha=0.5, label='mass')
plt.hist(mass, bins=20, alpha=0.5, label='mass_1')
plt.hist(linear_n_exponential, bins=20, alpha=0.5, label='mass_3')
plt.show()
stats(linear_n_exponential)


assessment(n, mass, linear_n_exponential, mass_2, exponential_mass, "Статистична характеристик: Експоненційний - Лінійний")

