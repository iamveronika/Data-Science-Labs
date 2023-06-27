import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ПІДГОТОВКА ВХІДНИХ ДАНИХ
# Парсінг файлу вхідних даних
# парсинг файлу вхідних даних
data = pd.read_excel('Pr15_sample_data.xlsx', parse_dates=['birth_date'])
print('Вхідні дані:\n', data)
data_desc = pd.read_excel('Pr15_data_description.xlsx')
print('Опис характеристик:\n', data_desc)
data_cb = data_desc[
    (data_desc.Place_of_definition == 'Указывает заемщик')
    | (data_desc.Place_of_definition == 'параметры связанные с выданным продуктом')]
data_cb.index = range(0, len(data_cb))

b = data_cb['Field_in_data']
if set(b).issubset(data.columns):
    print('Повне співпадіння')
else:
    print('Повного співпадіння немає')

# перевірка заявок на наявність у файлі з заявками
# відкинемо всі відсутні файли з заявками
n_columns = data_cb['Field_in_data'].size
j = 0
for i in range(n_columns):
    if {data_cb['Field_in_data'][i]}.issubset(data.columns):
        j += 1
print('Кількість співпадінь:', j)

Columns_Flag_True = np.zeros(j)
j = 0
for i in range(n_columns):
    if {data_cb['Field_in_data'][i]}.issubset(data.columns):
        Columns_Flag_True[j] = i
        j += 1

data_desc_cb_True = data_cb.iloc[Columns_Flag_True]
data_desc_cb_True.index = range(len(data_desc_cb_True))
print('Обрані характеристики:\n', data_desc_cb_True)

b = data_desc_cb_True['Field_in_data']
sample_data_cb = data[b]
# обробка полів, записи в яких відсутні
nullCols = data_desc_cb_True['Field_in_data'].to_numpy()[sample_data_cb.isnull().sum() > 0]
data_desc_cleaning = data_desc_cb_True.loc[~data_desc_cb_True['Field_in_data'].isin(nullCols)]
data_desc_cleaning.index = range(len(data_desc_cleaning))
print('Відсіяні незаповнені поля:\n', data_desc_cb_True)

sample_cleaning = sample_data_cb.drop(columns=nullCols)
sample_cleaning.index = range(len(sample_cleaning))
print('Дані з відсіяними незаповненими полями:\n', sample_cleaning)

# аналіз вхідних даних
# мінімаксна характеристика усіх властивостей
data_desc_minimax = pd.read_excel('d_segment_data_description_minimax.xlsx')
segment_data_desc_minimax = data_desc_minimax.loc[
    (data_desc_minimax['Minimax'] == 'min')
    | (data_desc_minimax['Minimax'] == 'max')]
segment_data_desc_minimax.index = range(len(segment_data_desc_minimax))
print("Minimax характеристика:\n", segment_data_desc_minimax[['Field_in_data',
'Minimax']])

cols = segment_data_desc_minimax['Field_in_data'].values.tolist()
d_segment_sample_minimax = sample_cleaning[cols]

# підготовка даних для багатокритеріального аналізу
d_segment_sample_min = d_segment_sample_minimax[cols].min()
d_segment_sample_max = d_segment_sample_minimax[cols].max()

m = d_segment_sample_minimax['loan_amount'].size
n = segment_data_desc_minimax['Field_in_data'].size
segment_sample_minimax_Normal = np.zeros((m, n))

delta_d = 0.3
for j in range(n):
    columns_d = segment_data_desc_minimax['Minimax'][j]
    columns_m = segment_data_desc_minimax['Field_in_data'][j]
    if columns_d == 'min':
        for i in range(m):
            max_max = d_segment_sample_max[j] + (2 * delta_d)
            segment_sample_minimax_Normal[i, j] = (delta_d +
d_segment_sample_minimax[columns_m][i]) / max_max
    else:
        for i in range(m):
            min_min = d_segment_sample_min[j] + (2 * delta_d)
            segment_sample_minimax_Normal[i, j] = (1 / (delta_d +
d_segment_sample_minimax[columns_m][i])) / min_min
# аналіз і кластеризація за допомогою функції Вороніна
def v_func(data, m, n):
    integ = np.zeros(m)
    score = np.zeros(m)
    for i in range(m):
        v_sum = 0
        for j in range(n):
            v_sum = v_sum + ((1 - data[i, j]) ** (-1))
        integ[i] = v_sum
        score[i] = 1000
    plt.title('Integro - Score')
    plt.scatter(range(m), integ)
    plt.plot(score, c='red')
    plt.show()
    return integ


v_func(segment_sample_minimax_Normal, 220, n)