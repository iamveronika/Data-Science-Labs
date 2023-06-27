import re
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


# парсер сайту для отримання html структури і вилучення з неї стрічки новин  --------
def Parser_URL_ukrinform(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')  # аналіз структури html документу
    y_no = input("Y\\N: ")
    flag = 0
    while flag == 0:

        if y_no == "Y":
            print(soup)
            flag = 1
            continue
        elif y_no == "N":
            flag = 1
            continue
        print("Введіть Y або N")
        y_no = input("Y\\N:  ")
    quotes = soup.find_all('div', class_='newsline')  # вілучення із html документу стрічки новин
    output_file = open('test_2.txt', 'w')
    print('----------------------- Стрічка новин', url, '---------------------------------')
    for quote in quotes:
        print(quote.text)
        quote.encoding = 'cp1251'
        output_file.write(quote.text)  # запис стрічки новин до текстового файлу
    print('------------------------------------------------------------------------------')
    return


# парсер сайту для отримання html структури і вилучення з неї стрічкі новин
def Parser_URL_tsn(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')  # аналіз структури html документу
    print("Вивести html структуру документу?")
    y_no = input("Y\\N: ")
    flag = 0
    while flag == 0:

        if y_no == "Y":
            print(soup)
            flag = 1
            continue
        elif y_no == "N":
            flag = 1
            continue
        print("Введіть Y або N")
        y_no = input("Y\\N:  ")

    quotes = soup.find_all('div', class_='event')  # вілучення із html документу стрічки новин
    output_file = open('test_2.txt', 'w', errors='ignore')
    print('----------------------- стрічка новин', url, '---------------------------------')
    for quote in quotes:
        print(quote.text)
        quote.encoding = 'cp1251'
        output_file.write(quote.text)  # запис стрічки новин до текстового файлу

    print('------------------------------------------------------------------------------')
    return


#Частотний text mining аналіз даних, від сайтів новин
def text_mining_wordcloud(f):
    text = str(f.readlines())
    print(text)
    #Аналіз тексту на частоту слів
    words = re.findall('[a-zA-Z]{2,}', text)  # Регулярний вираз дя слів - більше 2 букв
    stats = {}
    print(words)
    for w in words:
        stats[w] = stats.get(w, 0) + 1
    # print(stats)
    # Виявлення токенів у тексті
    w_ranks = sorted(stats.items(), key=lambda x: x[1], \
                     reverse=True)[0:10]
    # print(w_ranks)
    _wrex = re.findall('[a-zA-Z]+', str(w_ranks))
    _drex = re.findall('[0-9]+', str(w_ranks))

    pl = [p for p in range(1, 11)]
    for j in range(len(_wrex)):
        places = '{} place,{} - {} times'.format(pl[j], _wrex[j], _drex[j])
        print(places)

    # print(_wrex[1])
    text_raw = " ".join(_wrex)  # перетворення токінів у строку
    # ----------------- Побудова домінантної хмари  --------------
    wordcloud = WordCloud().generate(text_raw)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    return


# Докладний частотний text mining аналіз даних, від сайтів новин
def text_mining_ru(filename):
    with open(filename) as file:
        text = file.read()
    text = text.replace("\n", " ")
    text = text.replace(",", "").replace(".", "").replace("?", "").replace("!", "")
    text = text.lower()
    words = text.split()
    words.sort()
    words_dict = dict()
    for word in words:
        if word in words_dict:
            words_dict[word] = words_dict[word] + 1
        else:
            words_dict[word] = 1
    print("Кількість слів: %d" % len(words))
    print("Кількість унікальних слів: %d" % len(words_dict))
    print("Усі використані слова:")
    for word in words_dict:
        print(word.ljust(20), words_dict[word])
    return


#  Головні виклики парсера для отримання даних text mining
print('Оберіть інформаційне джерело:')
print('1 - https://www.ukrinform.ru/block-lastnews')
print('2 - https://tsn.ua/')
mode = int(input('mode:'))

if (mode == 1):
    print('Обрано інформаційне джерело: https://www.ukrinform.ru/block-lastnews')
    url = 'https://www.ukrinform.ru/block-lastnews'
    Parser_URL_ukrinform(url)
    # Частотний text mining аналіз даних від новосних сайтів
    f = open('D:\\KPI\\V семестр\\Data Science\\test_2.txt', 'r')
    print('Домінуючий контент сайту:', mode, ':', url)
    text_mining_wordcloud(f)
    print('Докладний частотний аналіз інформаційного джерела:', mode, ':', url)
    filename = 'D:\\KPI\\V семестр\\Data Science\\test_2.txt'
    text_mining_ru(filename)

if (mode == 2):
    print('Обрано інформаційне джерело: https://tsn.ua/')
    url = 'https://tsn.ua/'
    Parser_URL_tsn(url)
    # - Частотний text mining аналіз даних від новосних сайтів
    f = open('D:\\KPI\\V семестр\\Data Science\\test_2.txt', 'r')
    # f = open('d:\\Projects Python\\Data_Science\\test_text.txt','r')
    print('Домінуючий контент сайту:', mode, ':', url)
    text_mining_wordcloud(f)
    print('Докладний частотний аналіз інформаційного джерела:', mode, ':', url)
    filename = 'D:\\KPI\\V семестр\\Data Science\\test_2.txt'
    text_mining_ru(filename)
