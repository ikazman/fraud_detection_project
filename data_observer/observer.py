import os

import pandas as pd
from scipy.stats import chisquare, chi2
import matplotlib.pyplot as plt
import math


BENFORD = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]


class Observer:

    def __init__(self):
        self.df = None
        self.total_count = None
        self.name = None
        self.expected_counts = None
        self.data_observation = None
        self.observed_pct = None
        self.test_passed = None

    def cleaner(self):
        self.df.columns = ['date', 'stat']
        self.df['digit'] = self.df['stat'].astype(str).str[0]
        self.df['digit'].replace('-', '1', inplace=True)
        self.df = self.df[self.df['digit'] != '0']
        self.total_count = len(self.df)
        self.df['digit'] = self.df['digit'].astype(int)
        self.data_observation = list(
            pd.Series(self.df['digit'].value_counts()).sort_index())
        self.expected_counts = [
            round(p * self.total_count / 100) for p in BENFORD]
        values_count = self.df['digit'].value_counts()
        df_count = self.df['digit'].count()
        self.observed_pct = pd.Series(
            values_count / df_count).sort_index() * 100

    def check_equal_count(self):
        control_sum = sum(self.data_observation) - sum(self.expected_counts)
        if control_sum > 0:
            self.expected_counts[0] += abs(control_sum)
        elif control_sum < 0:
            self.data_observation[0] += abs(control_sum)
        else:
            pass

    def chi_square_test(self):
        chi_square_stat = 0
        for data, expected in zip(self.data_observation, self.expected_counts):
            chi_square = math.pow(data - expected, 2)
            chi_square_stat += chi_square / expected
        print(f'Статистика Хи-Квадрат = {chi_square_stat:3f}')
        print('Критическое значение статистики при p равном 0.05 - 15.51')
        if chi_square_stat < 15.51:
            print('Реальное распределение сооответствует',
                  'теоретическому распределению.')
            self.test_passed = True
        else:
            print('Реальное распределение не сооответствует',
                  'теоретическому распределению.')
            self.test_passed = False

        return chi_square_stat < 15.51

    def chi_square_test_double(self):
        alpha = 0.05
        chisq, p_value = chisquare(self.data_observation, self.expected_counts)
        critical = chi2.ppf(1 - alpha, self.df['digit'].nunique() - 1)
        if p_value <= alpha and self.test_passed:
            print('Проверь расчеты - в функции ошибка.',
                  f'Для справки: значение статистики - {chisq},',
                  f'критическое значение - {critical},',
                  f'p-value - {p_value}')

    def bar_chart(self):
        _, ax = plt.subplots()

        index = [i + 1 for i in range(len(self.observed_pct))]

        ax.set_title(
            f'{self.name}: прирост заболевших и Бенфорд', fontsize=15)
        ax.set_ylabel('Частота (%)', fontsize=16)
        ax.set_xticks(index)
        ax.set_xticklabels(index, fontsize=14)

        rects = ax.bar(index, self.observed_pct, width=0.95,
                       color='black', label='Данные')

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height,
                    f'{height:0.1f}', ha='center', va='bottom',
                    fontsize=13)

        ax.scatter(index, BENFORD, s=150, c='red', zorder=2, label='Бенфорд')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(prop={'size': 15}, frameon=False)

        plt.show()

    def reader(self):
        for _, _, files in os.walk('data'):
            for csv in files:
                self.df = pd.read_csv('data/' + csv)
                self.total_count = len(self.df)
                self.name = self.df.columns[1]
                self.cleaner()
                self.check_equal_count()
                self.bar_chart()
                self.chi_square_test()
                self.chi_square_test_double()
                print('--------------------------')

    def check_benford(self):
        self.reader()
