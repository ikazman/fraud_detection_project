import pandas as pd
import matplotlib.pyplot as plt
import math

BENFORD = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]


class Observer:

    def __init__(self, datafile):
        self.df = pd.read_csv(datafile)
        self.total_count = len(self.df)
        self.expected_counts = None
        self.data_observation = None
        self.observed_pct = None

    def cleaner(self):
        self.df.columns = ['date', 'stat']
        self.df['digit'] = self.df['stat'].astype(str).str[0]
        self.df['digit'] = self.df['digit'].astype(int)
        self.data_observation = list(
            pd.Series(self.df['digit'].value_counts()).sort_index()[1:])
        self.expected_counts = [
            round(p * self.total_count / 100) for p in BENFORD]
        values_count = self.df['digit'].value_counts()
        df_count = self.df['digit'].count()
        self.observed_pct = pd.Series(
            values_count / df_count).sort_index()[1:] * 100

    def chi_square_test(self):
        chi_square_stat = 0
        for data, expected in zip(self.data_observation, self.expected_counts):
            chi_square = math.pow(data - expected, 2)
            chi_square_stat += chi_square / expected
        print(f'Статистика Хи-Квадрат = {chi_square_stat:3f}')
        print('Критическое значение статистики при p равном 0.05 - 15.51')
        if chi_square_stat < 15.51:
            print("Реальное распределение сооответствует",
                  "теоретическому распределению.")
        else:
            print("Реальное распределение не сооответствует",
                  "теоретическому распределению.")

        return chi_square_stat < 15.51

    def bar_chart(self):
        _, ax = plt.subplots()

        index = [i + 1 for i in range(len(self.observed_pct))]

        ax.set_title(
            'Данные по приросту заболевших и значения Бенфорда', fontsize=15)
        ax.set_ylabel('Частота (%)', fontsize=16)
        ax.set_xticks(index)
        ax.set_xticklabels(index, fontsize=14)

        rects = ax.bar(index, self.observed_pct, width=0.95,
                       color='black', label='Data')

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height,
                    '{:0.1f}'.format(height), ha='center', va='bottom',
                    fontsize=13)

        ax.scatter(index, BENFORD, s=150, c='red', zorder=2, label='Бенфорд')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(prop={'size': 15}, frameon=False)

        plt.show()

    def check_benford(self):
        self.cleaner()
        self.chi_square_test()
        self.bar_chart()
