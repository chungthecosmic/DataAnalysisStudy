import numpy as np
import pandas as pd

if __name__ == '__main__':
    titanic_df = pd.read_csv(r'data/train.csv')

    print(type(titanic_df)) # titanic의 데이터 타입

        