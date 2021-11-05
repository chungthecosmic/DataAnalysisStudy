import numpy as np
import pandas as pd

if __name__ == '__main__':
    ### csv 파일 불러오기 ###
    titanic_df = pd.read_csv(r'data/titanic_train.csv')

    # print(type(titanic_df)) # titanic의 데이터 타입
    # print(titanic_df.shape) # 모양
    # print(titanic_df.describe()) # titanic의 상세 정보

    # value_counts (각각의 값들에 대한 count)
    value_counts = titanic_df['Pclass'].value_counts()
    # print(value_counts)

    # 특정 column (Series)
    titanic_pclass = titanic_df['Pclass']
    # print(titanic_pclass.head())

    # 1차원 배열 -> numpy로 DataFrame 만들기
    col_name1 = ['col']
    list1 = [1, 2, 3]
    array1 = np.array(list)
    df_list1 = pd.DataFrame(list1, columns=col_name1)
    # print(df_list1)

    # 2차원 배열 -> numpy로 DataFrame 만들기
    col_name2 = ['col1', 'col2', 'col3']
    list2 = [[1, 2, 3], [11, 12, 13]]
    array2 = np.array(list2)
    df_list2 = pd.DataFrame(list2, columns=col_name2)
    # print(df_list2)


    ############################
    ### Dictionary로 DF 만들기 ###
    ############################
    dict = {'col1':[1,11], 'col2':[2,22], 'col3':[3,33]}
    df_dict = pd.DataFrame(dict)
    # print(df_dict)

    titanic_df['Age_0'] = 0
    # print(titanic_df.head(3))
    titanic_df['Age_by_10'] = titanic_df['Age']*10
    # print(titanic_df['Age_by_10'])

    titanic_df['Age_by_10'] = titanic_df['Age_by_10'] + 100
    # print(titanic_df['Age_by_10'])


    # DF에서 데이터 삭제
    titanic_drop_df = titanic_df.drop('Age_0', axis=1) # 원본은 삭제되지 않음
    # print(titanic_drop_df.head(3))
    titanic_drop_df = titanic_df.drop('Age_0', axis=1, inplace=True) # 원복 삭제
    # print(titanic_df.head(3))


    ############
    ### 인덱스 ###
    ############
    titanic_df = pd.read_csv(r'data/titanic_train.csv')
    # print(titanic_df)
    indexes = titanic_df.index
    # print(indexes)  # 인덱스 정보
    # print(indexes.values) # 인덱스 값들
    # print(indexes[:5].values)   # 0~5 까지의 인덱스 값들
    # print(indexes[6])   # 6번째 인덱스 값


    # Series (SQL같은 집계 함수도 사용 가능!)
    series_fair = titanic_df['Fare']
    # print(series_fair)
    # print(series_fair.max())    # 최대값
    # print(series_fair.sum())    # 합계계
    # print(sum(series_fair))
    # print(series_fair + 3)    # 각각의 컬럼에 + 3

    titanic_reset_df = titanic_df.reset_index(inplace=False)    # 인덱스 리셋
    # print(titanic_reset_df)


    # 데이터 추출
    # print(titanic_df[['Fare', 'Cabin']])
    # print(titanic_df[0:2])
    # print(titanic_df[titanic_df['Pclass'] == 3])    # boolean 인덱싱


    ############
    ### 인덱싱 ###
    ############
    # loc, iloc
    data = {
        'Name' : ['Chulmin', 'Eunkyung', 'Jinwoong', 'Soobeom'],
        'Year' : [2011, 2016, 2015, 2015],
        'Gender' : ['Male', 'Female', 'Male', 'Male']
    }   # data form
    data_df = pd.DataFrame(data, index = ['one','two','three','four'])
    # print(data_df)

    # print(data_df.iloc[0,0])    # [0, 0] 위치의 데이터 출력
    # print(data_df.loc['one', 'Name'])   # index, Column Name 으로 출력

    # print(data_df.iloc[0:2,2])
    # print(data_df.loc['one':'three','Gender'])  # loc에서 범위 설정할 경우 끝의 것까지 포함됨!!!!


    # Boolean 인덱싱
    titanic_df = pd.read_csv(r'data/titanic_train.csv')
    titanic_boolean = titanic_df[titanic_df['Age'] > 60]    # Age가 60 초과하는 데이터만 추출
    # print(titanic_boolean)

    titanic_boolean = titanic_df[titanic_df['Age'] > 60][['Name', 'Age']]   # 특정 row 값만 추출
    # print(titanic_boolean)


    ###############
    ### AGG 함수 ###
    ###############

    titanic_sorted = titanic_df.sort_values(by=['Name'])
    # print(titanic_sorted)

    titanic_sorted = titanic_df.sort_values(by=['Name', 'Pclass'], ascending=False)
    # print(titanic_sorted)

    # print(titanic_df[['Age','Fare']].mean())    # Age, Fare column의 평균값

    titanic_groupby = titanic_df.groupby(by='Pclass')[['PassengerId', 'Survived']].count()
    # print(titanic_groupby)



    ####################
    ### 결손 데이터 처리 ###
    ####################

    # print(titanic_df.isna().sum())  # NA의 갯수 확인
    titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000')    # C000으로 NA 값 채우기



    ####################
    ### apply lambda ###
    ####################

    titanic_df['Name_len'] = titanic_df['Name'].apply(lambda x : len(x))    # 이름의 길이를 붙이는 작업
    # print(titanic_df[['Name','Name_len']].head(3))

    titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else ('Adult' if x<=60 else 'Elderly')) # if else 구문 사용 시 if 전에 결과값을 넣어야함
    print(titanic_df['Age_cat'])