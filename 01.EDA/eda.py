# 패키지 import

import pandas as pd
from IPython.display import display_html
from matplotlib import font_manager, rc
import platform
import glob
import re
your_os = platform.system()
if your_os == 'Linux':
    rc('font', family='NanumGothic')
elif your_os == 'Windows':
    ttf = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=ttf).get_name()
    rc('font', family=font_name)
elif your_os == 'Darwin':
    rc('font', family='AppleGothic')
rc('axes', unicode_minus=False)

# 통합 함수

def display_frames(frames, num_spaces=0):
    """
    Fuction summary : 
    --------------------------------------------------------------------------
    Parameters description
    - frames :
    - num_spaces :
    """
    t_style = '<table style="display: inline;"'
    tables_html = [df.to_html().replace('<table', t_style)
                   for df in frames]
    
    space = '&nbsp;' * num_spaces
    display_html(space.join(tables_html), raw=True)

def code_dict(code, name, file_name):
    """
    Fuction summary : 코드와 이름을 딕셔너리 형태로 생성하는 함수
    --------------------------------------------------------------------------
    Parameters description
    - code : 코드로 만들고자 하는 열
    - name : 이름으로 만들고자 하는 열
    """
    try:
        df = pd.read_csv('../data/{}.csv'.format(file_name))

        key = df[code].tolist()
        value = df[name].tolist()
    except:
        df = pd.read_csv('../data/{}.csv'.format(file_name), encoding='cp949')

        key = df[code].tolist()
        value = df[name].tolist()
    
    return {key[i]: value[i] for i in range(len(key))}    
   
def hdong_transform(x):
    if (any(map(str.isdigit, x))) | ('본' in x):
        return x[:2] + x[-1]
    else:
        return x
    
def hdong_integrated(df):
    df_copy = df.copy()
    df_copy['HDONG_NM'] = df_copy['HDONG_NM'].apply(hdong_transform)
    df_copy = df_copy.set_index('STD_YMD')
    df_copy = df_copy.groupby([pd.Grouper(freq='MS'), 'GU_NM', 'HDONG_NM'])['POP_CNT'] \
                   .sum().reset_index()
    
    return df_copy

def compared_to_previous(df, columns):
    
    result = []
    month = ['FEB', 'MAR', 'APR', 'MAY']
    epsilon = 1e-20
    
    for num, name in enumerate(month):    
        for col in columns:
            per_19 = df[f'2019-0{num+2}'][col].values
            per_20 = df[f'2020-0{num+2}'][col].values
            compared_prev = pd.Series((per_20 - per_19) / (per_19 + epsilon),
                                       name=col+f'_{name}')
            result.append(compared_prev)
    
    return result
    
    
# 유동인구데이터 전용 함수

def flow_concat(category, f_path):
    """
    Function summary : csv 형태의 유동인구 파일들을 결합하는 함수
    --------------------------------------------------------------------------
    Parameters description
    - category : 유동인구를 그룹핑할 범주형 변수
    - f_path : 찾고자 하는 csv 파일 위치 
    """
    paths = [path for path in glob.glob(f_path) if path.count(category) >= 1]
    df_list = [pd.read_csv(paths[i], sep='|') for i in range(len(paths))]
    df = pd.concat(df_list)
    
    return df




def flow_preprocessing(df):
    """
    Function summary : 유동인구 데이터 전처리 함수
    --------------------------------------------------------------------------
    Parameters description
    - 
    """

    # 첫번째 열의 내용을 두번째 열이 포함하므로 첫번째 열 삭제
    df.drop(['STD_YM'], axis=1, inplace=True)
    # 날짜형 데이터로 변환
    df['STD_YMD'] = df['STD_YMD'].astype('str').astype('datetime64')    
    # code_dict함수를 통해 구와 관련된 딕셔너리 생성
    gu = code_dict('ITG_GU_CD', 'ITG_GU_NM', 'hdong')
    # 행정동 코드를 통해 구 코드 열 생성
    df['GU_CD'] = df['HDONG_CD'].apply(lambda x: str(x)[2:5]).astype('int')
    # 구 코드 열을 이용하여 구 이름 열 생성
    df.insert(1, 'GU_NM', df['GU_CD'].map(gu))
    # 구 코드와 행정동 코드 열 삭제
    df.drop(['GU_CD', 'HDONG_CD'], axis=1, inplace=True)
    
    return df


def flow_melting(df, var_name='VAR', value_name='POP_CNT'):
    """
    Function summary : 유동인구 시간/나이별 데이터 결합
    --------------------------------------------------------------------------
    Parameters description
    - 
    """
    df_melt = pd.melt(df,
                      id_vars=['STD_YMD', 'GU_NM','HDONG_NM'],
                      value_vars=df.columns[4:],
                      var_name=var_name,
                      value_name=value_name).reset_index(drop=True)

    return df_melt

# 카드매출데이터 전용 함수

def card_preprocessing(df):
    """
    Function summary : 카드데이터 열 정리 함수
    --------------------------------------------------------------------------
    Parameters description
    - 
    """
    gu = code_dict('ITG_GU_CD', 'ITG_GU_NM', 'hdong')
    dong = code_dict('ITG_HDONG_CD', 'ITG_HDONG_NM', 'hdong')
    upjong = code_dict('UP_CD', 'UP_NM', 'upjong')
    
    #날짜데이터 datetime으로 변경
    df['STD_DD'] = df['STD_DD'].astype('str').astype('datetime64')
    df['GU_CD'] = df['GU_CD'].astype('str')
    df['DONG_CD'] = df['DONG_CD'].astype('str')
        
    
    if 'SEX_CD' in df.columns:
        df['SEX_CD'] = df['SEX_CD'].astype('category')
        
    if df['DONG_CD'].apply(lambda x: True if len(x) < 6 else False).all():
        df['DONG_CD'] = df['GU_CD'] + df['DONG_CD']
        df['GU_CD'] = df['GU_CD'].astype('int')
        df['DONG_CD'] = df['DONG_CD'].astype('int')
        
    #구이름, 행정동이름, 업종이름 열 생성
    df.insert(1, 'GU_NM', df['GU_CD'].map(gu))
    df.insert(2, 'DONG_NM', df['DONG_CD'].map(dong))
    df.insert(3, 'UP_NM', df['MCT_CAT_CD'].map(upjong))
    
    #구코드, 행정동코드, 업종코드 열 삭제
    df.drop(['GU_CD', 'DONG_CD', 'MCT_CAT_CD'], axis=1, inplace=True)
    
    return df

def upjong_sales(df):
    sharing_index = pd.read_csv('../data/sharing_index.csv', parse_dates=['STD_YMD'])
    up_col = df['UP_NM'].unique()
    renaming = ['LODGE', 'LEIS_ITEM', 'LEIS', 'ELEC', 'KITCH', 'FUEL', 'OPTICAL', 'DIST',
            'OFFICE', 'CAR_SERVICE', 'MEDIAN', 'HYGIENE', 'REST', 'GROCERY', 'REPAIR',
            'HOBBY', 'FURN', 'APPL', 'CLOTHES', 'ACC', 'BOOK', 'CROP', 'CAR_SALES']
    mapping_dec = {key: value for key, value in zip(up_col, renaming)}
    
    series_list = []
    for up, name in list(mapping_dec.items()):
        up_df = df.query('UP_NM == @up') \
                  .groupby(['STD_YMD', 'GU_NM', 'HDONG_NM'])[['USE_AMT']].sum() \
                  .reset_index()
        up_df.columns = up_df.columns[:3].tolist() + [name + '_AMT']
        up_df = pd.merge(sharing_index, up_df, how='left').fillna(0)
        up_df.iloc[:, -1] = up_df.iloc[:, -1].astype('int')
        series_list.append(up_df.iloc[:, -1])

    result = pd.concat(series_list, axis=1)
    
    return result

def sales_to_csv(df):
    up_col = df['UP_NM'].unique()
    renaming = ['LODGE', 'LEIS_ITEM', 'LEIS', 'ELEC', 'KITCH', 'FUEL', 'OPTICAL', 'DIST',
            'OFFICE', 'CAR_SERVICE', 'MEDIAN', 'HYGIENE', 'REST', 'GROCERY', 'REPAIR',
            'HOBBY', 'FURN', 'APPL', 'CLOTHES', 'ACC', 'BOOK', 'CROP', 'CAR_SALES']
    mapping_dec = {key: value for key, value in zip(up_col, renaming)}    
    sales = df.groupby(['STD_YMD', 'GU_NM', 'HDONG_NM'])['USE_AMT'].sum().reset_index()
    for up, name in list(mapping_dec.items()):
        for gender in ['F', 'M']:
            for aged in range(20, 70, 10):
                level1 = df.query('UP_NM == @up')
                for idx, std in enumerate(sales['STD_YMD'].unique()):
                    level2 = level1.query('STD_YMD == @std')
                    for idx2, hdong in enumerate(sales['HDONG_NM'].unique()):  
                        level3 = level2.query('HDONG_NM == @hdong')
                        selected = level3.query('AGE_CD in [@aged, @aged+5]' + \
                                                'and SEX_CD == @gender')['USE_AMT'].sum()
                        total = level3['USE_AMT'].sum()
                        
                        if total != 0:
                            sales.loc[idx*69 + idx2, f'SH_{name}_{aged}_{gender}'] = \
                                selected / total
                        else:
                            sales.loc[idx*69 + idx2, f'SH_{name}_{aged}_{gender}'] =  \
                                total
                            
    for idx, name in enumerate(list(map(lambda x: x.lower(), renaming[6:12]))):
        sales.iloc[:, (idx*10+4):(idx*10+14)] \
            .to_csv(f'../data/08_Percent_Matrix/{name}_per.csv', index=False)
                            
# SNS데이터 전용 함수

def initial_preprocessing(df):
    df.rename(columns={'GU_NM(삭제)':'GU_NM', 'DONG_NM(삭제)':'HDONG_NM'},
              inplace=True)
    df.drop(['SEQ', 'GU_CD', 'DONG_CD'], axis=1, inplace=True)
    
    return df

def sns_melting(df, keywords, var_name='VAR', value_name='CNT'):
    df_list = []
    if type(keywords) == str:
        keywords = [keywords]
    for category in keywords:
        cols = [col for col in df.columns if category in col]
        result =  pd.melt(df,
                          id_vars=['GU_NM', 'HDONG_NM'],
                          value_vars=cols,
                          var_name=var_name,
                          value_name=value_name)
        df_list.append(result)
    
    return pd.concat(df_list)

def sns_preprocessing(df, mapping=None, colname='UNNAMED', value_name='CNT'):
    
    df['STD_YMD'] = (df['VAR'].str[-6:] + '01').astype('datetime64')
    
    if mapping != None:
        df[colname] = df['VAR'].apply(lambda x: x[:x.find('_')]).map(mapping)
        df = df[['STD_YMD', 'GU_NM', 'HDONG_NM', colname, value_name]]
    else:
        df = df[['STD_YMD', 'GU_NM', 'HDONG_NM', value_name]]
        
    return df



# 유통데이터 전용 함수

def number_search(x):
    """
    문자열에서 숫자를 찾는 함수입니다.
    """
    
    pattern = re.compile('\d+')
    number = re.search(pattern, x).group()
    
    return number    

def logistics_preprocessing(df):
    df.insert(3, 'ITEM_NM', df['VAR'].apply(number_search))

    names = {'10':'식사', '20':'간식', '30':'마실거리', '40':'홈&리빙',
             '50':'헬스&뷰티', '60':'취미&여가활동', '70':'사회활동',
             '80':'임신/육아', '90':'기호품'}
    
    df['ITEM_NM'] = df['ITEM_NM'].map(names)

    df.drop('VAR', axis=1, inplace=True)
    
    return df


# 물류데이터 전용 함수

def delivery_preprocessing(df):
    
    mapping = {'DL_YMD':'STD_YMD', 'DL_GD_LCLS_NM':'GD_NM', 'CTGG_NM':'GU_NM',
               'HDNG_NM':'HDONG_NM', 'INVC_CONT':'INV_CNT'}
    df.rename(columns=mapping, inplace=True)
    df['STD_YMD'] = pd.to_datetime(df['STD_YMD'], format='%y%m%d')
    df['GU_NM'] = df['CTPV_NM'].str[:2] + ' '+ df['GU_NM']
    df.drop(['CTPV_NM','CTPV_CD', 'CTGG_CD', 'HDNG_CD', 'DL_GD_LCLS_CD'],
            axis=1,inplace=True)
    df['HDONG_NM'].replace('신당제5동', '신당5동', inplace=True)
    df = df[['STD_YMD', 'GU_NM', 'HDONG_NM', 'GD_NM', 'INV_CNT']]

    
    return df


# 감염병데이터 전용 함수

def paste_number(x):    
    pattern = re.compile('\d+')
    numbers = re.findall(pattern, x)
    if len(numbers[1]) == 2:
        result = ''.join(numbers)
    else:
        result = '0'.join(numbers)
    
    return result

def seoul_preprocessing(df):
    df['확진일'] = df['확진일'].apply(lambda x: '20200'+ paste_number(x))
    df['확진일'] = df['확진일'].astype('datetime64')
    df = df.set_index('확진일')
    df = df.loc[:'2020-05-31']
    df.index.name = 'STD_YMD'
    
    return df

def seoul_patient(df, gu_name):
    
    gu = df.query('거주지 == @gu_name').iloc[:, :1]

    gu = gu.resample('W-mon')[['연번']].count()

    gu.rename(columns={'연번':'PT_CNT'}, inplace=True)
    gu['CUM_CNT'] = gu['PT_CNT'].cumsum()
    gu.insert(0, 'GU_NM', '서울 {}'.format(gu_name))
    
    return gu

def daegu_patient(df, gu_name):

    df.rename(columns={'주간 확진자 수':'확진자 수'}, inplace=True)
    df.rename(columns={'날짜':'STD_YMD', '확진자 수':'PT_CNT', '확진자 누계':'CUM_PT'},
                  inplace=True)

    df = df.iloc[:, :3]
    df['STD_YMD'] = df['STD_YMD'].astype('datetime64')
    df = df.set_index('STD_YMD')
    df = df.resample('W-mon')[['PT_CNT', 'CUM_CNT']].sum()
    df = df[:'2020-05']
    df.insert(0, 'GU_NM', '대구 {}'.format(gu_name))
    
    return df 