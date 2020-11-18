import pandas as pd
import numpy as np

def Read_Data():
    Barcelona = pd.read_csv('./city_var2/Barcelona.csv')
    Bilbao = pd.read_csv('./city_var2/Bilbao.csv')
    Madrid = pd.read_csv('./city_var2/Madrid.csv')
    Seville = pd.read_csv('./city_var2/Seville.csv')
    Valencia = pd.read_csv('./city_var2/Valencia.csv')
    return Barcelona, Bilbao, Madrid, Seville, Valencia

def Average(Barcelona, Bilbao, Madrid, Seville, Valencia, column_name):
    Barcelona = Barcelona.iloc[:][column_name].values
    Bilbao = Bilbao.iloc[:][column_name].values
    Madrid = Madrid.iloc[:][column_name].values
    Seville = Seville.iloc[:][column_name].values
    Valencia = Valencia.iloc[:][column_name].values

    result = []
    for i in range (35064):
        t = (Barcelona[i] + Bilbao[i] + Madrid[i] + Seville[i] + Valencia[i])/5
        result.append(t)

    Result = pd.DataFrame(result,columns=[column_name])
    return Result

if __name__ == '__main__' :
    Barcelona, Bilbao, Madrid, Seville, Valencia = Read_Data() 
    
    C = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'temp')
    D = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'temp_min')
    E = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'temp_max')
    F = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'pressure')
    G = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'humidity')
    H = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'wind_speed')
    I = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'wind_direction_N')
    J = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'wind_direction_NE')
    K = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'wind_direction_E')
    L = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'wind_direction_ES')
    M = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'wind_direction_S')
    N = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'wind_direction_SW')
    O = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'wind_direction_W')    
    P = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'wind_direction_WN')
    Q = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'rain_1h')
    R = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'rain_3h')
    S = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'snow_3h')
    T = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'clouds_all')
    U = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'weather_id')
    V = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'weather_clear')
    W = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'weather_clouds')
    X = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'weather_drizzle')
    Y = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'weather_dust')
    Z = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'weather_fog')
    AA = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'weather_haze')
    AB = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'weather_mist')
    AC = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'weather_rain')
    AD = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'weather_smoke')
    AE = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'weather_snow')
    AF = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'weather_squall')
    AG = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'weather_thunderstorm')
    AI = Average(Barcelona, Bilbao, Madrid, Seville, Valencia, 'uv')
    
    result_table = pd.concat( [C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA,AB,AC,AD,AE,AF,AG,AI], axis=1 )
    result_table.to_csv('MergeTable.csv')
    