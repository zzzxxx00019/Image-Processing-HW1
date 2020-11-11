import pandas as pd


def Read_Data():
    Barcelona = pd.read_csv('./5cityfeatures/Barcelona.csv')
    Bilbao = pd.read_csv('./5cityfeatures/Bilbao.csv')
    Madrid = pd.read_csv('./5cityfeatures/Madrid.csv')
    Seville = pd.read_csv('./5cityfeatures/Seville.csv')
    Valencia = pd.read_csv('./5cityfeatures/Valencia.csv')
    return Barcelona, Bilbao, Madrid, Seville, Valencia

def Load_Energy_Data():
    Dataset = pd.read_csv('./energy_dataset.csv')
    Dataset = Dataset.drop( columns = ['time'] )
    return Dataset


def Data_Replace( City_Data ):
    City_Data["weather_description"] = City_Data["weather_description"].str.replace(" ","_")

    City_Data["weather_description"] = City_Data["weather_description"].str.replace("light_rain_and_snow", "0")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("ragged_shower_rain", "1")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("light_shower_snow", "2")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("rain_and_snow", "3")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("light_rain_and_snow", "4")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("rain_and_drizzle", "5")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("light_intensity_shower_rain", "6")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("heavy_intensity_shower_rain", "7")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("thunderstorm_with_light_rain", "8")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("light_intensity_drizzle_rain", "9")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("thunderstorm_with_heavy_rain", "10")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("thunderstorm_with_rain", "11")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("light_intensity_drizzle", "12")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("proximity_thunderstorm", "13")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("proximity_moderate_rain", "14")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("proximity_drizzle", "15")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("proximity_shower_rain", "16")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("sky_is_clear", "17")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("few_clouds", "18")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("scattered_clouds", "19")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("broken_clouds", "20")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("overcast_clouds", "21")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("light_rain", "22")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("light_thunderstorm", "23")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("moderate_rain", "24")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("heavy_intensity_rain", "25")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("very_heavy_rain", "26")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("heavy_snow", "27")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("thunderstorm", "28")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("shower_rain", "29")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("mist", "30")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("fog", "31")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("heavy_intensity_drizzle", "32")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("sand_dust_whirls", "33")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("sleet", "34")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("light_snow", "35")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("drizzle", "36")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("snow", "37")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("haze", "38") 
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("dust", "39")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("smoke", "40")
    City_Data["weather_description"] = City_Data["weather_description"].str.replace("squalls", "41")

    #City_Data.drop_duplicates('weather_description', keep='first', inplace = True )
    #City_Data = City_Data.loc[:,'weather_description']

    City_Data = City_Data.drop(columns = ["dt_iso", "weather_main", "weather_icon"])

    print ( City_Data )

    return City_Data


if __name__ == '__main__' :
    Barcelona, Bilbao, Madrid, Seville, Valencia = Read_Data()

    Barcelona = Data_Replace ( Barcelona )
    Barcelona.to_csv('./city_var2/Barcelona.csv')

    Bilbao = Data_Replace ( Bilbao )
    Bilbao.to_csv('./city_var2/Bilbao.csv')

    Madrid = Data_Replace ( Madrid )
    Madrid.to_csv('./city_var2/Madrid.csv')

    Seville = Data_Replace ( Seville )
    Seville.to_csv('./city_var2/Seville.csv')

    Valencia = Data_Replace ( Valencia )
    Valencia.to_csv('./city_var2/Valencia.csv')
    
    #Energy = Load_Energy_Data()

    #data = pd.concat ( [Barcelona, Energy], axis=1 )

    #x = data.iloc[:,1:61].values
    #y = data.iloc[:,61].values

    #print ( x )
    #print ( y )
