import precipitation_nl


PNL = precipitation_nl.PrecipitationNL(queue_size=300)

def find_wet_days(year):
    wet_days = []
    for month in range(12):
        for day in range(28):
            day_fall = PNL.get_precipation_data_past_hours_list(year, month, day, 23, 59, 52.1092717, 5.1809676, 23)
            sum_day_fall = sum(day_fall)
            print(sum_day_fall)
            
if __name__ == '__main__':
    find_wet_days(2017)
            