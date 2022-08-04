import precipitation_nl


PNL = precipitation_nl.PrecipitationNL(queue_size=300)

def find_wet_days(year, out_file):
    wet_days = []
    for month in range(1,13):
        print(month)
        for day in range(1,28):
            day_fall = PNL.get_precipation_data_past_hours_list(year, month, day, 23, 59, 52.1092717, 5.1809676, 23)
            sum_day_fall = sum(day_fall)
            if sum_day_fall > 700:
                wet_days.append(f"{year}-{month}-{day}")
    with open(out_file, 'w') as f:   
        for day in wet_days:
            f.write(day)
            f.write("\n")
        
            
if __name__ == '__main__':
    for year in range(2017,2022):
        out_file = f"wet_days{year}.txt"
        find_wet_days(year, out_file)
            