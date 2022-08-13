import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style("darkgrid")

dict_height_and_rain  ={
    "Performance":[],
    "Values": [],
    "Resolution": [],
}

files = ["result_precise_rain_height_all.txt", "result_postcode6_all.txt", "result_postcode4_all.txt"]
for file in files:
    with open("result_texts/" + file, "r") as f:
        all_lines = f.read().splitlines()
        for line in all_lines:
            if "Accuracy" in line:
                value = float(line.split(" ")[1])
                dict_height_and_rain["Performance"].append("Accuracy")
                dict_height_and_rain["Values"].append(value)
                if file == "result_precise_rain_height_all.txt":
                    dict_height_and_rain["Resolution"].append("Object level")
                elif file == "result_postcode6_all.txt":
                    dict_height_and_rain["Resolution"].append("Sub-district level")
                elif file == "result_postcode4_all.txt":
                    dict_height_and_rain["Resolution"].append("District level")
                    
            elif "Precision" in line:
                value = float(line.split(" ")[1])
                dict_height_and_rain["Performance"].append("Precision")
                dict_height_and_rain["Values"].append(value)
                if file == "result_precise_rain_height_all.txt":
                    dict_height_and_rain["Resolution"].append("Object level")
                elif file == "result_postcode6_all.txt":
                    dict_height_and_rain["Resolution"].append("Sub-district level")
                elif file == "result_postcode4_all.txt":
                    dict_height_and_rain["Resolution"].append("District level")
                    
            elif "Recall" in line:
                value = float(line.split(" ")[1])
                dict_height_and_rain["Performance"].append("Recall")
                dict_height_and_rain["Values"].append(value)
                if file == "result_precise_rain_height_all.txt":
                    dict_height_and_rain["Resolution"].append("Object level")
                elif file == "result_postcode6_all.txt":
                    dict_height_and_rain["Resolution"].append("Sub-district level")
                elif file == "result_postcode4_all.txt":
                    dict_height_and_rain["Resolution"].append("District level")
                    
df_rain = pd.DataFrame(data=dict_height_and_rain)

plt.rcParams['figure.figsize'] = (12,8)

ax = sns.boxplot(x='Performance', y='Values', hue='Resolution', data=dict_height_and_rain, width=0.5)
plt.ylabel("Performance", fontsize= 12)
plt.savefig("hoogtenneerslag.png")