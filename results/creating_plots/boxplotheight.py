import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style("darkgrid")

dict_height  ={
    "Performance":[],
    "Values": [],
    "Resolution": [],
}

files = ["result_precise_height_all.txt", "result_postcode6_height_all.txt", "result_postcode4_height_all.txt"]
for file in files:
    with open("result_texts/" + file, "r") as f:
        all_lines = f.read().splitlines()
        for line in all_lines:
            if "Accuracy" in line:
                value = float(line.split(" ")[1])
                dict_height["Performance"].append("Accuracy")
                dict_height["Values"].append(value)
                if file == "result_precise_height_all.txt":
                    dict_height["Resolution"].append("Object level")
                elif file == "result_postcode6_height_all.txt":
                    dict_height["Resolution"].append("Sub-district level")
                elif file == "result_postcode4_height_all.txt":
                    dict_height["Resolution"].append("District level")
                    
            elif "Precision" in line:
                value = float(line.split(" ")[1])
                dict_height["Performance"].append("Precision")
                dict_height["Values"].append(value)
                if file == "result_precise_height_all.txt":
                    dict_height["Resolution"].append("Object level")
                elif file == "result_postcode6_height_all.txt":
                    dict_height["Resolution"].append("Sub-district level")
                elif file == "result_postcode4_height_all.txt":
                    dict_height["Resolution"].append("District level")
                    
            elif "Recall" in line:
                value = float(line.split(" ")[1])
                dict_height["Performance"].append("Recall")
                dict_height["Values"].append(value)
                if file == "result_precise_height_all.txt":
                    dict_height["Resolution"].append("Object level")
                elif file == "result_postcode6_height_all.txt":
                    dict_height["Resolution"].append("Sub-district level")
                elif file == "result_postcode4_height_all.txt":
                    dict_height["Resolution"].append("District level")
                    
df_rain = pd.DataFrame(data=dict_height)

plt.rcParams['figure.figsize'] = (12,8)

ax = sns.boxplot(x='Performance', y='Values', hue='Resolution', data=dict_height, width=0.5)
plt.ylabel("Performance", fontsize= 12)


plt.savefig("alleenhoogte.png")
