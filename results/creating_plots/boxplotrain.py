import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style("darkgrid")

dict_rain  ={
    "Performance":[],
    "Values": [],
    "Resolution": [],
}

files = ["resultpreciserain.txt", "resultpostcode6rain.txt", "resultpostcode4rain.txt"]
for file in files:
    with open("result_texts/" + file, "r") as f:
        all_lines = f.read().splitlines()
        for line in all_lines:
            if "Accuracy" in line:
                value = float(line.split(" ")[1])
                dict_rain["Performance"].append("Accuracy")
                dict_rain["Values"].append(value)
                if file == "resultpreciserain.txt":
                    dict_rain["Resolution"].append("Adres")
                elif file == "resultpostcode6rain.txt":
                    dict_rain["Resolution"].append("Postcode 6")
                elif file == "resultpostcode4rain.txt":
                    dict_rain["Resolution"].append("Postcode 4")
                    
            elif "Precision" in line:
                value = float(line.split(" ")[1])
                dict_rain["Performance"].append("Precision")
                dict_rain["Values"].append(value)
                if file == "resultpreciserain.txt":
                    dict_rain["Resolution"].append("Adres")
                elif file == "resultpostcode6rain.txt":
                    dict_rain["Resolution"].append("Postcode 6")
                elif file == "resultpostcode4rain.txt":
                    dict_rain["Resolution"].append("Postcode 4")
                    
            elif "Recall" in line:
                value = float(line.split(" ")[1])
                dict_rain["Performance"].append("Recall")
                dict_rain["Values"].append(value)
                if file == "resultpreciserain.txt":
                    dict_rain["Resolution"].append("Adres")
                elif file == "resultpostcode6rain.txt":
                    dict_rain["Resolution"].append("Postcode 6")
                elif file == "resultpostcode4rain.txt":
                    dict_rain["Resolution"].append("Postcode 4")
                    
df_rain = pd.DataFrame(data=dict_rain)

plt.rcParams['figure.figsize'] = (12,8)

ax = sns.boxplot(x='Performance', y='Values', hue='Resolution', data=dict_rain, width=0.4)
plt.ylabel("Prestatie", fontsize= 12)
plt.title("Alleen neerslag attribuut", fontsize= 15)
ax.set(ylim=(0, 1))

plt.savefig("alleenneerslag.png")