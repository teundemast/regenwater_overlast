import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style("darkgrid")

dict_height  ={
    "Performance":[],
    "Values": [],
    "Resolution": [],
}

files = ["resultpreciseheight.txt", "resultpostcode6height.txt", "resultpostcode4height.txt"]
for file in files:
    with open("result_texts/" + file, "r") as f:
        all_lines = f.read().splitlines()
        for line in all_lines:
            if "Accuracy" in line:
                value = float(line.split(" ")[1])
                dict_height["Performance"].append("Accuracy")
                dict_height["Values"].append(value)
                if file == "resultpreciseheight.txt":
                    dict_height["Resolution"].append("Adres")
                elif file == "resultpostcode6height.txt":
                    dict_height["Resolution"].append("Postcode 6")
                elif file == "resultpostcode4height.txt":
                    dict_height["Resolution"].append("Postcode 4")
                    
            elif "Precision" in line:
                value = float(line.split(" ")[1])
                dict_height["Performance"].append("Precision")
                dict_height["Values"].append(value)
                if file == "resultpreciseheight.txt":
                    dict_height["Resolution"].append("Adres")
                elif file == "resultpostcode6height.txt":
                    dict_height["Resolution"].append("Postcode 6")
                elif file == "resultpostcode4height.txt":
                    dict_height["Resolution"].append("Postcode 4")
                    
            elif "Recall" in line:
                value = float(line.split(" ")[1])
                dict_height["Performance"].append("Recall")
                dict_height["Values"].append(value)
                if file == "resultpreciseheight.txt":
                    dict_height["Resolution"].append("Adres")
                elif file == "resultpostcode6height.txt":
                    dict_height["Resolution"].append("Postcode 6")
                elif file == "resultpostcode4height.txt":
                    dict_height["Resolution"].append("Postcode 4")
                    
df_rain = pd.DataFrame(data=dict_height)

plt.rcParams['figure.figsize'] = (12,8)

ax = sns.boxplot(x='Performance', y='Values', hue='Resolution', data=dict_height, width=0.4)
plt.ylabel("Prestatie", fontsize= 12)
plt.title("Alleen hoogte attributen", fontsize= 15)
ax.set(ylim=(0, 1))

plt.savefig("alleenhoogte.png")
