import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style("darkgrid")

dict_rain = {
    "Performance": [],
    "Values": [],
    "Resolution": [],
}

files = ["result_precise_bouwjaar_no_bouwjaar.txt", "result_precise_bouwjaar_wl_bouwjaar.txt"]
for file in files:
    with open("result_texts/" + file, "r") as f:
        all_lines = f.read().splitlines()
        for line in all_lines:
            if "Accuracy" in line:
                value = float(line.split(" ")[1])
                dict_rain["Performance"].append("Accuracy")
                dict_rain["Values"].append(value)
                if file == "result_precise_bouwjaar_no_bouwjaar.txt":
                    dict_rain["Resolution"].append("Without construction year")
                elif file == "result_precise_bouwjaar_wl_bouwjaar.txt":
                    dict_rain["Resolution"].append("With construction year")

            elif "Precision" in line:
                value = float(line.split(" ")[1])
                dict_rain["Performance"].append("Precision")
                dict_rain["Values"].append(value)
                if file == "result_precise_bouwjaar_no_bouwjaar.txt":
                    dict_rain["Resolution"].append("Without construction year")
                elif file == "result_precise_bouwjaar_wl_bouwjaar.txt":
                    dict_rain["Resolution"].append("With construction year")

            elif "Recall" in line:
                value = float(line.split(" ")[1])
                dict_rain["Performance"].append("Recall")
                dict_rain["Values"].append(value)
                if file == "result_precise_bouwjaar_no_bouwjaar.txt":
                    dict_rain["Resolution"].append("Without construction year")
                elif file == "result_precise_bouwjaar_wl_bouwjaar.txt":
                    dict_rain["Resolution"].append("With construction year")

df_rain = pd.DataFrame(data=dict_rain)

plt.rcParams['figure.figsize'] = (12, 8)

ax = sns.boxplot(x='Performance', y='Values', hue='Resolution', data=dict_rain, width=0.5)
plt.ylabel("Performance", fontsize=12)

plt.savefig("construction.png")