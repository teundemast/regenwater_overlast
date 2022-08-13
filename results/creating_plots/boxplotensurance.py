import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style("darkgrid")

dict_height_and_rain = {
    "Performance": [],
    "Values": [],
    "Resolution": [],
}

files = ["result_ensurance_all.txt", "result_ensurance_height.txt", "result_ensurance_rain.txt"]
for file in files:
    with open("result_texts/" + file, "r") as f:
        all_lines = f.read().splitlines()
        for line in all_lines:
            if "Accuracy" in line:
                value = float(line.split(" ")[1])
                dict_height_and_rain["Performance"].append("Accuracy")
                dict_height_and_rain["Values"].append(value)
                if file == "result_ensurance_rain.txt":
                    dict_height_and_rain["Resolution"].append("Only rain")
                elif file == "result_ensurance_height.txt":
                    dict_height_and_rain["Resolution"].append("Only height")
                elif file == "result_ensurance_all.txt":
                    dict_height_and_rain["Resolution"].append("Both rain and height")

            elif "Precision" in line:
                value = float(line.split(" ")[1])
                dict_height_and_rain["Performance"].append("Precision")
                dict_height_and_rain["Values"].append(value)
                if file == "result_ensurance_rain.txt":
                    dict_height_and_rain["Resolution"].append("Only rain")
                elif file == "result_ensurance_height.txt":
                    dict_height_and_rain["Resolution"].append("Only height")
                elif file == "result_ensurance_all.txt":
                    dict_height_and_rain["Resolution"].append("Both rain and height")

            elif "Recall" in line:
                value = float(line.split(" ")[1])
                dict_height_and_rain["Performance"].append("Recall")
                dict_height_and_rain["Values"].append(value)
                if file == "result_ensurance_rain.txt":
                    dict_height_and_rain["Resolution"].append("Only rain")
                elif file == "result_ensurance_height.txt":
                    dict_height_and_rain["Resolution"].append("Only height")
                elif file == "result_ensurance_all.txt":
                    dict_height_and_rain["Resolution"].append("Both rain and height")

df_rain = pd.DataFrame(data=dict_height_and_rain)

plt.rcParams['figure.figsize'] = (12, 8)

ax = sns.boxplot(x='Performance', y='Values', hue='Resolution', data=dict_height_and_rain, width=0.5, palette="Set2")
plt.ylabel("Performance", fontsize=12)
plt.savefig("ensurance.png")