import json 

def load_data(file):
    with open(file, 'r') as file:
        data = json.load(file)
        return data
    
data1  = load_data('src/data/merged_precise_coord.json')
data2  = load_data('src/data/resolution_change.json')
x = []
for i1 in data1:
    in_dataset = False
    for i2 in data2:
        if i1["date"] == i2["date"]:
            in_dataset = True
    if not in_dataset:
        x.append(i1)
        
with open("src/data/missing.json", "w") as file:
    json.dump(x, file, indent=4)
        
print(x)