import json 

file = "p2000/checkpoints/parsed_w_precise_coords.json"
with open(file, 'r') as file:
    data = json.load(file)
    
file2 = "p2000/checkpoints/precise_coord.json"
with open(file2, 'r') as file2:
    data2 = json.load(file2)
    
data = data + data2

out_file = open("merged_precise_coord.json", "w")
json.dump(data, out_file, indent=6)
out_file.close()    