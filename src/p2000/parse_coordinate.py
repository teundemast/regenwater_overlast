import json

# Check for high number 
# def delete_too_high_number(call):
#     adres =  call["address"]
#     adres = adres.replace(",", " ")
#     integers_in_adres = [int(s) for s in adres.split() if s.isdigit()]
    
#     for i in integers_in_adres:
#         if i > 9999:
#             return True
        
#     return False

# file = "p2000/checkpoints/parsed_w_coords.json"
# with open(file, 'r') as file:
#     data = json.load(file)
    
# copy_data = data
# for call in copy_data:
    
#     if delete_too_high_number(call):
#         print(call)
#         data.remove(call)
        
# out_file = open("coord.json", "w")
# json.dump(data, out_file, indent=6)
# out_file.close()      
        
def check_precise(call):
    location_type = call["google_results"][0]["geometry"]["location_type"] 
    if location_type == "ROOFTOP" or location_type == "RANGE_INTERPOLATED":
        return True 
    return False
    

file = "p2000/checkpoints/parsed_w_coords.json"
with open(file, 'r') as file:
    data = json.load(file)
    
precise_data = []
for call in data:
    try:
        if check_precise(call):
            precise_data.append(call)
    except KeyError:
        continue
        
out_file = open("parsed_w_precise_coords.json", "w")
json.dump(precise_data, out_file, indent=6)
out_file.close()      
        

    