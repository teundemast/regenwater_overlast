import os 
cwd = os.getcwd()
fname = "__init__.py"

for subdir, dirs, files in os.walk(cwd):
    with open(os.path.join(subdir, fname), 'w') as f:
        continue
