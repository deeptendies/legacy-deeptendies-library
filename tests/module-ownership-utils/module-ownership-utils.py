import yaml
import os
import os.path
from pathlib import Path

# read configs
with open(r'module-owners.yaml') as file:
    ownership = yaml.load(file, Loader=yaml.FullLoader)
    # print(ownership)

# get to the root path of this project ( from child dir)
path = os.path.dirname(os.path.abspath(__file__))
file_to_search = "setup.py"
while file_to_search not in os.listdir(path):
    # if path == os.path.dirname(path):
    #     raise FileNotFoundError(f"could not find {file_to_search}")
    path = os.path.dirname(path)
os.chdir(path)

# patch directories if not exist & create github owners
for i in ownership:
    module_dir = os.path.join(path, "src/deeptendies", i)
    if not os.path.exists(module_dir):
        Path(module_dir).mkdir(parents=True, exist_ok=True)
        open(os.path.join(module_dir, "__init__.py"), 'a').close()
        # update github code owner
        codeowners_dir = os.path.join(path, ".github/CODEOWNERS")
        repo_dir = ownership.get(i).get("path")
        owners = ownership.get(i).get("owners")
        owners = " ".join(owners)
        with open(codeowners_dir, 'a') as file:
            file.write(f'\n{repo_dir}'+'\t'*5+f'{owners}')