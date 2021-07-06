import pandas as pd
import glob
import os

owd = os.getcwd()

files = []
os.chdir('dataset/cuisine_recipe_ingredient')
for name in glob.glob('*.txt'):
    files.append(name)

first_file = pd.read_csv(files[0], sep='\t', names=['recipe_id', 'ingredient'])

os.chdir(owd)
os.makedirs(owd+'/cuisine_recipe_ingredient_CSV')

for txts_path in files:
    os.chdir('dataset/cuisine_recipe_ingredient')
    csv_file = pd.read_csv(txts_path, sep='\t', names=['recipe_id', 'ingredient'])
    os.chdir(owd)
    csv_file.to_csv(f'cuisine_recipe_ingredient_CSV/{txts_path.split("_",1)[0]}.csv')
