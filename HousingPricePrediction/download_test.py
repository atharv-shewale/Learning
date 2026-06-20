import urllib.request
urls = [
    'https://raw.githubusercontent.com/Shivanandroy/House-Price-Prediction/main/Housing.csv',
    'https://raw.githubusercontent.com/ashbabu/House-Price-Prediction/master/Housing.csv',
    'https://raw.githubusercontent.com/amankharwal/Website-data/master/Housing.csv',
    'https://raw.githubusercontent.com/codebasics/py/master/ML/1_linear_reg/Exercise/hiring.csv',
    'https://raw.githubusercontent.com/yasserh/housing-prices-dataset/main/Housing.csv',
    'https://raw.githubusercontent.com/krishnaik06/Multiple-Linear-Regression/master/50_Startups.csv'
]
import urllib.error
success = False
for url in urls:
    try:
        urllib.request.urlretrieve(url, 'Housing.csv')
        print(f"Downloaded from {url}")
        success = True
        break
    except Exception as e:
        pass
if not success:
    print("Could not download.")
