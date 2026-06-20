import urllib.request
url = 'https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Housing.csv'
urllib.request.urlretrieve(url, 'Housing.csv')
print("Downloaded successfully.")
