from bs4 import BeautifulSoup

with open('test.html','r') as html_file:
    html = BeautifulSoup(html_file)