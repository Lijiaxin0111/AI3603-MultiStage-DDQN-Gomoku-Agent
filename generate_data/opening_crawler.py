from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re

proxy_list = [""]

def check(data_list):
    data_str = ""
    for data in data_list:
        x = int(data[0][0]) - 7
        y = int(data[0][1]) - 7
        if x < -4 or x > 4 or y < -4 or y > 4:
            return None
        data_str =  data_str +str(x) + ',' + str(y) + ", "
        
    data_str = data_str[:-2]
    return data_str



# 设置ChromeDriver的路径
chrome_driver_path = 'D:\Program_File\googledrive\chromedriver.exe'
proxy_url = 'http://103.1.50.45:3125'

# 创建一个Chrome浏览器实例
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--ignore-certificate-errors')
# chrome_options.add_argument('--headless')  # 无头模式，不显示浏览器界面
chrome_options.add_argument('--proxy-server=' + proxy_url)
driver = webdriver.Chrome(service=ChromeService(executable_path=chrome_driver_path), options=chrome_options)

# 目标网页的URL
url = 'http://qpk.rifchina.com/library/viewOrEdit?id=3018'

# 打开网页
driver.get(url)

# 等待页面加载完成（你可能需要调整等待时间）
wait = WebDriverWait(driver, 10)

wait.until(EC.presence_of_element_located((By.ID, 'board')))

# 获取页面内容
page_content = driver.page_source





soup = BeautifulSoup(page_content, 'html.parser')

td_elements = soup.find_all('td', class_='boardSpot', style=re.compile(r'^color'))
print(td_elements)

data_list = []
# 遍历找到的元素并输出
for td_element in td_elements:
    data = ( td_element["id"].split('_')[1:], int(td_element.text))
    data_list.append(data)

data_list = sorted(data_list, key=lambda x: x[1])
# print(data_list[:10])
Open_N = 10
opening_str = check(data_list[:Open_N])
print(opening_str)


# 在这里进行你的爬取操作，例如找到特定标签的内容
# 例如，找到所有的<a>标签
# print(soup)


# 关闭浏览器
driver.quit()
