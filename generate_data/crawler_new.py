from  selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re

proxy_list = [""]
Open_N = 10

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
# 设置代理避免被ban
# proxy_url = 'http://103.1.50.45:3125'

# 创建一个Chrome浏览器实例
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--ignore-certificate-errors')
# chrome_options.add_argument('--headless')  # 无头模式，不显示浏览器界面
# chrome_options.add_argument('--proxy-server=' + proxy_url)
driver = webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)
# driver = webdriver.Chrome(service=ChromeService(executable_path=chrome_driver_path), options=chrome_options)

opening_file = r"generate_data\one_thousand_opening.txt"
cnt = 0
# [CHANGE] 查看该比赛棋谱的网页数量修改下面的数字
N_page = 5
for i in range(1,N_page):

    # 目标网页的URL
    # [CHANGE] 修改
    url = f'https://www.ljrenju.com/news/cnjsu2022/zhesp2022r{i}.htm'

    # 打开网页
    driver.get(url)

    # 等待页面加载完成（你可能需要调整等待时间）
    wait = WebDriverWait(driver, 10)

    # wait.until(EC.presence_of_element_located((By.valign, 'top')))
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'td[valign="top"]')))
    # 获取页面内容
    page_content = driver.page_source

    soup = BeautifulSoup(page_content, 'html.parser')
    # print(soup)

    table_elements = soup.find_all('table', style=re.compile(r'^border'))
    opening_str_out = ""




    for table in table_elements:
        opening_str_out = ""
        keys =  table.find_all("span",style="margin-left: auto; margin-right: auto; font-size: 11.5294px;")
        data = []
        for key in keys:
            posi =  key.parent.parent["id"]
            pattern = r'\d+'  # 匹配一个或多个数字
            numbers = re.findall(pattern, posi)
            posi =  (numbers[2] , numbers[3])
            data.append((posi, int(key.text)))
        # print(data)
        data = sorted(data, key=lambda x: x[1])
        opening_str = check(data[:Open_N])
        opening_str_out = opening_str_out + opening_str + "\n"
        cnt += 1
        print(opening_str_out)
        with open(opening_file, "a") as f:
            f.write(opening_str_out)


print("Total: ",cnt)



        

    









# 在这里进行你的爬取操作，例如找到特定标签的内容
# 例如，找到所有的<a>标签
# print(soup)


# 关闭浏览器
driver.quit()
