"""
FileName: app.py
Author: Benhao Huang
Create Date: 2023/11/19
Description: Main Page of our Demo
"""

import streamlit as st
# 设置页面配置
st.set_page_config(
    page_title="AI 3603 Gomoku Project",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# 大标题
st.write('<h1 style="text-align: center; color: black; font-weight: bold;">AI 3603 Gomoku Project 👋</h1>', unsafe_allow_html=True)
# 项目参与者
st.write('<p style="text-align: center; font-size: 20px;"><a href="https://github.com" style="color: blue; font-weight: normal; margin-right: 20px; text-decoration: none;">Jiaxin Li</a> \
<a href="https://github.com" style="color: blue; font-weight: normal; margin-right: 20px; text-decoration: none;">Junzhe Shen</a> \
<a href="https://github.com" style="color: blue; font-weight: normal; text-decoration: none;">Benhao Huang</a></p>', unsafe_allow_html=True)
# 标签
st.markdown("""
<div style="text-align: center;">
<a href="#" style="background-color: #343a40; color: white; font-size: 15px; padding: 10px 15px; margin: 5px; border-radius: 15px; text-decoration: none;">📄 Report</a>
<a href="#" style="background-color: #343a40; color: white; font-size: 15px; padding: 10px 15px; margin: 5px; border-radius: 15px; text-decoration: none;">💻 Code</a>
<a href="#" style="background-color: #343a40; color: white; font-size: 15px; padding: 10px 15px; margin: 5px; border-radius: 15px; text-decoration: none;">🌐 Space</a>
<a href="#" style="background-color: #343a40; color: white; font-size: 15px; padding: 10px 15px; margin: 5px; border-radius: 15px; text-decoration: none;">📊 PPT</a>
</div>
</br>
</br>
""", unsafe_allow_html=True)
# 项目介绍
st.markdown("""
<div style='color: black; font-size:18px'>Gomoku is an abstract strategy board game. Also called <span style='color:red;'>Gobang</span> or <span style='color:red;'>Five in a Row</span>, 
it is traditionally played with Go pieces (black and white stones) 
on a Go board. It is straightforward and fun, but also full of strategy and challenge.
Our project is aiming to apply Machine Learning techniques to build a powerful Gomoku AI.</div>
""",
unsafe_allow_html=True)
# 创新点和图片展示
st.write("<h2 style='text-align: center; color: black; font-weight: bold;'>Innovations We Made 👍</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.image("assets/favicon_circle.png", width=50)  # 替换为你的图片 URL
    st.caption("Innovation 1")
with col2:
    st.image("assets/favicon_circle.png", width=50)  # 替换为你的图片 URL
    st.caption("Innovation 2")
with col3:
    st.image("assets/favicon_circle.png", width=50)  # 替换为你的图片 URL
    st.caption("Innovation 3")
# 代码框架阐述和代码组件
st.write("<h2 style='text-align: center; color: black; font-weight: bold;'>Code Structure 🛠️</h2>", unsafe_allow_html=True)
st.code("""
import os
import streamlit as st
def main():
    # your code here
if __name__ == "__main__":
    main()
""", language="python")
st.markdown("---")