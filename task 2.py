#!/usr/bin/env python
# coding: utf-8

# In[58]:


import time


# In[4]:


get_ipython().system('pip install selenium')


# Task #2
# 

# In[10]:


get_ipython().system('pip install wget')


# In[8]:


#Selenium imports here
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait


# In[12]:


#Other imports here
import os
import wget


# In[41]:


driver = webdriver.Chrome('C:/Aditya/driver/chromedriver.exe')
driver.get('https://twitter.com/i/flow/login')

username = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='text']")))

username.send_keys("jardinains_")
next_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div[role='button'][class='css-18t94o4 css-1dbjc4n r-sdzlij r-1phboty r-rs99b7 r-ywje51 r-usiww2 r-2yi16 r-1qi8awa r-1ny4l3l r-ymttw5 r-o7ynqc r-6416eg r-lrvibr r-13qz1uu']")))
next_button.click()

password = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[class='r-30o5oe r-1niwhzg r-17gur6a r-1yadl64 r-deolkf r-homxoj r-poiln3 r-7cikom r-1ny4l3l r-t60dpp r-1dz5y72 r-fdjqy7 r-13qz1uu']")))
password.send_keys("Astromathsci2005")

log_in = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div[role='button'][class='css-18t94o4 css-1dbjc4n r-sdzlij r-1phboty r-rs99b7 r-19yznuf r-64el8z r-1ny4l3l r-1dye5f7 r-o7ynqc r-6416eg r-lrvibr']")))
log_in.click()



# In[66]:


driver.get('https://twitter.com/elonmusk')


# In[75]:


WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='tweet']")))

n = int(input("Enter the number of tweets you want to print: "))

tweets = []
for i in range(n):
    tweets.extend(driver.find_elements(By.CSS_SELECTOR, "[data-testid='tweet']"))
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)


for tweet in tweets[-n:]:
    print(tweet.text)
    print('--------------')


# In[ ]:




