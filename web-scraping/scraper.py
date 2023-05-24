# Attempt at parsing through Bloomberg & WSJ
# Does not work due to bs4 not being able to interact with javascript

# from bs4 import BeautifulSoup
# import requests
#
# url = "https://www.bloomberg.com/search?query=microsoft"
# page = requests.get(url)
# soup = BeautifulSoup(page.text, 'html.parser')
# print(soup.prettify())

import praw
reddit = praw.Reddit(client_id='KsgaDoyeRrplqEem8B7Ytw', client_secret='', user_agent='cnglang web')
sr = reddit.subreddit("stocks")
for sub in sr.hot(limit=10):
    print(sub.title)
    print(sub.score)
    print(sub.url)
