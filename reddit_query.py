import os
import json
import re
import praw
import pandas as pd
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser

PROMPT_STR = """
Come up with general thoughts that youtube viewers would have regarding {} based on these video titles. I don't want topics for every video title, just the ones that stick out to you! :) Don't give me any more than 10 thoughts in total. Separate each idea with the following characters : || 

List the ideas in the following format : "thoughts: thought1||thought2||,etc"

"""

class RedditSearcher:
    def __init__(self, subreddit, search_term, sort_by='relevance'):
        self.sub_choice = subreddit
        self.search_term = search_term
        self.post_id_list = []
        self.main_dir = './data/reddit_search_results'
        with open('reddit_credentials.json', 'r') as creds_file:
            creds = json.load(creds_file)
            client_id = creds['client_id']
            client_secret = creds['client_secret']

        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent='Mozilla/5.0 (Windows NT 10.0; rv:78.0) Gecko/20100101 Firefox/78.0')

        self.subreddit = self.reddit.subreddit(self.sub_choice)

    def search_posts(self, post_limit=22):
        if post_limit < 0:
            post_limit = None
        
        search_results = self.subreddit.search(self.search_term, limit=post_limit, sort=parsed_args.sort, time_filter=parsed_args.time_filter)
        post_list = []
        
        for post in search_results:
            post_id = post.id
            self.post_id_list.append(post_id)
            
            post_data = {
                "post_id": post_id,
                "title": post.title,
                "author": post.author.name if post.author else None,
                "created_utc": post.created_utc,
                "url": post.url,
                "num_comments": post.num_comments,
                "score": post.score,
            }
            post_list.append(post_data)
        
        self.save_posts_to_csv(post_list)

    def save_posts_to_csv(self, post_list):
        if not os.path.exists(self.main_dir):
            os.makedirs(self.main_dir)
        
        post_df = pd.DataFrame(post_list)
        search_term_dir = Path(self.main_dir) / self.sub_choice / self.search_term
        if not os.path.exists(search_term_dir):
            os.makedirs(search_term_dir)
        csv_path = Path(self.main_dir) / self.sub_choice / self.search_term / f'{parsed_args.sort}_{parsed_args.time_filter}_posts.csv'
        #csv_path = Path(self.main_dir) / f'{self.sub_choice}_{self.search_term}_posts.csv'
        post_df.to_csv(csv_path, index=False)

    def get_comments(self):
        
        for post_id in self.post_id_list:
            comment_list = []
            post = self.reddit.submission(id=post_id)
            #post.comments.replace_more(limit=None)
            comments = post.comments.list()
            for comment in comments:
                comment_data = {
                    "post_id": post_id,
                    "comment_id": comment.id,
                    "author": comment.author.name if comment.author else None,
                    "created_utc": comment.created_utc,
                    "body": comment.body,
                    "score": comment.score,
                }
                comment_list.append(comment_data)
            
            self.save_comments_to_csv(comment_list)

    def save_comments_to_csv(self, comment_list):
        comment_df = pd.DataFrame(comment_list)
        search_term_dir = Path(self.main_dir) / self.sub_choice / self.search_term
        if not os.path.exists(search_term_dir):
            os.makedirs(search_term_dir)
        csv_path = Path(self.main_dir) / self.sub_choice / self.search_term / f'{parsed_args.sort}_{parsed_args.time_filter}_comments.csv'
        comment_df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    subreddit_name = "musicmarketing"
    search_query = "social media"
    args = ArgumentParser()
    args.add_argument("--subreddit", type=str, default=subreddit_name)
    args.add_argument("--search_query", type=str, default=search_query)
    args.add_argument('--sort', type=str, default='relevance', choices=['relevance', 'new', 'top', 'comments', 'hot'])
    args.add_argument('--time_filter', type=str, default='all', choices=['all', 'day', 'hour', 'month', 'week', 'year'])
    parsed_args = args.parse_args()

    
    reddit_searcher = RedditSearcher(parsed_args.subreddit, parsed_args.search_query, parsed_args.sort)
    reddit_searcher.search_posts()
    reddit_searcher.get_comments()
