#%%
import os
import re
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from collections import Counter
import networkx as nx
import openai
from wordcloud import WordCloud
import subprocess
import pandas as pd
import traceback
import numpy as np
import seaborn as sns
#%%

def write_response(response, subreddit, search_term, sort, time_filter, comments=False):
    print('Writing response to file...')
    bot_message = response.strip()
    if comments:
        bot_message = f'==================================================\n\nSubreddit: r/{subreddit} Term: {search_term}\n\nSort: {sort} Time Filter: {time_filter}\n\n COMMENTS\n\n==================================================\n\n' + bot_message
    else:
        bot_message = f'==================================================\n\nSubreddit: r/{subreddit} Term: {search_term}\n\nSort: {sort} Time Filter: {time_filter}\n\n==================================================\n\n' + bot_message
    #print(bot_message)
    label_str = 'ideas'
    if comments:
        label_str = 'comment_ideas'
    #add header to response
    #Create directory for subreddit
    subreddit_dir = f'./data/reddit_insights/{subreddit}'
    if not os.path.exists(subreddit_dir):
        os.makedirs(subreddit_dir)
    #Create directory for search term
    search_term_dir = f'{subreddit_dir}/{search_term}'
    if not os.path.exists(search_term_dir):
        os.makedirs(search_term_dir)
    #Write response to file
    with open(f'{search_term_dir}/{sort}_{time_filter}_{label_str}.txt', 'w') as f:
        f.write(bot_message)
    print('Done writing response to file.')
    print(f'File saved to {search_term_dir}/{sort}_{time_filter}_{label_str}.txt')

def write_youtube_response(response, terms, subreddits, out_dir):
    
    print('Writing response to file...')
    bot_message = response.strip()
    bot_message = f'==================================================\n\nTerms: {terms}\n\nSubreddits: {subreddits}\n\n==================================================\n\n' + bot_message
    #print(bot_message)
    #Create directory for subreddit
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #Write response to file
    with open(f'{out_dir}/youtube_ideas.txt', 'w') as f:
        f.write(bot_message)
    with open('./insights.txt', 'w') as f:
        f.write(bot_message)
    print('Done writing response to file.')
    print(f'File saved to {out_dir}/youtube_ideas.txt')
    print(f'Starting youtube title logic...')
    subprocess.run(['python3', 'youtube_gpt.py'])
    print('Done with youtube title logic.')

#sort_choices = ['relevance', 'top', 'new', 'comments', 'hot']
#time_filter_choices = ['year', 'month', 'week', 'day', 'hour', 'all']
sort_choices = ['relevance']
time_filter_choices = ['year']
# Set up NLTK
nltk.download('stopwords')
nltk.download('punkt')

#subreddit = 'vdultcreators'
#target_term = 'grow'
#target_dir = f'./data/reddit_insights/{subreddit}/{target_term}'

# Combine all text files in the given directory into one text file called combined.txt
def combine_text_files(target_dir):
    print(f'COMBINING TEXT FILES IN {target_dir}')
    #if directory doesn't exist, run query
    for text_file in os.listdir(target_dir):
        if text_file.endswith(".txt"):
            if 'comment_ideas' in text_file:
                continue
            text = None
            with open(f'{target_dir}/{text_file}', 'r') as f:
                text = f.read()
            with open(f'{target_dir}/combined.txt', 'w') as f:
                f.write(f'{text}\n\n')


def run_query(subreddit, search_term):
    sort_choices = ['relevance']
    time_filter_choices = ['year']
    for sort in sort_choices:
        for time_filter in time_filter_choices:
            post_csv = f'./data/reddit_search_results/{subreddit}/{search_term}/{sort}_{time_filter}_posts.csv'
            comment_csv = f'./data/reddit_search_results/{subreddit}/{search_term}/{sort}_{time_filter}_comments.csv'
            if not os.path.exists(post_csv):
                print(f'CALLING SUBPROCESS FOR {subreddit} {search_term} {sort} {time_filter}')
                subprocess.run(['python3', 'reddit_query.py', f'--subreddit={subreddit}', f'--search_query={search_term}', f'--sort={sort}', f'--time_filter={time_filter}'])
            try:
                post_df = pd.read_csv(post_csv)
            except Exception:
                print(f'No posts found for {subreddit} {search_term} {sort} {time_filter}')
                write_response('No response found', subreddit, search_term, sort, time_filter)
                continue
            comment_df = pd.DataFrame()
            try:
                comment_df = pd.read_csv(comment_csv)
            except Exception:
                print(f'No comments found for {subreddit} {search_term} {sort} {time_filter}')
            post_df = post_df.sort_values(by='score', ascending=False)
            if len(comment_df) > 1:
                comment_df = comment_df.sort_values(by='score', ascending=False)
            comment_df = comment_df[0:10]
            PROMPT_STR = """
            Come up with general thoughts that reddit users on the subreddit r/{} would have regarding {} based on these post titles. I don't want topics for every post title, just the ones that stick out to you! :) Don't give me any more than 10 thoughts in total.
            \n
            """

            COMMENT_PROMPT_STR = """
            Come up with general thoughts that reddit users on the subreddit r/{} would have regarding {} based on these comments. I don't want topics for every comments, just the ones that stick out to you! :) Don't give me any more than 10 thoughts in total.
            \n
            """
            PROMPT_STR = PROMPT_STR.format(subreddit, search_term)
            #COMMENT_PROMPT_STR = COMMENT_PROMPT_STR.format(subreddit, search_term)
            posts = post_df['title'].tolist()
            #comments = comment_df['body'].tolist()
            if len(posts) < 3:
                print(f'Not enough posts found for {subreddit} {search_term} {sort} {time_filter}')
                write_response('No response found', subreddit, search_term, sort, time_filter)
                continue
            posts = '\n====================\n'.join(posts)
            #comments = '\n====================\n'.join(comments)
            PROMPT_STR += posts
            #COMMENT_PROMPT_STR += comments
            try:
                bot = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages = [
                    {"role": "user", "content": f"{PROMPT_STR}"}
                    ],

                )
                response = bot.choices[0].message.content
                write_response(response, subreddit, search_term, sort, time_filter)
            except Exception:
                write_response('No response found', subreddit, search_term, sort, time_filter)
                print(f'No response found for {subreddit} {search_term} {sort} {time_filter}')
                traceback.print_exc()
                continue


def de_header(text):
    return re.sub(r"={50,}\n\nSubreddit:.*?\n\nSort:.*?\n\n={50,}\n\n", "", text)

# Create a wordcloud from combined.txt
def create_wordcloud(target_dir, subreddit, target_term):
    with open(f'{target_dir}/combined.txt', 'r') as f:
        text = f.read()
        text = de_header(text)
        wordcloud = WordCloud().generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f'Wordcloud for Community : r/{subreddit} Term: {target_term}')
        plt.show()


def create_graph(text):
    tokens = word_tokenize(text.lower())
    text = de_header(text)
    stop_words = set(stopwords.words('english'))
    additional_stop_words = ['may', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.']

    for word in additional_stop_words:
        stop_words.add(word)
    punctuation = set(['.', ',', ':', ';', '!', '?', '-', '(', ')', '[', ']', '{', '}'])
    filtered_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
    finder = BigramCollocationFinder.from_words(filtered_tokens)
    finder.apply_word_filter(lambda word: len(word) < 3)
    bigram_measures = BigramAssocMeasures()
    bigrams = finder.nbest(bigram_measures.raw_freq, 10)
    counter = Counter(filtered_tokens + bigrams)
    graph = nx.Graph()

    counter = dict(counter.most_common(10))
    for word, freq in counter.items():
        graph.add_node(word, weight=freq)
        for other_word, other_freq in counter.items():
            if word != other_word:
                weight = (freq + other_freq) / 2.0
                if len(graph.edges()) > 3:
                    continue
                if (word, other_word) not in graph.edges():
                    graph.add_edge(word, other_word, weight=weight)

    pos = nx.spring_layout(graph, k=0.5)
    nx.draw_networkx_nodes(graph, pos, node_size=[freq * 100 for freq in counter.values()], alpha=0.5)
    nx.draw_networkx_edges(graph, pos, width=[weight for weight in nx.get_edge_attributes(graph, 'weight').values()], alpha=0.3, edge_color='gray')
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='Arial', font_weight='bold')

    edge_labels = nx.get_edge_attributes(graph, 'weight')
    edge_labels = {k: round(v, 1) for k, v in edge_labels.items()}  # Round the edge weights to 1 decimal place
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

    plt.axis('off')
    plt.show()

def create_bar_chart(text):
    text = de_header(text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    additional_stop_words = ['may', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.']

    for word in additional_stop_words:
        stop_words.add(word)

    punctuation = set(['.', ',', ':', ';', '!', '?', '-', '(', ')', '[', ']', '{', '}'])
    filtered_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
    finder = BigramCollocationFinder.from_words(filtered_tokens)
    finder.apply_word_filter(lambda word: len(word) < 3)
    bigram_measures = BigramAssocMeasures()
    bigrams = finder.nbest(bigram_measures.raw_freq, 10)
    counter = Counter(filtered_tokens + bigrams)
    
    counter = dict(counter.most_common(10))
    words = list(counter.keys())
    frequencies = list(counter.values())

    y_pos = np.arange(len(words))

    plt.bar(y_pos, frequencies, align='center', alpha=0.5)
    plt.xticks(y_pos, words, rotation=45, ha='right')
    plt.ylabel('Frequency')
    plt.title('Most Common Words and Bigrams')

    plt.show()


def parse_and_format(text):
    lines = text.split("\n")
    formatted_text = []
    
    for line in lines:
        line = line.strip()
        #if line.startswith("Subreddit:") or line.startswith("Sort:") or line.startswith("Time Filter:"):
            #formatted_text.append(line)
        if line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or line.startswith("4.") or line.startswith("5.") or line.startswith("6.") or line.startswith("7.") or line.startswith("8.") or line.startswith("9.") or line.startswith("10."):
            formatted_text.append(line.split(".")[1].strip())
    
    return "\n".join(formatted_text)

def cross_combine_text_files(target_dirs, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(f'{out_dir}/combined.txt', 'w') as f:
        f.write('')
    for target_dir in target_dirs:
        search_term = target_dir.split('/')[-1]
        subreddit = target_dir.split('/')[-2]
        if not os.path.exists(f'{target_dir}/relevance_year_ideas.txt'):
            print(f'{target_dir}/relevance_year_ideas.txt does not exist')
            print(f'RUNNING QUERY FOR {search_term} on {subreddit}. NO IDEAS')
            run_query(subreddit, search_term)
        
        print(f'Processing files in {target_dir}...')
        # CLEAR FILE
        '''
        with open(f'{out_dir}/combined.txt', 'w') as f:
            f.write('')
        '''
        for text_file in os.listdir(target_dir):
            if text_file.endswith(".txt"):
                if 'comment_ideas' in text_file:
                    continue
                
                with open(f'{target_dir}/{text_file}', 'r') as f:
                    text = f.read()
                print(f'APPENDING TO {out_dir}/combined.txt')
                with open(f'{out_dir}/combined.txt', 'a') as f:
                    f.write(f'{text}\n\n')

def get_word_frequencies(text):
    tokens = word_tokenize(text.lower())
    text = de_header(text)
    stop_words = set(stopwords.words('english'))
    additional_stop_words = ['may', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.']

    for word in additional_stop_words:
        stop_words.add(word)
    punctuation = set(['.', ',', ':', ';', '!', '?', '-', '(', ')', '[', ']', '{', '}'])
    filtered_tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
    finder = BigramCollocationFinder.from_words(filtered_tokens)
    finder.apply_word_filter(lambda word: len(word) < 3)
    bigram_measures = BigramAssocMeasures()
    bigrams = finder.nbest(bigram_measures.raw_freq, 10)
    counter = Counter(filtered_tokens + bigrams)
    
    counter = dict(counter.most_common(10))
    
    return counter


def create_heatmap(df):
    plt.figure(figsize=(12, 6))
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt='g', linewidths=0.5)
    plt.title('Word Frequencies in Different Subreddits')
    plt.xlabel('Common Words')
    plt.ylabel('Subreddits')
    plt.show()
'''

create_heatmap(df)
# Step 1: Get the intersection of sets of words for each subreddit
common_words = set(df.columns)
for subreddit in df.index:
    subreddit_words = set(df.loc[subreddit].dropna().index)
    common_words = common_words.intersection(subreddit_words)

# Step 2: Create a new DataFrame with only the common words
common_words_df = df[list(common_words)]

# Step 3: Visualize the common word frequencies
ax = common_words_df.plot.bar(figsize=(12, 8))
ax.set_ylabel('Frequency')
ax.set_title('Common Word Frequencies Across Subreddits')
plt.xticks(rotation=45)
plt.show()
'''

# Main script
if __name__ == "__main__":

    with open("openai_api_key.txt", "r") as key_file:
        openai_api_key = key_file.readline().strip()

    openai.api_key = openai_api_key
    target_dirs = []
    terms = ['idea validation', 'market research']
    subreddits = ['startups']
    dir_name = 'study_startups_research'
    #lowercase all term
    terms = [term.lower() for term in terms]
    subreddits = [subreddit.lower() for subreddit in subreddits]
    for subreddit in subreddits:
        for term in terms:
            target_dir = f'./data/reddit_insights/{subreddit}/{term}'
            target_dirs.append(target_dir)
    #come up with directory title
    
    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            run_query(target_dir.split('/')[-2], target_dir.split('/')[-1])
        combine_text_files(target_dir)
    cross_combine_text_files(target_dirs, out_dir=f'./data/reddit_insights/{dir_name}')
    target_dir = f'./data/reddit_insights/{dir_name}'
    create_wordcloud(target_dir, subreddit, target_term='combined')
    with open(f'{target_dir}/combined.txt', 'r') as f:
        text = f.read()
    create_bar_chart(text)
    word_freq_dicts = []
    subreddit_word_freqs = {}
    #insights_str = Nonetarget_dir
    #get_insights(dir_name)
    with open(f'{target_dir}/combined.txt', 'r') as f:
        insights_str = f.read()
    
    #Get youtube queries that people in this space might search for, get the youtubers associated with those queries, and then scrape the videos associated with those youtubers to see which videos get the most engagement
    target_str = 'querie'
    insights_str = parse_and_format(insights_str)
    #print(f'INSIGHTS STR: \n\n\n{insights_str}')
    prompt_intro = f'Act as a reccommender system. Come up with general {target_str}s that reddit users would search for on youtube based on these sentiment/problem bullet points. I don\'t want {target_str}s for every bullet point, just the ones that stick out to you! :) For example How to become a tiktok influencer, How to budget, etc. Try not to let the queries exceed 6 words'
    posts_prompt = f'Here are the points:\n\n====================\n{insights_str}\n===================='
    prompt_str = f'{prompt_intro}\n\n{posts_prompt}'
    prompt_str = prompt_str[0:1500]


    print(prompt_str)
    

    yt_prompt = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages = [
        {"role": "user", "content": f"{prompt_str}"}
        ],

    )
    #print(yt_prompt.choices)
    response = yt_prompt.choices[0].message.content
    #print(f'YOUTUBE RESPONSE: {response}')
    
    #start youtube scrape
    write_youtube_response(response, terms, subreddits, target_dir)


    subreddit = target_dir.split('/')[-2]  # Extract the subreddit name from the target_dir string
    word_frequencies = get_word_frequencies(de_header(text))
    subreddit_word_freqs[subreddit] = word_frequencies

    df = pd.DataFrame(subreddit_word_freqs).transpose()
    #print(df)
