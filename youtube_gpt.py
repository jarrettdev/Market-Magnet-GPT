import openai
import os
import subprocess

# Configure OpenAI
openai.api_key = None
openai_api_key = None

with open("openai_api_key.txt", "r") as key_file:
        openai_api_key = key_file.readline().strip()

openai.api_key = openai_api_key

# Insights from the Reddit scraper (replace with the actual insights)
insights = None
with open('insights.txt', 'r') as f:
    insights = f.read()

def generate_youtube_search_queries(insights):
    prompt = f"""
    You are an AI assistant skilled in generating creative and accurate YouTube search queries based on insights from Reddit. Your task is to generate the best and most relevant search queries possible. Keep in mind that the main goal is to help the user discover valuable content related to their industry.

    Based on the following insights from Reddit:

    {insights}

    Generate creative and accurate YouTube search queries:
    """

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.8,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    queries = response.choices[0].text.strip().split('\n')
    return queries

def execute_youtube_scraper(query):
    if query[0].isdigit() and query[1] == '.': query = query.strip()[2:].strip()
    query = query.replace(' ', '+').replace('-','')
    command_string = f'xvfb-run -a node index.js "{query}"'
    #make sure the command string is file path safe
    command_string = command_string.replace('"', '').strip()
    #if query begins with 1. or 2., etc, remove it
    print(f'Running YouTube Creator Finder for query: {query}')
    process = None
    try:
        process = subprocess.run(command_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        print('Done running YouTube Creator Finder\n')
        #print(process.stdout)
        #print(process.stderr)
        return
    except subprocess.TimeoutExpired:
        print("Process timed out")
        print('Done running YouTube Creator Finder\n')
        #print(process.stdout)
        #print(process.stderr)
        return


if __name__ == "__main__":
    queries = generate_youtube_search_queries(insights)
    
    print("\033[96m\033[1m" + "\n🚀 Generated YouTube search queries: 🚀\n" + "\033[0m\033[0m")
    for query in queries:
        print(query)
        execute_youtube_scraper(query)
