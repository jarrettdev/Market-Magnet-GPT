#%%
import pandas as pd
import sqlite3
import os
import sys
import subprocess
import traceback

#%%
dir_path = sys.argv[1]
dir_path = 'youtube_data/' + dir_path
#create path if it doesn't exist
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


channels = []
with open('channels.txt', 'r') as f:
    channels = list(set(f.read().splitlines()))
for channel in channels:
    try:
        channel_name = channel
        channel_name = channel_name.replace("\"", '').replace(" ", "_")
        channel_path = os.path.join(dir_path, 'channels')
        #create path if it doesn't exist
        if not os.path.exists(channel_path):
            os.makedirs(channel_path)
        #channel_name = channel_name.replace("\"", '').replace(" ", "_")
        query_name = channel_name.replace("\"", '').replace(" ", "_")
        print(channel_name)
        try:
            subprocess.run(["python3", 'collector.py', '-c', f'{query_name}'], timeout=30)
        except subprocess.TimeoutExpired:
            print("Process timed out, took longer than 30 seconds")
        #%%
        cnx = sqlite3.connect('all_data.sqlite')

        df = pd.read_sql_query("SELECT * FROM basic_video_data", cnx)
        #mk dir
        if not os.path.exists(f'{dir_path}/channels/' + channel_name):
            os.mkdir(f'{dir_path}/channels/' + channel_name)
        #move all_data.sqlite to channel_name
        os.rename('all_data.sqlite', f'{dir_path}/channels/' + channel_name + '/' + 'all_data.sqlite')

        #%%
        #drop rows with na
        df = df[~ df['video_views'].astype(str).str.startswith('NaN')]
        #%%
        df_split_num = int(len(df) / 2)

        #%%
        df_split_num
        # %%
        df['video_views'] = df['video_views'].str.replace(',', '').str.replace(' views', '').astype(int)
        # %%
        df_sixnine = df[0:df_split_num].sort_values('video_views')
        # %%
        df_sixnine
        # %%
        df_sixnine.to_csv(f'{channel_name}.csv', index=True)
        #move channel_name.csv to channel_name
        os.rename(f'{channel_name}.csv', f'{dir_path}/channels/' + channel_name + '/' + f'{channel_name}.csv')
        # %%
        top_twenty_views = df_sixnine[-int(len(df_sixnine)/5):]
        top_twenty_views.loc[top_twenty_views.index < len(df_sixnine)/5].to_csv(f'{channel_name}_top_fifth.csv', index=True)
        #move channel_name_top_fifth.csv to channel_name
        os.rename(f'{channel_name}_top_fifth.csv', f'{dir_path}/channels/' + channel_name + '/' + f'{channel_name}_top_fifth.csv')
    except Exception:
        print('not successful')
        traceback.print_exc()

with open('channels.txt', 'w') as f:
    f.write('')
# %%

# %%
