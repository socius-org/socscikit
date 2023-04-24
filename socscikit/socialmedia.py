import conlp
import sys
import tweepy
import datetime
import time
import csv
import pickle 
import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from collections import deque

"""
df_list 

[[author], [tweet], [relational]]

- [author] = [author_id(int), name(str), description(str), followers(int)]
- [tweet] = [tweet_id(int), created_at(datetime), text(str), context([str]), attachment(str), includes([dict]), cashtags([str]), hashtags([str])]
- [relational] = [mentioned author ids([int]), root tweet(int), referred tweet([dict])]

(*Note:  
Root tweet means the original tweet that sparked the conversation. 
Replies to a given Tweet, as well as replies to those replies, are all included in the 
conversation stemming from the single original Tweet. That is, regardless of how many 
reply threads result, they will all share a common "root" node. 
Referred tweet is a list of Tweets such a Tweet refers to. It is a "parent" node)
) 

(*Note: 
includes = [{media1}, {media2}, ...] (see media_include and poll_include)
referred = [{type: , tweet_id: }, {type: , tweet_id}, ...]
)
"""
class plotlyWidget: 
    def __init__(self): 
        pass
    
    def dashboard(self): 
        fig = go.FigureWidget(
            make_subplots(rows=7, cols=3, 
                          specs = [[{'rowspan': 5, 'colspan': 2}, None, {'type': "table", 'rowspan': 7}],
                                   [None, None, None],
                                   [None, None, None],
                                   [None, None, None],
                                   [None, None, None],
                                   [{'rowspan': 2, 'colspan': 2}, None, None],
                                   [None, None, None]
                                  ], 
                          vertical_spacing=0.1,
                          horizontal_spacing=0.1,
                          shared_xaxes='columns'
                         )
        )
        
        NOW_HMstrp = datetime.datetime.strptime(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M"), "%Y-%m-%dT%H:%M")
        
        #avg sentiment
        fig.add_scatter(x=[NOW_HMstrp],
                       y=pd.Series([None]),
                       row=1, col=1,
                       name="sentiment (avg)", 
                        line=dict(color="darkorange"),
                        hovertemplate='<br>%{y}'
                       )
        
        #alphaVADER
        fig.add_scatter(x=[NOW_HMstrp],
                       y=pd.Series([None]),
                       row=1, col=1,
                       name="alphaVADER", 
                        line=dict(color="cornflowerblue"),
                        hovertemplate='<br>%{y}'
                       )
        #roBERTa_v1
        fig.add_scatter(x=[NOW_HMstrp],
                       y=pd.Series([None]),
                       row=1, col=1,
                       name="roBERTa_v1", 
                        line=dict(color="olive"),
                        hovertemplate='<br>%{y}'
                       )
        #roBERTa_v2
        fig.add_scatter(x=[NOW_HMstrp],
                       y=pd.Series([None]),
                       row=1, col=1,
                       name="roBERTa_v2", 
                        line=dict(color="olivedrab"),
                        hovertemplate='<br>%{y}'
                       )
        #XLM_roBERTa
        fig.add_scatter(x=[NOW_HMstrp],
                       y=pd.Series([None]),
                       row=1, col=1,
                       name="XLM_roBERTa", 
                        line=dict(color="deepskyblue"),
                        hovertemplate='<br>%{y}'
                       )
        #finBERT
        fig.add_scatter(x=[NOW_HMstrp],
                       y=pd.Series([None]),
                       row=1, col=1,
                       name="finBERT", 
                        line=dict(color="dodgerblue"),
                        hovertemplate='<br>%{y}'
                       )
        
        #Volume 
        fig.add_bar(x=[NOW_HMstrp], 
                    y=pd.Series([0]), 
        #             showlegend=False, 
                    row=6,col=1,
                    name='Tweet Volume',
                    marker=dict(color="lightgray"),
                    hovertemplate='<br>%{y}'
                   )
        
        #table 
        fig.add_table( 
                      header=dict(values=["Live Tweets"], font=dict(size=11), align='center'),
                      cells=dict(values=np.array([[None]]), font=dict(size=10), align='left'),
                      row=1, col=3
                     )

        fig.update_layout(title = "SENTIMENT DASHBOARD (UTC)", title_x = 0.465, 
                          annotations=[go.layout.Annotation(text="©SOCIALSCIENCE AI", 
                                                            xref='paper',
                                                            yref='paper',
                                                            x=1.225,
                                                            y=-0.1,
                                                            font=dict(size=5), showarrow=False)])
        
        fig.update_xaxes(tickformat='%H:%M', rangeslider= {'visible':False}, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=6, col=1)
        
        fig.update_yaxes(title_text="Average Sentiment", row=1, col=1, secondary_y=False)
        
        fig.update_annotations(font=dict(size=11))
        
        self.fig = fig 
        return self.fig

class tweepy_v2_StreamingClient(tweepy.StreamingClient):
    
    def __init__(self, 
                 bearer_token, 
                 hashtag_exception, 
                 senti, keyword, 
                 senti_analyze, 
                 keyword_extract, 
                 rate_limit, 
                 save_to_file, 
                 display_widget, 
                 dashboard): 
        
        tweepy.StreamingClient.__init__(self, bearer_token=bearer_token, wait_on_rate_limit=True)        
        self.client = tweepy.Client(bearer_token=bearer_token)
        
        self.hashtag_exception = hashtag_exception 
        self.senti_analyze = senti_analyze
        self.keyword_extract = keyword_extract 
        if self.senti_analyze is True: 
            self.senti = senti
        else: 
            pass
        if self.keyword_extract is True: 
            self.keyword = keyword
        else: 
            pass 
        self.preprocess = nlp.preprocess
        
        self.num_rate = 0
        self.rate_limit = rate_limit
        
        if isinstance(save_to_file, dict): 
            if 'name' in save_to_file.keys() and 'format' in save_to_file.keys(): 
                self.filename = save_to_file['name']
                self.fileformat = save_to_file['format']
            else: 
                if (('name' not in save_to_file.keys() and 'format' in save_to_file.keys()) or
                    ('format' not in save_to_file.keys() and 'name' in save_to_file.keys())): 
                    raise KeyError("Both 'name' and 'format' must be assigned to keys for the input 'save_to_file' (i.e {'name': name of the file to be saved, 'format': csv or json})")
                else: 
                    raise KeyError("Only 'name' and 'format' must be assigned to keys for the input 'save_to_file' (spelling and case sensitive)")
        else: 
            raise TypeError("Input 'save_to_file' format must be dictionary (i.e {'name': name of the file to be saved, 'format': csv or json})")
        
        if self.fileformat.lower() == 'csv': 
            with open('{}.csv'.format(self.filename), 'w', encoding='utf-8') as f: 
                writer = csv.writer(f)
                writer.writerow(["author_id", "name", "description", "followers", 
                                 "tweet_id", "created_at", "text", "context", 
                                 "attachment", "includes", "cashtags", "hashtags", "labels", 
                                 "mentions", "root_tweet", "referred", 
                                 "processed_text", "keyword_extract", "sentiments"
                                ])
        elif self.fileformat.lower() == 'json': 
            with open('{}.pkl'.format(self.filename), 'wb') as f:
                pickle.dump([], f)
        else: 
            raise ValueError("Fileformat 'csv' and 'json' are only supported")
        
        self.display_widget = display_widget
        
        if self.display_widget is True: 
            self.dashboard = dashboard
            display(self.dashboard)
        else: 
            pass 
        
        self.VOLUME_dict = {datetime.datetime.strptime(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M"), "%Y-%m-%dT%H:%M"): list()}
        self.SENTI_dict = {datetime.datetime.strptime(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M"), "%Y-%m-%dT%H:%M"): {'alphaVADER': list(),
                                                                                                                                                   'roBERTa_v1': list(),
                                                                                                                                                   'roBERTa_v2': list(), 
                                                                                                                                                   'XLM_roBERTa': list(), 
                                                                                                                                                   'finBERT': list(),
                                                                                                                                                   'total': list()
                                                                                                                                                  }}    
    def to_csv(self, df_ls):
            
        author_id, author_name, author_description, author_followers = df_ls[0][0], df_ls[0][1], df_ls[0][2], df_ls[0][3]
        tweet_id, created_at, text, context = df_ls[1][0], df_ls[1][1], df_ls[1][2], df_ls[1][3]
        attachment, includes, cashtags, hashtags, labels = df_ls[1][4], df_ls[1][5], df_ls[1][6], df_ls[1][7], df_ls[1][8]
        mentions, root_tweet, referred = df_ls[2][0], df_ls[2][1], df_ls[2][2]
        processed_text, keyword_extract, sentiments = df_ls[3][0], df_ls[3][1], df_ls[3][2]
        
        expanded_ls = [author_id, author_name, author_description, author_followers, 
                       tweet_id, created_at, text, context, 
                       attachment, includes, cashtags, hashtags, labels,
                       mentions, root_tweet, referred, 
                       processed_text, keyword_extract, sentiments]
        
        with open('{}.csv'.format(self.filename), 'a', encoding='utf-8') as f: 
            writer = csv.writer(f)
            writer.writerow(expanded_ls)
    
    def to_json(self, df_ls): 
        
        author_id, author_name, author_description, author_followers = df_ls[0][0], df_ls[0][1], df_ls[0][2], df_ls[0][3]
        tweet_id, created_at, text, context = df_ls[1][0], df_ls[1][1], df_ls[1][2], df_ls[1][3]
        attachment, includes, cashtags, hashtags, labels = df_ls[1][4], df_ls[1][5], df_ls[1][6], df_ls[1][7], df_ls[1][8]
        mentions, root_tweet, referred = df_ls[2][0], df_ls[2][1], df_ls[2][2]
        processed_text, keyword_extract, sentiments = df_ls[3][0], df_ls[3][1], df_ls[3][2]
        
        df_dict = {"author": {"id": author_id, 
                      "name": author_name, 
                      "description": author_description, 
                      "followers": author_followers
                     }, 
           "tweet": {"id": tweet_id, 
                     "created_at": str(created_at), 
                     "text": text, 
                     "context": context, 
                     "attachment": attachment, 
                     "includes": includes,
                     "cashtags": cashtags, 
                     "hashtags": hashtags, 
                     "labels": labels
                    }, 
           "relational": {"mentions": mentions, 
                          "root_tweet": root_tweet, 
                          "referred": referred
                         },
           "analysis": {"processed_text": processed_text,
                        "keyword": keyword_extract, 
                        "sentiment": sentiments
                       }
          }

        with open('{}.pkl'.format(self.filename), 'rb') as f:
            pseudoJson = pickle.load(f)
            pseudoJson.append(df_dict)
            pickle.dump(pseudoJson, open('{}.pkl'.format(self.filename), 'wb'))
            
    def on_errors(self): 
        pass
    
    def on_connection_error(self): 
        pass
    
    def on_exception(self): 
        pass
    
    def on_request_error(self): 
        pass 
    
    def on_connect(self):
        print("="*50)
        print("Twitter API Connnected")
        print("="*50)
        
    def media_include(self, include):
        if include['type'] == 'video':
            media_dict = {#'type': include['type'],
                          'media_key': include['media_key'], 
                          'preview_image_url': include['preview_image_url'], 
                          'duration_ms': include['duration_ms'], 
                          'public_metric': include['public_metrics']['view_count']
                         } 
        elif include['type'] == 'photo':
            media_dict = {#'type': include['type'],
                          'media_key': include['media_key'],
                          'url': include['url']
                         }
        
        elif include['type'] == 'animated_gif': 
            media_dict = {#'type': include['type'],
                          'media_key': include['media_key'],
                          'preview_image_url': include['preview_image_url'] 
                         }
        
        return media_dict
    
    
    def poll_include(self, include): 
        poll_dict = {'poll_id': include['id'],
                     'options': include['options'], 
                     'duration_minutes': include['duration_minutes'], 
                     'end_datetime': include['end_datetime'], 
                     'voting_status': include['voting_status']
                    }
        
        return poll_dict
    
    
    def on_response(self, response):
        df_list = [[], [], [], []]
        labels = []
        tweet = response.data
        #Author
        author = self.client.get_user(id=tweet['author_id'], 
                                 user_fields = ['description', 'name', 'profile_image_url', 'public_metrics', 'verified'])
        author_idx = author.data 
        
        author_id = author_idx.id
        author_name = author_idx.name
        author_description = author_idx.description
        author_followers = author_idx.public_metrics['followers_count']
        
        df_list[0].append(author_id)
        df_list[0].append(author_name)
        df_list[0].append(author_description)
        df_list[0].append(author_followers)
        
        #Tweet
        tweet_id = tweet['id']
        tweet_created = tweet['created_at']
#         tweet_created = tweet['created_at'].strftime("%Y-%m-%d %H:%M:%S")
        tweet_text = tweet['text']
        
        df_list[1].append(tweet_id)
        df_list[1].append(tweet_created)
        df_list[1].append(tweet_text)
        
        ca = len(tweet['context_annotations'])
        l_str = [tweet['context_annotations'][i]['entity']['name'] for i in range(ca)]
        tweet_context = list(set(l_str))
        labels += tweet_context
        df_list[1].append(tweet_context)
        
        if tweet['attachments'] is not None: 
            #Is it possible for a tweet to have media AND poll? 
            if 'media_keys' in tweet['attachments'].keys(): 
                if 'type=photo' in str(response.includes.values()): 
                    tweet_attachments = 'photo'
                    tweet_includes = [self.media_include(include) for include in response.includes['media']]
                elif 'type=video' in str(response.includes.values()):
                    tweet_attachments = 'video'
                    tweet_includes = [self.media_include(include) for include in response.includes['media']]
                elif 'type=animated_gif' in str(response.includes.values()):
                    tweet_attachments = 'gif'
                    tweet_includes = [self.media_include(include) for include in response.includes['media']]
            elif 'poll_ids' in tweet['attachments'].keys(): 
                tweet_attachments = 'poll'
                tweet_includes = [self.poll_include(include) for include in response.includes['polls']]
            else: 
                tweet_attachments = None
                tweet_includes = None
            
            df_list[1].append(tweet_attachments)
            df_list[1].append(tweet_includes)
        
        if 'cashtags' in tweet["entities"].keys(): 
            ct = len(tweet["entities"]['cashtags'])
            cashtags = [tweet["entities"]['cashtags'][i]['tag'] for i in range(ct)]
            labels += cashtags 
        if 'cashtags' not in tweet["entities"].keys():
            cashtags = None
        df_list[1].append(cashtags)
        if 'hashtags' in tweet["entities"].keys(): 
            ht = len(tweet["entities"]['hashtags'])
            hashtags = [tweet["entities"]['hashtags'][i]['tag'] for i in range(ht)]
            labels += hashtags
        if 'hashtags' not in tweet["entities"].keys(): 
            hashtags = None
        df_list[1].append(hashtags)
        df_list[1].append(labels)

        #Relational
        if 'mentions' in tweet["entities"].keys():
            m = len(tweet["entities"]['mentions'])
            mentions = [tweet["entities"]['mentions'][i]['id'] for i in range(m)]
        if 'mentions' not in tweet["entities"].keys():
            mentions = None
        df_list[2].append(mentions)
        
        root_tweet = tweet["conversation_id"]
        df_list[2].append(root_tweet)
        
        if tweet["referenced_tweets"] is not None: 
            reft = len(tweet["referenced_tweets"]) 
            referred = [{'type': tweet["referenced_tweets"][i]['type'], 
                         'ID': tweet["referenced_tweets"][i]['id']} for i in range(reft)]
        if tweet["referenced_tweets"] is None:
            referred = None
        df_list[2].append(referred)
        
        #Analysis 
        processed_text = self.preprocess.tweets(tweet_text, remove_hashtag=True, hashtag_exception=self.hashtag_exception)
        df_list[3].append(processed_text)
        
        if self.keyword_extract is True: 
            unigram_extract = self.keyword.keyBERT(processed_text, text_type='tweets', preprocess_exception=self.hashtag_exception)
            bigram_extract = self.keyword.keyBERT(processed_text, text_type='tweets', preprocess_exception=self.hashtag_exception, ngram=2)
            keyword_extract = {'unigram': unigram_extract, 'bigram': bigram_extract}
            df_list[3].append(keyword_extract)
        else: 
            df_list[3].append(None)
        if self.senti_analyze is True: 
            sentiments = self.senti.aggregate(processed_text, text_type='tweets')
            df_list[3].append(sentiments)
        else: 
            df_list[3].append(None)
        
        #Widget (10sec lag, due to Tweet Volume)
        if self.display_widget is True:  
            NOW = datetime.datetime.now(datetime.timezone.utc)
            NOW_HMstrft = datetime.datetime.strptime(NOW.strftime("%Y-%m-%dT%H:%M"), "%Y-%m-%dT%H:%M")
            print("REQUEST: {}".format(NOW))
        
            #AVOID TwitterAPI error "'end_time' must be a minimum of 10 seconds prior to the request time"
            if NOW.microsecond < 90000: 
                time.sleep(0.1)
            else: 
                pass
            
            #Sentiments 
            if NOW_HMstrft in self.SENTI_dict.keys(): 
                #Update 
                self.SENTI_dict[NOW_HMstrft]['alphaVADER'].append(sentiments['alphaVADER'])
                self.SENTI_dict[NOW_HMstrft]['roBERTa_v1'].append(sentiments['roBERTa_v1'])
                self.SENTI_dict[NOW_HMstrft]['roBERTa_v2'].append(sentiments['roBERTa_v2'])
                self.SENTI_dict[NOW_HMstrft]['XLM_roBERTa'].append(sentiments['XLM_roBERTa'])
                self.SENTI_dict[NOW_HMstrft]['finBERT'].append(sentiments['finBERT'])
                self.SENTI_dict[NOW_HMstrft]['total'].append(sentiments['total'])
                ##alphaVADER 
                dashboard_alphaVADER = np.array(self.dashboard.data[1].y)
                dashboard_alphaVADER[-1] = np.mean(self.SENTI_dict[NOW_HMstrft]['alphaVADER'])
                self.dashboard.data[1].y = dashboard_alphaVADER
                ##roBERTa_v1 
                dashboard_roBERTa_v1 = np.array(self.dashboard.data[2].y)
                dashboard_roBERTa_v1[-1] = np.mean(self.SENTI_dict[NOW_HMstrft]['roBERTa_v1'])
                self.dashboard.data[2].y = dashboard_roBERTa_v1
                ##roBERTa_v2
                dashboard_roBERTa_v2 = np.array(self.dashboard.data[3].y)
                dashboard_roBERTa_v2[-1] = np.mean(self.SENTI_dict[NOW_HMstrft]['roBERTa_v2'])
                self.dashboard.data[3].y = dashboard_roBERTa_v2
                ##XLM_roBERTa
                dashboard_XLM_roBERTa = np.array(self.dashboard.data[4].y)
                dashboard_XLM_roBERTa[-1] = np.mean(self.SENTI_dict[NOW_HMstrft]['XLM_roBERTa'])
                self.dashboard.data[4].y = dashboard_XLM_roBERTa
                ##finBERT
                dashboard_finBERT = np.array(self.dashboard.data[5].y)
                dashboard_finBERT[-1] = np.mean(self.SENTI_dict[NOW_HMstrft]['finBERT'])
                self.dashboard.data[5].y = dashboard_finBERT
                ##total 
                dashboard_total = np.array(self.dashboard.data[0].y)
                dashboard_total[-1] = np.mean(self.SENTI_dict[NOW_HMstrft]['total'])
                self.dashboard.data[0].y = dashboard_total
            
            else: 
                #Introduce new min 
                self.SENTI_dict.update({NOW_HMstrft: {'alphaVADER': [sentiments['alphaVADER']], 
                                                      'roBERTa_v1': [sentiments['roBERTa_v1']], 
                                                      'roBERTa_v2': [sentiments['roBERTa_v2']],
                                                      'XLM_roBERTa': [sentiments['XLM_roBERTa']], 
                                                      'finBERT': [sentiments['finBERT']],
                                                      'total': [sentiments['total']]}})
                
                ##alphaVADER 
                self.dashboard.data[1].x = np.append(self.dashboard.data[1].x, NOW_HMstrft)
                self.dashboard.data[1].y = np.append(self.dashboard.data[1].y, self.SENTI_dict[NOW_HMstrft]['alphaVADER'][0])
                ##roBERTa_v1 
                self.dashboard.data[2].x = np.append(self.dashboard.data[2].x, NOW_HMstrft)
                self.dashboard.data[2].y = np.append(self.dashboard.data[2].y, self.SENTI_dict[NOW_HMstrft]['roBERTa_v1'][0])
                ##roBERTa_v2
                self.dashboard.data[3].x = np.append(self.dashboard.data[3].x, NOW_HMstrft)
                self.dashboard.data[3].y = np.append(self.dashboard.data[3].y, self.SENTI_dict[NOW_HMstrft]['roBERTa_v2'][0])
                ##XLM_roBERTa
                self.dashboard.data[4].x = np.append(self.dashboard.data[4].x, NOW_HMstrft)
                self.dashboard.data[4].y = np.append(self.dashboard.data[4].y, self.SENTI_dict[NOW_HMstrft]['XLM_roBERTa'][0])
                ##finBERT
                self.dashboard.data[5].x = np.append(self.dashboard.data[5].x, NOW_HMstrft)
                self.dashboard.data[5].y = np.append(self.dashboard.data[5].y, self.SENTI_dict[NOW_HMstrft]['finBERT'][0])
                ##total 
                self.dashboard.data[0].x = np.append(self.dashboard.data[0].x, NOW_HMstrft)
                self.dashboard.data[0].y = np.append(self.dashboard.data[0].y, self.SENTI_dict[NOW_HMstrft]['total'][0])
                
                if len(self.dashboard.data[0].y) < 2: 
                    pass
                else: 
                    #Finalise prev min 
                    PREV_HMstrft = datetime.datetime.strptime((NOW_HMstrft - datetime.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M"), "%Y-%m-%dT%H:%M")

                    ##alphaVADER 
                    dashboard_alphaVADER = np.array(self.dashboard.data[1].y)
                    dashboard_alphaVADER[-2] = np.mean(self.SENTI_dict[PREV_HMstrft]['alphaVADER'])
                    self.dashboard.data[1].y = dashboard_alphaVADER
                    ##roBERTa_v1 
                    dashboard_roBERTa_v1 = np.array(self.dashboard.data[2].y)
                    dashboard_roBERTa_v1[-2] = np.mean(self.SENTI_dict[PREV_HMstrft]['roBERTa_v1'])
                    self.dashboard.data[2].y = dashboard_roBERTa_v1
                    ##roBERTa_v2
                    dashboard_roBERTa_v2 = np.array(self.dashboard.data[3].y)
                    dashboard_roBERTa_v2[-2] = np.mean(self.SENTI_dict[PREV_HMstrft]['roBERTa_v2'])
                    self.dashboard.data[3].y = dashboard_roBERTa_v2
                    ##XLM_roBERTa
                    dashboard_XLM_roBERTa = np.array(self.dashboard.data[4].y)
                    dashboard_XLM_roBERTa[-2] = np.mean(self.SENTI_dict[PREV_HMstrft]['XLM_roBERTa'])
                    self.dashboard.data[4].y = dashboard_XLM_roBERTa
                    ##finBERT
                    dashboard_finBERT = np.array(self.dashboard.data[5].y)
                    dashboard_finBERT[-2] = np.mean(self.SENTI_dict[PREV_HMstrft]['finBERT'])
                    self.dashboard.data[5].y = dashboard_finBERT
                    ##total 
                    dashboard_total = np.array(self.dashboard.data[0].y)
                    dashboard_total[-2] = np.mean(self.SENTI_dict[PREV_HMstrft]['total'])
                    self.dashboard.data[0].y = dashboard_total
            
            query = 'bitcoin (context:166.1301195966125494272 OR context:174.1007360414114435072) -nft -"follow me" -follow -"visit us" -like -"like and retweet" -"like and rt" -tag -giveaway -"giving away" -free -"comment your wallet" -"comment your BTC address" -"comment your ETH address" -telegram -"give $" -"win $" -"give USD" -"win USD" -airdrop -"air-drop" -ico -"play at" -"join me" -"make money online" -"making money online" -"zero experience needed" -"with the help" lang:en'
            #Volume 
            if 0 <= NOW.second and NOW.second <= 10: 
                start = datetime.datetime.strptime((NOW - (datetime.timedelta(minutes=1)+datetime.timedelta(seconds=NOW.second))).strftime("%Y-%m-%dT%H:%M:%S"), "%Y-%m-%dT%H:%M:%S")
                end = datetime.datetime.strptime((NOW - datetime.timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%S"), "%Y-%m-%dT%H:%M:%S")
                current_VOLUME = self.client.get_recent_tweets_count(query, start_time=start, end_time=end).data[0]['tweet_count']
                self.past_VOLUME = current_VOLUME 
            elif 10 < NOW.second and NOW.second <= 59: 
                start = datetime.datetime.strptime((NOW - datetime.timedelta(seconds=NOW.second)).strftime("%Y-%m-%dT%H:%M:%S"), "%Y-%m-%dT%H:%M:%S")
                end = datetime.datetime.strptime((NOW - datetime.timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%S"), "%Y-%m-%dT%H:%M:%S")
                current_VOLUME = self.client.get_recent_tweets_count(query, start_time=start, end_time=end).data[0]['tweet_count']
                self.past_VOLUME = current_VOLUME
            
#             if start in self.VOLUME_dict.keys(): 
#                 #Update
#                 self.VOLUME_dict[NOW_HMstrft].append(self.past_VOLUME)
#                 self.VOLUME_dict[NOW_HMstrft] = sorted(list(set(self.VOLUME_dict[NOW_HMstrft])))
#                 dashboard_VOLUME = np.array(self.dashboard.data[6].y)
#                 dashboard_VOLUME[-1] = self.VOLUME_dict[NOW_HMstrft][-1]
#                 self.dashboard.data[6].y = dashboard_VOLUME
#             else: 
#                 #Introduce new min 
#                 self.VOLUME_dict.update({NOW_HMstrft: [self.past_VOLUME]})
#                 self.dashboard.data[6].x = np.append(self.dashboard.data[6].x, NOW_HMstrft)
#                 self.dashboard.data[6].y = np.append(self.dashboard.data[6].y, self.VOLUME_dict[NOW_HMstrft][0])

#                 if len(self.dashboard.data[6].y) < 2:
#                     pass
#                 else: 
#                     #Finalise prev min 
#                     PREV_HMstrft = datetime.datetime.strptime((NOW_HMstrft - datetime.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M"), "%Y-%m-%dT%H:%M")
#                     dashboard_VOLUME = np.array(self.dashboard.data[6].y)
#                     dashboard_VOLUME[-2] = self.VOLUME_dict[PREV_HMstrft][-1]
#                     self.dashboard.data[6].y = dashboard_VOLUME
            
            if NOW_HMstrft == start: 
                if start in self.VOLUME_dict.keys(): 
                    #Update
                    self.VOLUME_dict[NOW_HMstrft].append(self.past_VOLUME)
                    self.VOLUME_dict[NOW_HMstrft] = sorted(list(set(self.VOLUME_dict[NOW_HMstrft])))
                    dashboard_VOLUME = np.array(self.dashboard.data[6].y)
                    dashboard_VOLUME[-1] = self.VOLUME_dict[NOW_HMstrft][-1]
                    self.dashboard.data[6].y = dashboard_VOLUME
                else: 
                    #Introduce new min 
                    self.VOLUME_dict.update({NOW_HMstrft: [self.past_VOLUME]})
                    self.dashboard.data[6].x = np.append(self.dashboard.data[6].x, NOW_HMstrft)
                    self.dashboard.data[6].y = np.append(self.dashboard.data[6].y, self.VOLUME_dict[NOW_HMstrft][0])

                    if len(self.dashboard.data[6].y) < 2:
                        pass
                    else: 
                        #Finalise prev min 
                        PREV_HMstrft = datetime.datetime.strptime((NOW_HMstrft - datetime.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M"), "%Y-%m-%dT%H:%M")
                        dashboard_VOLUME = np.array(self.dashboard.data[6].y)
                        dashboard_VOLUME[-2] = self.VOLUME_dict[PREV_HMstrft][-1]
                        self.dashboard.data[6].y = dashboard_VOLUME
                        
            elif NOW_HMstrft != start:
                PREV_HMstrft = datetime.datetime.strptime((NOW_HMstrft - datetime.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M"), "%Y-%m-%dT%H:%M")
                #Update prev min (10 sec lag)
                self.VOLUME_dict[PREV_HMstrft].append(self.past_VOLUME)
                self.VOLUME_dict[PREV_HMstrft] = sorted(list(set(self.VOLUME_dict[PREV_HMstrft])))
                dashboard_VOLUME = np.array(self.dashboard.data[6].y)
                dashboard_VOLUME[-1] = self.VOLUME_dict[PREV_HMstrft][-1]
                self.dashboard.data[6].y = dashboard_VOLUME
            
            print("volume: {}".format(self.dashboard.data[6].y))
            print("total: {}".format(self.dashboard.data[0].y))
            print("aVADER: {}".format(self.dashboard.data[1].y))
            print("roBERTa_v1: {}".format(self.dashboard.data[2].y))
            print("roBERTa_v2: {}".format(self.dashboard.data[3].y))
            print("XLM_roBERTa: {}".format(self.dashboard.data[4].y))
            print("finBERT: {}".format(self.dashboard.data[5].y))
            print("X-AXIS: ", self.dashboard.data[0].x, "/", self.dashboard.data[6].x)
            
            #Table (LIVE Tweets)
            cell_values = self.dashboard.data[7].cells.values.tolist()
            cell_values[0].insert(0, processed_text)
            self.dashboard.data[7].cells.values = np.array(cell_values)
        
        else: 
            pass 
        
        #Rate Limit
        self.num_rate += 1 
        
        if self.rate_limit is None: 
            print("collecting tweet_{}".format(tweet_id))
            if self.fileformat.lower() == 'csv':
                return self.to_csv(df_list)
            elif self.fileformat.lower() == 'json':
                return self.to_json(df_list)
        else: 
            if self.num_rate <= self.rate_limit: 
                print("collecting tweet_{}".format(tweet_id))
                if self.fileformat.lower() == 'csv': 
                    return self.to_csv(df_list)
                elif self.fileformat.lower() == 'json': 
                    return self.to_json(df_list)
            else: 
                print("Disconnecting Twitter API")
                sys.exit("{} successfully collected".format(self.rate_limit))

class twitter: 
    def __init__(self, 
                 bearer_token, 
                 hashtag_exception=None, 
                 senti_analyze=True, 
                 keyword_extract=True, 
                 display_widget=False): 
        
        self.bearer_token = bearer_token 
        self.hashtag_exception = hashtag_exception
        self.senti_analyze = senti_analyze
        self.keyword_extract = keyword_extract
        self.display_widget = display_widget 

        if self.senti_analyze is True: 
            self.senti = nlp.sentiment(load_models='all')            
        else: 
            self.senti = None
        if self.keyword_extract is True: 
            self.keyword = nlp.keyword_extraction(load_models='keyBERT')
        else: 
            self.keyword = None 
        
        if self.display_widget is True: 
            self.widget = plotlyWidget()
            self.dashboard = self.widget.dashboard()
        else: 
            self.dashboard = None 
    
    def stream(self, 
               query, 
               rate_limit,
               save_to_file={'name': 'test', 'format': 'csv'}, 
               sample=False): 
        
        streamer = tweepy_v2_StreamingClient(bearer_token=self.bearer_token, 
                                             hashtag_exception=self.hashtag_exception,
                                             senti_analyze=self.senti_analyze,
                                             keyword_extract=self.keyword_extract,
                                             senti=self.senti, 
                                             keyword=self.keyword, 
                                             rate_limit=rate_limit, 
                                             save_to_file=save_to_file,
                                            display_widget=self.display_widget, 
                                            dashboard=self.dashboard)
        
        streamer.add_rules(tweepy.StreamRule(query))
        
        if sample is False: 
            streamer.filter(expansions=["attachments.poll_ids", "attachments.media_keys"],
                           tweet_fields=["author_id","conversation_id","context_annotations","created_at","attachments","entities","possibly_sensitive","public_metrics","referenced_tweets"],
                           media_fields=["duration_ms", "preview_image_url", "type", "url", "alt_text", "public_metrics"],
                           poll_fields=["duration_minutes", "end_datetime", "options", "voting_status"])
        else: 
            streamer.sample(expansions=["attachments.poll_ids", "attachments.media_keys"],
                           tweet_fields=["author_id","conversation_id","context_annotations","created_at","attachments","entities","possibly_sensitive","public_metrics","referenced_tweets"],
                           media_fields=["duration_ms", "preview_image_url", "type", "url", "alt_text", "public_metrics"],
                           poll_fields=["duration_minutes", "end_datetime", "options", "voting_status"])
