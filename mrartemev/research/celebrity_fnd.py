import json
import requests

def check_tag(link, filename = 'selebrities.txt'):
    href = 'https://www.instagram.com/p/' + link + '/'
    url = href + '?__a=1'
    response = json.loads(requests.get(url).text)
    users_json = response['graphql']['shortcode_media']['edge_media_to_tagged_user']['edges']
    users = [entry['node']['user']['username'] for entry in users_json]
    for i in users:
        with open(filename, 'a') as f:
            print('\'' + i + '\'', end = ', ', file = f)