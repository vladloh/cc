import json
import re
import requests
import time
from multiprocessing import Pool, Value
from bs4 import BeautifulSoup

class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value

completed = Counter()
failed = Counter()

def parse_site(url, num_of_words=110):
    global completed, failed
    start_time = time.time()

    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, features='lxml')

        headline = soup.find('meta', property='og:title')['content']
        headline = ' '.join(re.findall(r'\w+', headline))

        article = soup.find('div', {'class': 'article-body'}).findAll('p')
        text = ' '.join([block.text for block in article])
        text = ' '.join(re.findall(r'\w+', text)[0:num_of_words])
    except (AttributeError, TypeError, requests.exceptions.SSLError, requests.exceptions.ConnectionError):
        print('Failed to parse link: {}'.format(url))
        failed.increment()
        headline = ''
        text = ''
    finally:
        completed.increment()
        print('Parsed #%d in %.3f seconds' % (completed.value, time.time() - start_time))

    return {'url': url, 'headline': headline, 'content': text}

with open('vk_cosmo_data.json', 'r') as f:
    file_parsed = json.loads(f.read())

all_urls = [entry['url'] for entry in file_parsed]

iteration_start = 37
step = 500

completed.increment(step * iteration_start)
for i in range(iteration_start, 161):
    print('Iteration: i = ', i)
    saved_time = time.time()

    urls = all_urls[i * step:(i + 1) * step]
    print('Number of tasks = ', len(urls))

    with Pool(200) as p:
        articles = p.map(parse_site, urls)

    print('Start writing entries to the file')
    with open('articles-%d.json' % i, 'w') as f:
        print(json.dumps(articles), file=f)
    print('Completed!')

    print('Failed entries = ', failed.value)
    print('Speed = {} seconds per article'.format((time.time() - saved_time) / step))

print('Total failed: ', failed.value)