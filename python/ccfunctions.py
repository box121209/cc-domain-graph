#import boto
#from boto.s3.key import Key
#from gzipstream import GzipStreamFile
#from pyspark.sql.types import *
#import warc
#import ujson as json

from math import log
from collections import Counter
import urlparse
import re


def unpack(uri):
    """
    Takes as argument one URI from
    watlist = sc.textFile("s3://commoncrawl/crawl-data/CC-MAIN-2017-04/wat.paths.gz")
    or WARC or WET file, and outputs the file for iterating over records.
    """
    conn = boto.connect_s3(anon=True, host='s3.amazonaws.com')
    bucket = conn.get_bucket('commoncrawl')
    key_ = Key(bucket, uri)
    file_ = warc.WARCFile(fileobj=GzipStreamFile(key_))
    return file_

def extract_json(id_, iterator):
    """
    Iterates through WARC records of an unpacked file in a Spark job.
    Usage:
    json_rdd = files.mapPartitionsWithIndex(extract_json)
    """
    for uri in iterator:
        file = unpack(uri)
        for record in file:
            if record['Content-Type'] == 'application/json':
                try:
                    content = json.loads(record.payload.read())
                    yield content['Envelope']
                except:
                    yield None

def parse_links(record):
    """
    Takes WARC record and returns domain of target URI plus Counter for
    domains of out-linked pages -- if these exist.
    """
    try:
        page_url = record['WARC-Header-Metadata']['WARC-Target-URI']
        page_domain = urlparse.urlparse(page_url).netloc
        links = record['Payload-Metadata']['HTTP-Response-Metadata']['HTML-Metadata']['Links']
        out_links = Counter([urlparse.urlparse(url['url']).netloc for url in links])
        return (page_domain, out_links)
    except:
        return None
                    
def parse_urls(record):
    """
    Takes WARC record and outputs all triples (domain, path, was_crawled) from URIs, if these exist.
    It searches both target URI (was_crawled = 1) and out-links (was_crawled = 0). 
    """
    url_list = []
    try:
        page_url = record['WARC-Header-Metadata']['WARC-Target-URI']
        x = urlparse.urlparse(page_url)
        if len(x.path) > 1:
            url_list += [(x.netloc, x.path, 1)]
    except:
        pass
    try:    
        links = record['Payload-Metadata']['HTTP-Response-Metadata']['HTML-Metadata']['Links']
        for url in links:
            x = urlparse.urlparse(url['url'])
            if len(x.path) > 1:
                url_list += [(x.netloc, x.path, 0)]
    except:
        pass
        
    return url_list

def depth(uri):
    return uri[:-1].count('/')

def length(uri):
    return len(uri) - uri.count('/')


def hexify(c):
    """
    Temporary ASCII encoding for human readable hex with ' - ' as delimiter for detecting 
    non-Latin unicode.
    """
    try:
        s = c.encode("utf-8").encode("hex")
    except UnicodeDecodeError:
        s = 0
    n = len(s)
    if n <= 2: return s
    a = ' - '.join([s[i:i+2] for i in range(0,n,2)])
    return a[:-1]

def domain_string(domain, path_set):
    """
    Takes domain and concatenates with path URIs separated by newlines.
    """
    out = domain + '\n' + '\n'.join(sorted([x[0] for x in list(path_set)]))
    return out

def nonlatin_detector(dom):
    """
    Computes the excess nr bytes over nr characters in a domain string.
    """
    str = domain_string(dom[0], dom[1])
    N = len(str)
    hex = [c.encode('utf-8').encode('hex') for c in list(str)]
    return float(sum([len(h)/2 for h in hex]) - N)/N

def hx(i):
    """
    Normalised 2-char hex representation of 0-255
    """
    a = hex(i)[2:]
    if len(a)<2: a = ''.join(['0',a])
    return a

hexabet = [hx(x) for x in range(256)]

def string_features_v1(str):
    """
    Coarse first version of a feature vector for a string.
    A placeholder for stronger versions.
    """
    N = float(length(str))
    if N==0: return None
    a = depth(str)
    b = len(re.findall(r'\.', str))/N
    c = len(re.findall(r'-', str))/N
    d = len(re.findall(r'_', str))/N
    cap = len(re.findall(r'[A-Z]', str))/N
    num = len(re.findall(r'[0-9]', str))/N
    return [log(N), a, b, c, d, num, cap]

def string_byte_dist(str):
    bytes = [int(b.encode('utf-8').encode('hex'), 16) for b in str]
    out = dict([(x,0) for x in range(256)])
    ct = dict(Counter(bytes))
    for k in out.keys():
         if k in ct.keys():
            out[k] += ct[k]
    out = [v[1] for v in sorted(out.iteritems(), key=lambda (k,v): k)]
    out = [float(x)/sum(out) for x in out]
    return out

def string_features_v2(str):
    """
    Version 2: combine the hexal distribution with the previous string statistics.
    """
    return string_features_v1(str) + string_byte_dist(str)

"""
PLAN: evaluate features via a trained RNN for additional vector representation
"""
    
def domain_features(domain, path_set):
    """
    Takes domain + set of paths as output by parse_urls() and 
    applies extracts statistics of the signature string.
    """
    return string_features_v2(domain_string(domain, path_set))

def entropy(pred):
    """
    Entropy of a soft-max vector.
    """
    return sum([-p*math.log(p) for p in pred])

def sample(a, temperature=1.0):
    """
    Samples an index from a probability array;
    higher temperature raises the entropy and vice versa    
    """
    a = np.log(a) / temperature
    dist = np.exp(a) / np.sum(np.exp(a))
    choices = range(len(a)) 
    return np.random.choice(choices, p=dist)

def hx(i):
    """
    Normalised 2-char hex representation of 0-255
    """
    a = hex(i)[2:]
    if len(a)<2: a = ''.join(['0',a])
    return a





