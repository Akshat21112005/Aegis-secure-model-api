import whois
import tldextract
import ssl
import socket
import pandas as pd
from datetime import datetime
from urllib.parse import urlparse
from OpenSSL import SSL
import re
import sys
from tqdm import tqdm

def get_lexical_features(url):
    features = {}
    try:
        if not (url.startswith('http://') or url.startswith('https://')):
             url = 'http://' + url
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname if parsed_url.hostname else ''
        path = parsed_url.path
        query = parsed_url.query
        fragment = parsed_url.fragment
    except Exception:
        hostname = ''
        path = ''
        query = ''
        fragment = ''
        url = ''

    features['url_length'] = len(url)
    features['hostname_length'] = len(hostname)
    features['path_length'] = len(path)
    features['query_length'] = len(query)
    features['fragment_length'] = len(fragment)
    features['count_dot'] = url.count('.')
    features['count_hyphen'] = url.count('-')
    features['count_underscore'] = url.count('_')
    features['count_slash'] = url.count('/')
    features['count_at'] = url.count('@')
    features['count_equals'] = url.count('=')
    features['count_percent'] = url.count('%')
    features['count_digits'] = sum(c.isdigit() for c in url)
    features['count_letters'] = sum(c.isalpha() for c in url)
    features['count_special_chars'] = len(re.findall(r'[^a-zA-Z0-9\s]', url))

    ip_regex = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    features['has_ip_address'] = 1 if re.match(ip_regex, hostname) else 0
    features['has_http'] = 1 if 'http:' in url else 0
    features['has_https'] = 1 if 'https:' in url else 0
    
    return features

def get_whois_features(registrable_domain, whois_cache):
    if registrable_domain in whois_cache:
        return whois_cache[registrable_domain]

    features = {
        'domain_age_days': -1,
        'domain_lifespan_days': -1,
        'days_since_domain_update': -1,
        'registrar_name': 'N/A',
    }

    try:
        w = whois.whois(registrable_domain)
        if not w.creation_date:
            whois_cache[registrable_domain] = features
            return features

        creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        expiration_date = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
        updated_date = w.updated_date[0] if isinstance(w.updated_date, list) else w.updated_date

        if creation_date:
            features['domain_age_days'] = (datetime.now() - creation_date).days
        
        if creation_date and expiration_date:
            features['domain_lifespan_days'] = (expiration_date - creation_date).days
            
        if updated_date:
            features['days_since_domain_update'] = (datetime.now() - updated_date).days

        if w.registrar:
            features['registrar_name'] = str(w.registrar).split(' ')[0].replace(',', '').replace('"', '')

    except Exception as e:
        pass

    whois_cache[registrable_domain] = features
    return features

def get_ssl_features(hostname, ssl_cache):
    if hostname in ssl_cache:
        return ssl_cache[hostname]

    features = {
        'cert_age_days': -1,
        'cert_validity_days': -1,
        'cert_issuer_cn': 'N/A',
        'cert_subject_cn': 'N/A',
        'ssl_protocol_version': 'N/A',
        'cert_has_valid_hostname': 0,
    }

    try:
        context = SSL.Context(SSL.SSLv23_METHOD)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        
        ssl_sock = SSL.Connection(context, sock)
        ssl_sock.set_tlsext_host_name(hostname.encode('utf-8'))
        ssl_sock.connect((hostname, 443))
        ssl_sock.do_handshake()
        
        cert = ssl_sock.get_peer_certificate()
        
        features['ssl_protocol_version'] = ssl_sock.get_protocol_version_name()
        not_before_str = cert.get_notBefore().decode('utf-8')
        not_before_date = datetime.strptime(not_before_str, '%Y%m%d%H%M%SZ')
        features['cert_age_days'] = (datetime.now() - not_before_date).days
        not_after_str = cert.get_notAfter().decode('utf-8')
        not_after_date = datetime.strptime(not_after_str, '%Y%m%d%H%M%SZ')
        features['cert_validity_days'] = (not_after_date - not_before_date).days
        issuer_components = dict(cert.get_issuer().get_components())
        subject_components = dict(cert.get_subject().get_components())
        
        features['cert_issuer_cn'] = issuer_components.get(b'CN', b'N/A').decode('utf-8')
        features['cert_subject_cn'] = subject_components.get(b'CN', b'N/A').decode('utf-8')
        if features['cert_subject_cn'] == hostname or f"*.{tldextract.extract(hostname).registered_domain}" == features['cert_subject_cn']:
            features['cert_has_valid_hostname'] = 1

        ssl_sock.close()
        sock.close()

    except SSL.Error as e:
        features['cert_issuer_cn'] = 'SSL_ERROR'
    except socket.timeout:
        features['cert_issuer_cn'] = 'TIMEOUT'
    except socket.error:
        features['cert_issuer_cn'] = 'CONN_FAILED'
    except Exception as e:
        features['cert_issuer_cn'] = 'SSL_UNKNOWN_ERROR'
        pass

    ssl_cache[hostname] = features
    return features

def process_row(row, whois_cache, ssl_cache):
    url = row['url']
    try:
        if not (url.startswith('http://') or url.startswith('https://')):
             url_for_parse = 'http://' + url
        else:
             url_for_parse = url
             
        parsed_url = urlparse(url_for_parse)
        hostname = parsed_url.hostname if parsed_url.hostname else ''
        
        ext = tldextract.extract(url_for_parse)
        registrable_domain = f"{ext.domain}.{ext.suffix}"
        
    except Exception:
        hostname = ''
        registrable_domain = ''
    lexical_data = get_lexical_features(url)
    
    whois_data = {}
    if registrable_domain:
        whois_data = get_whois_features(registrable_domain, whois_cache)
        
    ssl_data = {}
    if hostname:
        ssl_data = get_ssl_features(hostname, ssl_cache)
    all_features = {**lexical_data, **whois_data, **ssl_data}
    
    return pd.Series(all_features)


def extract_features_from_dataframe(df):
    if 'url' not in df.columns:
        raise ValueError("DataFrame must contain a 'url' column.")
    whois_cache = {}
    ssl_cache = {}
    
    print("Starting feature extraction... This may take a very long time.")
    try:
       
        tqdm.pandas(desc="Extracting features")
        feature_df = df.progress_apply(process_row, args=(whois_cache, ssl_cache), axis=1)
    except ImportError:
        print("tqdm not found. Running without progress bar. `pip install tqdm` to see progress.")
        feature_df = df.apply(process_row, args=(whois_cache, ssl_cache), axis=1)

    print("Feature extraction complete.")
    
    final_df = pd.concat([df, feature_df], axis=1)
    
    return final_df
