import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

TRAIN_SAMPLE_FRACTION = 0.00002
ENGINEERED_TRAIN_FILE = os.path.join(BASE_DIR, "engineered_features.csv")

REPORT_SAMPLE_SIZE = 1500
ENGINEERED_TEST_FILE = os.path.join(BASE_DIR, "engineered_features_test.csv")

LEXICAL_FEATURES = [
    'url_length',
    'hostname_length',
    'path_length',
    'query_length',
    'fragment_length',
    'count_dot',
    'count_hyphen',
    'count_underscore',
    'count_slash',
    'count_at',
    'count_equals',
    'count_percent',
    'count_digits',
    'count_letters',
    'count_special_chars',
    'has_ip_address',
    'has_http',
    'has_https',
]

WHOIS_FEATURES = [
    'domain_age_days',
    'domain_lifespan_days',
    'days_since_domain_update',
    'registrar_name',
]

SSL_FEATURES = [
    'cert_age_days',
    'cert_validity_days',
    'cert_issuer_cn',
    'cert_subject_cn',
    'ssl_protocol_version',
    'cert_has_valid_hostname',
]

ALL_FEATURE_COLUMNS = (
    LEXICAL_FEATURES +
    WHOIS_FEATURES +
    SSL_FEATURES
)

CATEGORICAL_FEATURES = [
    'registrar_name',
    'cert_issuer_cn',
    'cert_subject_cn',
    'ssl_protocol_version'
]

NUMERICAL_FEATURES = [
    col for col in ALL_FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES
]

ML_MODEL_RANDOM_STATE = 42
ML_TEST_SIZE = 0.2

DL_EPOCHS = 50
DL_BATCH_SIZE = 64
DL_LEARNING_RATE = 0.001