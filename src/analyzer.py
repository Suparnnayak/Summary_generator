
import pandas as pd
import numpy as np
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN


TEXT_COL_THRESHOLD = 0.4 # fraction of columns that are text to consider dataset text-heavy




def is_float_col(s):
    try:
        float(s)
        return True
    except Exception:
        return False




def detect_columns(df: pd.DataFrame):
    """Return detected column types: 'text', 'numeric', 'latlon'"""
    col_types = {}
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(50)
        # simple lat/lon detection
        if any(re.search(r"lat|lon|latitude|longitude", c.lower()) for _ in [c]):
            col_types[c] = 'latlon'
            continue
        # numeric ratio
        numeric_count = sample.apply(lambda x: is_float_col(x)).sum()
        if numeric_count / max(1, len(sample)) > 0.8:
            col_types[c] = 'numeric'
        else:
            avg_len = sample.apply(lambda x: len(x.split())).mean() if len(sample) else 0
            if avg_len > 2:
                col_types[c] = 'text'
            else:
                col_types[c] = 'categorical'
    return col_types




def top_keywords(texts, k=10):
    if not texts:
        return []
    vec = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vec.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    top_idx = np.argsort(sums)[-k:][::-1]
    return [terms[i] for i in top_idx]




def analyze_text_column(series: pd.Series, max_samples=200):
    """Return simple statistics and keyword/sentiment placeholders for a text column"""
    s = series.dropna().astype(str)
    n = len(s)
    sample = s.sample(n=min(n, max_samples), random_state=42).tolist()
    avg_len = np.mean([len(x.split()) for x in sample]) if sample else 0
    keywords = top_keywords(sample, k=8)
    return {
        'n_reviews': n,
        'avg_length_words': float(avg_len),
        'sample_reviews': sample[:10],
    }

def analyze_numeric_column(series: pd.Series):
    s = pd.to_numeric(series, errors='coerce')
    return {
        'count': int(s.count()),
        'mean': float(s.mean()) if s.count() else None,
        'std': float(s.std()) if s.count() else None,
        'min': float(s.min()) if s.count() else None,
        'max': float(s.max()) if s.count() else None,
        'top_5_values': list(s.value_counts().head(5).index.astype(str))
    }

def analyze_gps(df, lat_col, lon_col):
    # simple clustering to find densest cluster
    coords = df[[lat_col, lon_col]].dropna().astype(float).values
    res = {'n_points': int(len(coords))}
    if len(coords) >= 3:
        try:
            cl = DBSCAN(eps=0.001, min_samples=3).fit(coords)
            labels = cl.labels_
            counts = Counter(labels[labels >= 0])
            if counts:
                top_label = counts.most_common(1)[0][0]
                cluster_coords = coords[labels == top_label]
                centroid = list(cluster_coords.mean(axis=0))
                res.update({'top_cluster_count': int(counts[top_label]), 'top_cluster_centroid': centroid})
        except Exception:
            pass
    return res




def analyze_csv(path: str):
    df = pd.read_csv(path)
    if df.shape[0] == 0:
        return {'error': 'empty csv'}
    col_types = detect_columns(df)
    report = {'n_rows': int(df.shape[0]), 'n_cols': int(df.shape[1]), 'columns': col_types, 'analysis': {}}


    # Text analysis
    text_cols = [c for c, t in col_types.items() if t == 'text']
    for c in text_cols:
        report['analysis'][c] = analyze_text_column(df[c])


    # Numeric analysis
    numeric_cols = [c for c, t in col_types.items() if t == 'numeric']
    for c in numeric_cols:
        report['analysis'][c] =analyze_numeric_column(df[c])


    # latlon
    lat_cols = [c for c, t in col_types.items() if t == 'latlon' or 'lat' in c.lower()]
    lon_cols = [c for c, t in col_types.items() if t == 'latlon' or 'lon' in c.lower()]
    if lat_cols and lon_cols:
        lat_col = lat_cols[0]
        lon_col = lon_cols[0]
        report['analysis']['_gps'] = analyze_gps(df, lat_col, lon_col)


    return report




if __name__ == '__main__':
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default='analysis.json')
    args = p.parse_args()
    r = analyze_csv(args.csv)
    with open(args.out, 'w') as f:
        json.dump(r, f, indent=2)
    print('Saved analysis to', args.out)