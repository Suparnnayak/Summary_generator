"""
End-to-end entrypoint.
Usage:
    python src/inference.py --csv data/sample_event_reviews.csv
"""

import argparse
import json
from analyzer import analyze_csv
from summarizer import generate_summary


def run(csv_path, out_json='report.json'):
    """Run the full pipeline: analyze CSV → summarize → save JSON."""
    report = analyze_csv(csv_path)
    summary = generate_summary(report)
    final = {'report': report, 'summary': summary}
    with open(out_json, 'w') as f:
        json.dump(final, f, indent=2)
    print('Saved final report to', out_json)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run CSV Insight Summarizer pipeline.')
    p.add_argument('--csv', required=True, help='Path to the input CSV file.')
    p.add_argument('--out', default='report.json', help='Output JSON file path.')
    args = p.parse_args()

    run(args.csv, args.out)
