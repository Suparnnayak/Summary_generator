import streamlit as st
import pandas as pd
import json
import io
from src.analyzer import analyze_csv
from src.summarizer import generate_summary

st.set_page_config(
    page_title="Event Review Summarizer",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Event Review Summarizer")
st.write("Upload your event review CSV file and get an AI-generated summary + sentiment breakdown!")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Load and show data
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Preview of Uploaded Data:")
        st.dataframe(df.head())

        # Process file
        with st.spinner("🔍 Analyzing reviews and generating summary..."):
            # Save temporarily to a CSV path
            temp_path = "temp_uploaded.csv"
            df.to_csv(temp_path, index=False)

            # Run analysis pipeline
            report = analyze_csv(temp_path)
            summary = generate_summary(report)

        st.success("✅ Analysis complete!")

        # Show summary text
        st.subheader("🧾 Event Summary:")
        if isinstance(summary, dict) and "summary" in summary:
            st.write(summary["summary"])
        else:
            st.write(summary)

        # Show sentiment data (if available)
        if "sentiment_summary" in report:
            st.subheader("📊 Sentiment Summary:")
            sentiment = report["sentiment_summary"]
            st.json(sentiment)

            # Optional: quick display
            st.markdown(f"**Overall Sentiment:** {sentiment.get('overall_sentiment', 'N/A').capitalize()}")
            st.markdown(
                f"**Positive:** {sentiment.get('positive', 0)} | "
                f"**Neutral:** {sentiment.get('neutral', 0)} | "
                f"**Negative:** {sentiment.get('negative', 0)}"
            )

        # Save and offer download
        final = {"report": report, "summary": summary}
        json_str = json.dumps(final, indent=2)
        st.download_button(
            label="💾 Download JSON Report",
            data=json_str,
            file_name="event_summary_report.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("👆 Upload a CSV file to get started.")
