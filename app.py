import streamlit as st
import pandas as pd
import time
import re
import matplotlib.pyplot as plt
from io import BytesIO
from collections import Counter

# Optional: wordcloud support
try:
    from wordcloud import WordCloud, STOPWORDS
    _wordcloud_available = True
except ImportError:
    _wordcloud_available = False

from sentiment_engine import (
    analyze_review_structured,
    generate_sentiment_summary,
    create_pdf_report,
    extract_top_negative_reviews_by_aspect,
    build_recommendations_for_aspect,
    clean_text_for_pdf,
    auto_detect_review_column,
    auto_detect_nps_column,
    detect_language_of_reviews,
    safe_read_csv,
    limit_large_df
)

TOP_N_ASPECTS = 10

st.set_page_config(layout="wide", page_title="Sentiment Insight BI Dashboard", page_icon="📊")

st.markdown("""
    <style>
    .big-title { font-size: 2.7rem; font-weight: 900; letter-spacing: .035rem; color: #023e8a; }
    .subtitle { font-size: 1.2rem; color:#2774ae; font-weight:600; }
    .footer { color:#29506d; font-size:1.04em; padding:2em 0 2em 0; text-align:center; }
    .stButton>button, .stDownloadButton>button {
        color: white !important; background: linear-gradient(90deg,#0096c7,#00b4d8);
        border: 0px; border-radius: 7px; font-weight: bold; font-size: 1.13em;
        margin-bottom: 2em; transition: box-shadow .12s;
        box-shadow: 0 2px 9px #7ae0ebaa;
    }
    .stChatMessage {
        background-color: #f0f8ff; border-radius: 10px; padding: 10px; margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style='padding:1.17em .7em 0.9em .7em;background: linear-gradient(90deg,#e0f7fd 15%, #bdeaff 85%);
        border-radius: 16px; border: 1.4px solid #ade8f4; margin-bottom:1.2em;'>
        <div style="font-size:1.23em; font-weight:800; color:#12537a; margin-bottom:.13em;">
            ⏺ ABSA Quick Start
        </div>
        <ul style="font-size:1.07em; color:#244d67; line-height:1.7; margin-bottom:0.9em;">
            <li>Upload reviews (CSV/XLSX)</li>
            <li>Click <b>Analyze Reviews</b></li>
            <li>Download Executive PDF, CSVs</li>
            <li>Chat with your data!</li>  </ul>
        <b>Made with</b> 🤭 <b>Streamlit, spaCy, NLTK</b><br>
        <span style="font-size:0.97em;">Contact: insights@org.com</span>
    </div>
    """, unsafe_allow_html=True)
    st.info("Your data is processed locally in your session and never uploaded or stored.")

st.markdown("""
    <div style="padding: 1.1em 1.2em 1.1em 1.25em; border-radius: 18px; margin-bottom: 1.2em;
                background: linear-gradient(90deg,#e0f7fd 60%, #a6e8ff 100%);
                border: 1.5px solid #90e0ef;">
        <span class='big-title'>Sentiment Insight BI Dashboard 📊</span>
        <br><span class='subtitle'>Executive Analytics Suite for Aspect-Based Sentiment + NPS Analysis</span>
        <ul style="padding-left:1.04em; padding-top:.73em;">
            <li><b>1.</b> Upload your review file (.csv or .xlsx)</li>
            <li><b>2.</b> Select the review and NPS columns (auto-selected if left blank)</li>
            <li><b>3.</b> Click <b>Analyze Reviews</b> 👇</li>
            <li><b>4.</b> Chat with your analysis results!</li> </ul>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload CSV or Excel file with reviews & NPS", type=["csv", "xlsx"])
df = None
df_error = None

def clear_analysis_state():
    for k in ["absa_results", "absa_summary", "top_neg_reviews_by_aspect", "pdf_bytes", "chat_suggestions", "messages"]:
        if k in st.session_state:
            del st.session_state[k]

if uploaded_file:
    if "current_file_name" not in st.session_state or uploaded_file.name != st.session_state["current_file_name"]:
        clear_analysis_state()
        st.session_state["current_file_name"] = uploaded_file.name
    with st.spinner("Loading file..."):
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = safe_read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            df = limit_large_df(df)
            if len(df) > 20000:
                st.warning("Large file detected: Processing may take longer for datasets over 20,000 reviews.")
        except Exception as e:
            df_error = f"Could not read uploaded file: {e}"
    if df is not None:
        try:
            auto_col = auto_detect_review_column(df)
        except Exception as e:
            auto_col = df.columns[0] if len(df.columns) > 0 else ""
            st.error(f"Could not auto-detect review column: {e}")
        try:
            auto_nps_col = auto_detect_nps_column(df)
        except Exception as e:
            auto_nps_col = ""
        review_col_selection = st.selectbox(
            "✨ Select review column",
            df.columns,
            index=list(df.columns).index(auto_col) if auto_col in df.columns else 0,
            help="Choose the column containing user opinions"
        )
        nps_col_selection = st.selectbox(
            "💠 Select NPS score column (0-10 scale, optional)",
            ["<AUTO-DETECT>"] + list(df.columns),
            index=1 if (auto_nps_col and auto_nps_col in df.columns) else 0,
            help="Choose the column (0-10) with NPS/score (or leave blank for auto-detect)"
        )
        if nps_col_selection == "<AUTO-DETECT>":
            nps_col_selection = auto_nps_col
    elif df_error:
        st.error(df_error)

def analyze_reviews(df: pd.DataFrame, review_col: str, nps_col: str):
    progress_area = st.empty()
    def streamlit_progress_callback(progress, msg):
        progress_area.text(msg)
    lang = detect_language_of_reviews(df, review_col)
    if lang != "en":
        st.warning(f"Detected language: {lang}. Only English is supported; results may not be accurate.")
    with st.spinner("Analyzing reviews..."):
        start_time = time.time()
        df_out = analyze_review_structured(df, review_col=review_col, nps_col=nps_col or None, progress_callback=streamlit_progress_callback)
        summary_df = generate_sentiment_summary(df_out)
        top_aspects = list(summary_df["Aspect"].head(TOP_N_ASPECTS))
        top_neg_reviews = extract_top_negative_reviews_by_aspect(df_out, top_aspects)
        pdf_bytes = create_pdf_report(df_out, summary_df)
        st.success(f"🏃‍♂️ Analysis completed in {time.time() - start_time:.1f} seconds.")
        progress_area.empty()
    return df_out, summary_df, top_neg_reviews, pdf_bytes

if uploaded_file and df is not None:
    if st.button("🧑‍🔬 Analyze Reviews"):
        try:
            df_out, summary_df, top_neg_reviews_by_aspect, pdf_bytes = analyze_reviews(df, review_col_selection, nps_col_selection)
            st.session_state["absa_results"] = df_out
            st.session_state["absa_summary"] = summary_df
            st.session_state["top_neg_reviews_by_aspect"] = top_neg_reviews_by_aspect
            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["chat_suggestions"] = None
            st.session_state["messages"] = None
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

    if "absa_results" in st.session_state and "absa_summary" in st.session_state:
        df_out = st.session_state["absa_results"]
        summary_df = st.session_state["absa_summary"]
        top_neg_reviews_by_aspect = st.session_state.get("top_neg_reviews_by_aspect", {})

        def show_eda_metrics(df, summary_df, review_col_selection):
            st.markdown("<h3>🔍 Data Summary & Review Analytics</h3>", unsafe_allow_html=True)
            colA, colB, colC, colD, colE = st.columns(5)
            main_col = review_col_selection if review_col_selection in df.columns else df.columns[0]
            length_series = df[main_col].astype(str).str.len()
            avg_length = int(length_series.mean())
            colA.metric("Total Reviews", f"{len(df):,}")
            colB.metric("Avg. Length (chars)", f"{avg_length:,}")
            colC.metric("Total Aspects Found", f"{summary_df['Aspect'].nunique()}")
            colD.metric("Top Aspect", summary_df.iloc[0]['Aspect'] if not summary_df.empty else "-")
            if "Avg NPS" in summary_df.columns:
                non_na = summary_df["Avg NPS"].replace("N/A", pd.NA).dropna()
                try:
                    colE.metric("Avg NPS per Aspect", f"{pd.to_numeric(non_na, errors='coerce').mean():.2f}" if not non_na.empty else "N/A")
                except Exception:
                    colE.metric("Avg NPS per Aspect", "N/A")

        def plot_review_length(df, main_col):
            st.markdown("**Review Length Distribution**")
            fig, ax = plt.subplots(figsize=(8, 3))
            length_series = df[main_col].astype(str).str.len()
            ax.hist(length_series, bins=25, color="#2493b4", alpha=0.65, edgecolor="black")
            ax.set_xlabel("Length of Review (chars)", fontsize=12)
            ax.set_ylabel("Number of Reviews", fontsize=12)
            ax.set_title("How long are the reviews?", fontsize=14, pad=10, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        def plot_wordcloud_and_topwords(df, main_col):
            if _wordcloud_available:
                try:
                    st.markdown("**Most Frequent Words in Reviews**")
                    text_blob = " ".join(str(txt) for txt in df[main_col].dropna() if isinstance(txt, str))
                    tokens = [
                        word.lower() for word in re.findall(r'\b\w+\b', text_blob)
                        if word.lower() not in STOPWORDS and len(word) > 2
                    ]
                    freq_counter = Counter(tokens)
                    top_words = freq_counter.most_common(10)
                    wc = WordCloud(
                        width=1200, height=350, background_color="white",
                        stopwords=STOPWORDS, max_words=100, colormap="PuBu"
                    ).generate(text_blob)
                    st.image(wc.to_array(), use_container_width=True)
                    st.caption("Top words sized by frequency")
                    if top_words:
                        words, counts_words = zip(*top_words)
                        figw, axw = plt.subplots(figsize=(8, 3))
                        axw.barh(words[::-1], counts_words[::-1], color="#4682b4")
                        for i, v in enumerate(counts_words[::-1]):
                            axw.text(
                                v - max(counts_words) * 0.03, i, str(v),
                                va='center', ha='right', color='white', fontsize=12, fontweight='bold'
                            )
                        axw.set_xlabel("Frequency", fontsize=13)
                        axw.set_title("Top 10 Words (excluding stopwords)", fontsize=14, pad=15, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(figw)
                        plt.close(figw)
                except Exception as e:
                    st.info(f"WordCloud not available: {e}")

        def plot_aspect_popularity(summary_df):
            st.markdown("**Most Discussed Aspects**")
            aspect_names = summary_df["Aspect"].head(15).tolist()
            mention_counts = summary_df["Total Mentions"].head(15).tolist()
            fig_bar, ax_bar = plt.subplots(figsize=(9, 4))
            bars = ax_bar.bar(range(len(aspect_names)), mention_counts, color="#6fa8dc", edgecolor="black", width=0.6)
            for bar in bars:
                yval = bar.get_height()
                ax_bar.annotate(
                    f'{int(yval)}',
                    xy=(bar.get_x() + bar.get_width() / 2, yval),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold'
                )
            ax_bar.set_xticks(range(len(aspect_names)))
            ax_bar.set_xticklabels(aspect_names, rotation=25, ha='right', fontsize=11)
            ax_bar.set_ylabel("Mentions", fontsize=12)
            ax_bar.set_title("Top Aspects by Number of Mentions", fontsize=14, pad=12, fontweight='bold')
            ax_bar.set_ylim(0, max(mention_counts) * 1.18 + 4)
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        def plot_sentiment_breakdown(summary_df):
            st.markdown("**Sentiment Breakdown for Top Aspects**")
            color_palette = ["#53b944", "#f7c948", "#ed2939"]
            for idx, (_, row) in enumerate(summary_df.head(3).iterrows()):
                labels = ["Positive", "Neutral", "Negative"]
                sizes = [row.get("Positive", 0), row.get("Neutral", 0), row.get("Negative", 0)]
                fig2, ax2 = plt.subplots(figsize=(7, 3))
                bars = ax2.bar(labels, sizes, color=color_palette, width=0.60, edgecolor='black')
                for bar in bars:
                    yval = bar.get_height()
                    ax2.annotate(f'{int(yval)}', xy=(bar.get_x() + bar.get_width()/2, yval),
                                 xytext=(0, 6), textcoords="offset points",
                                 ha='center', va='bottom', fontsize=12, fontweight='bold')
                ax2.set_ylabel("Mentions", fontsize=12)
                ax2.set_ylim(0, max(sizes) * 1.23 + 2)
                ax2.set_title(f"{row['Aspect']} ({int(row['Total Mentions'])} mentions)", fontsize=13, pad=10, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
                st.markdown("<br>", unsafe_allow_html=True)

        def plot_overall_sentiment(df_out):
            st.markdown("**Overall Sentiment Distribution (All Aspects Combined)**")
            overall_sentiments = pd.Series(df_out["Aspect_Sentiment"])
            value_counts = overall_sentiments.value_counts()
            fig3, ax3 = plt.subplots(figsize=(5, 3.5))
            color_palette = ["#53b944", "#f7c948", "#ed2939"]
            wedges, texts, autotexts = ax3.pie(
                value_counts.values,
                labels=value_counts.index,
                autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100. * sum(value_counts.values)))})",
                colors=color_palette,
                textprops={'fontsize': 11, 'weight': 'bold'}
            )
            ax3.set_title("Overall Sentiment Share", fontsize=13, pad=10, fontweight='bold')
            st.pyplot(fig3)
            plt.close(fig3)

        show_eda_metrics(df, summary_df, review_col_selection)
        main_col = review_col_selection if review_col_selection in df.columns else df.columns[0]
        plot_review_length(df, main_col)
        plot_wordcloud_and_topwords(df, main_col)
        plot_aspect_popularity(summary_df)
        plot_sentiment_breakdown(summary_df)
        plot_overall_sentiment(df_out)

        st.markdown("---")
        st.markdown("<h3>📄 Download Analysis Results</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([3,3,4])
        with col1:
            st.download_button(
                label="⬇️ CSV: Full Sentiment Data",
                data=df_out.to_csv(index=False).encode('utf-8'),
                file_name="absa_structured_results.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="⬇️ CSV: Aspect Sentiment Summary",
                data=summary_df.to_csv(index=False).encode('utf-8'),
                file_name="aspect_sentiment_summary.csv",
                mime="text/csv"
            )
        with col3:
            st.download_button(
                label="⬇️ Download PDF",
                data=st.session_state.get("pdf_bytes", b""),
                file_name="sentiment_bi_dashboard.pdf",
                mime="application/pdf",
                key="download_pdf_button"
            )
        st.caption("Download will activate after a successful analysis. Data is not shared or stored; all analysis is local in your browser session.")

        st.markdown("---")
        st.markdown("<h3>💬 Chat with Your Review Data</h3>", unsafe_allow_html=True)
        st.write("Ask questions or explore your analysis with one click:")

        def generate_dynamic_suggestions(summary_df):
            sugg = []
            top_aspects = summary_df["Aspect"].head(3).tolist()
            for a in top_aspects:
                sugg.append(f"Sentiment for {a}")
            for a in top_aspects:
                sugg.append(f"Recommendations for {a}")
            sugg.append("What are the main negative aspects?")
            return sugg

        if "chat_suggestions" not in st.session_state or not st.session_state["chat_suggestions"]:
            st.session_state["chat_suggestions"] = generate_dynamic_suggestions(summary_df)

        if "messages" not in st.session_state or st.session_state["messages"] is None:
            st.session_state.messages = [
                {"role": "assistant", "content": "👋 Hi! I can answer questions about top aspects, sentiments, themes, and recommendations. Pick a suggestion below or ask anything about your customer data."}
            ]

        button_cols = st.columns(len(st.session_state["chat_suggestions"]))
        suggestion_clicked = None
        for i, option in enumerate(st.session_state["chat_suggestions"]):
            if button_cols[i].button(option, key=f"chat_suggestion_{option}"):
                suggestion_clicked = option

        chat_input = st.chat_input("Type your question, or pick a suggestion above...")

        def chatbot_response(user_input: str, summary_df: pd.DataFrame, top_neg_reviews_by_aspect: dict) -> str:
            user_lower = user_input.lower()
            if "main negative aspects" in user_lower or "top problems" in user_lower:
                negative_aspects = summary_df[summary_df['Dominant Sentiment'] == 'Negative'].head(3)
                if not negative_aspects.empty:
                    resp = "The main negative aspects are:\n"
                    for _, row in negative_aspects.iterrows():
                        resp += f"- **{row['Aspect']}** (mentioned {int(row['Total Mentions'])} times, {row['Negative (%)']:.2f}% negative)\n"
                    return resp
                else:
                    return "Good news! No dominant negative aspects found."
            elif "sentiment for" in user_lower or "nps for" in user_lower:
                match = re.search(r"(?:sentiment|nps) for (.+)", user_lower)
                if match:
                    aspect_name = match.group(1).strip().title()
                    filtered = summary_df[summary_df['Aspect'].str.lower() == aspect_name.lower()]
                    if filtered.empty:
                        filtered = summary_df[summary_df['Aspect'].str.contains(aspect_name, case=False, na=False)]
                    if filtered.empty:
                        return f"Sorry, no data found for aspect '{aspect_name}'."
                    row = filtered.iloc[0]
                    avg_nps_display = row['Avg NPS'] if isinstance(row['Avg NPS'], (int, float)) else "N/A"
                    return (
                        f"For '**{row['Aspect']}**':\n"
                        f"- Positive: **{row['Positive (%)']:.2f}%**\n"
                        f"- Neutral: **{row['Neutral (%)']:.2f}%**\n"
                        f"- Negative: **{row['Negative (%)']:.2f}%**\n"
                        f"- Dominant sentiment: **{row['Dominant Sentiment']}**\n"
                        f"- Avg NPS: **{avg_nps_display}**\n"
                        f"- Promoters: **{int(row['Promoters'])}**\n"
                        f"- Passives: **{int(row['Passives'])}**\n"
                        f"- Detractors: **{int(row['Detractors'])}**"
                    )
                else:
                    return "Please specify an aspect. Example: *Sentiment for Delivery*"
            elif "recommendations for" in user_lower or "suggestions for" in user_lower:
                aspect_match = re.search(r"(?:recommendations|suggestions) for (.+)", user_lower)
                if aspect_match:
                    aspect_name = aspect_match.group(1).strip().title()
                    neg_reviews_list = top_neg_reviews_by_aspect.get(aspect_name, [])
                    if not neg_reviews_list:
                        matched_aspect = next(
                            (a for a in top_neg_reviews_by_aspect.keys() if aspect_name.lower() in a.lower()),
                            None
                        )
                        if matched_aspect:
                            neg_reviews_list = top_neg_reviews_by_aspect.get(matched_aspect, [])
                    if neg_reviews_list:
                        clean_reviews = [
                            re.sub(r"^\(.*?\)\s*", "", rev).strip()
                            for rev in neg_reviews_list
                            if rev and len(re.sub(r"^\(.*?\)\s*", "", rev).strip()) > 10
                        ]
                        if not clean_reviews:
                            return f"No substantial negative feedback found for **'{aspect_name}'**."
                        recommendations = build_recommendations_for_aspect(aspect_name, clean_reviews)
                        if recommendations:
                            resp = f"Here are actionable recommendations for **'{aspect_name}'** based on customer feedback:\n"
                            for i, rec in enumerate(recommendations, 1):
                                resp += f"{i}. {clean_text_for_pdf(rec)}\n"
                            return resp
                        else:
                            return f"No strong patterns found in negative feedback for **'{aspect_name}'** to suggest recommendations."
                    else:
                        return f"No negative mentions found for **'{aspect_name}'** in the reviews."
                else:
                    return "Please specify an aspect. Example: *Recommendations for Delivery*"
            elif "thank you" in user_lower or "bye" in user_lower:
                return "You're welcome! Feel free to ask more questions or explore your customer reviews anytime. 😊"
            else:
                return ("Try one of these: " +
                        " | ".join([f"`{s}`" for s in st.session_state['chat_suggestions']]) +
                        "\nOr ask about an aspect, e.g., *'sentiment for Food'* or *'recommendations for Service'*.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_turn = suggestion_clicked or chat_input
        if user_turn:
            st.session_state.messages.append({"role": "user", "content": user_turn})
            with st.chat_message("user"):
                st.markdown(user_turn)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = chatbot_response(user_turn, summary_df, top_neg_reviews_by_aspect)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state["chat_suggestions"] = generate_dynamic_suggestions(summary_df)

        st.markdown("""
            <div class='footer'>
            <b>Prepared for executive review. Powered by Streamlit, spaCy, NLTK.</b>
            </div>
        """, unsafe_allow_html=True)

elif uploaded_file and df_error:
    st.error(df_error)
