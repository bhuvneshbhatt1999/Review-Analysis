import re
import string
import textwrap
import tempfile
import os
import time
from collections import defaultdict, Counter
from io import BytesIO
import html
import pandas as pd
import numpy as np

# === NLP Setup ===
# Ensure resources are auto-downloaded for Streamlit Cloud

import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

import matplotlib.pyplot as plt
from fpdf import FPDF

# These dependencies are imported only if features use them.
try:
    from langdetect import detect
except ImportError:
    detect = None

try:
    from rake_nltk import Rake
except ImportError:
    Rake = None

########################
# --- Text Processing helpers --- #
########################

ARTICLES = {'the', 'a', 'an'}
DEMONSTRATORS = {'this', 'that', 'these', 'those'}

def clean_phrase(phrase: str) -> str:
    phrase = re.sub(r'[^\w\s]', '', phrase.lower()).strip()
    tokens = phrase.split()
    while tokens and tokens[0] in ARTICLES.union(DEMONSTRATORS):
        tokens.pop(0)
    doc = nlp(' '.join(tokens))
    lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas).strip()

def clean_text_for_pdf(text):
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '…': '...', '—': '-', '–': '-', '−': '-', '\u2212': '-',
        '‘': "'", '’': "'", '“': '"', '”': '"',
        '•': '-', '‒': '-', '―': '-',
        '″': '"', '′': "'", '\u2014': '-', '\u2013': '-',
        '\u2010': '-', '\u00A0': ' ', '\u202F': ' ', '\u2009': ' ', '\u200A': ' ',
        '\u2022': '-', '\u2032': "'", '\u2033': '"', '\t': ' '
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = ''.join(ch for ch in text if ch.isprintable() or ch in ['\n', '\t', '\r'])
    return text.strip()

def safe_quote(q):
    q = str(q)
    if len(q) > 300:
        q = q[:297] + "..."
    q = re.sub(r"(\w{20,})", lambda m: ' '.join(textwrap.wrap(m.group(0), 20)), q)
    q = re.sub(r'\s+', ' ', q)
    return q

def limit_large_df(df):
    return df  # No sampling by default; adjust if needed.

############################
# --- Heuristics, Column Detection --- #
############################

def auto_detect_column(df, candidates):
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    return None

def auto_detect_review_column(df):
    candidates = [
        'review', 'review_text', 'review content', 'feedback', 'comment', 'text',
        'body', 'message', 'content', 'remarks'
    ]
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols) == 0:
        raise ValueError("No text column found for analysis.")
    lengths = df[obj_cols].apply(lambda col: col.fillna("").astype(str).map(len).mean())
    return lengths.idxmax()

def auto_detect_nps_column(df):
    candidates = ['nps', 'nps_score', 'score', 'rating', 'net promoter score']
    for cand in candidates:
        col = auto_detect_column(df, [cand])
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return col
    numcols = df.select_dtypes(include=[np.number]).columns
    for col in numcols:
        vals = df[col].dropna()
        if not vals.empty and (vals.between(0, 10).mean() > 0.9):
            return col
    return None

###########################
# --- File Reading --- #
###########################

def safe_read_csv(file, **kwargs):
    filename = getattr(file, 'name', file)
    if isinstance(filename, str) and filename.endswith('.xlsx'):
        try:
            if hasattr(file, 'file'):
                return pd.read_excel(file.file, **kwargs)
            else:
                return pd.read_excel(file, **kwargs)
        except ImportError:
            raise ImportError("Install 'openpyxl': pip install openpyxl")
        except Exception as e:
            raise RuntimeError(f"Error reading Excel file: {e}")
    else:
        try:
            if hasattr(file, 'file'):
                return pd.read_csv(file.file, **kwargs)
            else:
                return pd.read_csv(file, **kwargs)
        except UnicodeDecodeError:
            try:
                if hasattr(file, 'file'):
                    return pd.read_csv(file.file, encoding='utf-8-sig', **kwargs)
                else:
                    return pd.read_csv(file, encoding='utf-8-sig', **kwargs)
            except UnicodeDecodeError:
                try:
                    if hasattr(file, 'file'):
                        return pd.read_csv(file.file, encoding='latin1', **kwargs)
                    else:
                        return pd.read_csv(file, encoding='latin1', **kwargs)
                except Exception as e:
                    raise RuntimeError(f"Error reading CSV file: {e}")

def detect_language_of_reviews(df, review_col):
    if detect is None:
        raise ImportError("Please install 'langdetect'; pip install langdetect")
    sample_texts = df[review_col].dropna().astype(str).sample(min(20, len(df)), random_state=42)
    lang_counts = Counter(detect(t) for t in sample_texts if t.strip())
    return lang_counts.most_common(1)[0][0] if lang_counts else "unknown"

################################
# --- Sentiment & Aspect Extraction --- #
################################

def aggregate_sentiment(counts):
    pos = counts.get('Positive', 0)
    neu = counts.get('Neutral', 0)
    neg = counts.get('Negative', 0)
    if pos > neu and pos > neg:
        return 'Positive'
    elif neg > pos and neg > neu:
        return 'Negative'
    else:
        return 'Neutral'

def aggregate_aspect_sentiments(occurrences):
    sentiments = [s for _, s, _ in occurrences]
    return aggregate_sentiment(Counter(sentiments))

def extract_dynamic_aspects(review_text):
    review_text = html.unescape(review_text)
    doc = nlp(review_text)
    extracted = defaultdict(list)
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        sent_score = sia.polarity_scores(sent_text)["compound"]
        if sent_score >= 0.3:
            label = "Positive"
        elif sent_score <= -0.3:
            label = "Negative"
        else:
            label = "Neutral"
        for chunk in sent.noun_chunks:
            asp_text = chunk.text.strip()
            norm = clean_phrase(asp_text)
            if len(norm) < 2 or norm in nlp.Defaults.stop_words:
                continue
            extracted[norm].append((sent_text, label, asp_text))
    data = []
    for aspect, mentions in extracted.items():
        agg_sent = aggregate_aspect_sentiments(mentions)
        context = "; ".join(sorted(set(m[0] for m in mentions)))
        quotes = [m[2] for m in mentions]
        data.append({
            "Review_ID": None,
            "Review": None,
            "Aspect": aspect.title(),
            "Aspect_Sentiment": agg_sent,
            "Aspect_Context": context,
            "Quotes": quotes,
        })
    return data

def analyze_review_structured(df, review_col, nps_col=None, progress_callback=None):
    if review_col not in df.columns:
        raise ValueError(f"Review column '{review_col}' not found.")
    if nps_col and nps_col not in df.columns:
        raise ValueError(f"NPS column '{nps_col}' not found.")
    if df[review_col].isnull().all():
        raise ValueError("All entries in review column are empty.")
    records = []
    total = len(df)
    for i, row in df.iterrows():
        text = str(row[review_col]).strip()
        if not text:
            continue
        nps_val = row.get(nps_col) if nps_col else None
        if isinstance(nps_val, str):
            nps_val = nps_val.strip()
            if nps_val.lower() in ('none', '', 'nan'):
                nps_val = None
        try:
            nps_val = float(nps_val)
            if not (0 <= nps_val <= 10):
                nps_val = None
        except (ValueError, TypeError):
            nps_val = None
        aspects = extract_dynamic_aspects(text)
        for asp in aspects:
            asp["Review_ID"] = i + 1
            asp["Review"] = text
            asp["NPS_Score"] = nps_val
            records.append(asp)
        if progress_callback:
            progress_callback((i + 1) / total, f"Processed {i + 1}/{total} reviews...")
    return pd.DataFrame(records)

def generate_sentiment_summary(df):
    summary_rows = []
    grouped = df.groupby("Aspect")
    for aspect, group in grouped:
        counts = Counter(group["Aspect_Sentiment"])
        total = len(group)
        pos = counts.get("Positive", 0)
        neu = counts.get("Neutral", 0)
        neg = counts.get("Negative", 0)
        pos_pct = total and pos / total * 100 or 0
        neu_pct = total and neu / total * 100 or 0
        neg_pct = total and neg / total * 100 or 0
        dominant = aggregate_sentiment(counts)
        avg_nps = promoters = passives = detractors = 0
        if "NPS_Score" in group.columns and group["NPS_Score"].notna().any():
            valid_nps = group["NPS_Score"].dropna()
            avg_nps = valid_nps.mean() if not valid_nps.empty else np.nan
            promoters = valid_nps[(valid_nps >= 9) & (valid_nps <= 10)].count()
            passives = valid_nps[(valid_nps >= 7) & (valid_nps <= 8)].count()
            detractors = valid_nps[(valid_nps >= 0) & (valid_nps <= 6)].count()
        raw_quotes = []
        for quotes_cell in group["Quotes"]:
            if isinstance(quotes_cell, list):
                raw_quotes.extend(quotes_cell)
            elif isinstance(quotes_cell, str):
                raw_quotes.append(quotes_cell)
        unique_quotes = []
        seen = set()
        for q in raw_quotes:
            q_clean = q.strip()
            if q_clean and q_clean not in seen:
                unique_quotes.append(q_clean)
                seen.add(q_clean)
            if len(unique_quotes) >= 3:
                break
        summary_rows.append({
            "Aspect": aspect,
            "Total Mentions": total,
            "Positive": pos,
            "Neutral": neu,
            "Negative": neg,
            "Positive (%)": round(pos_pct, 2),
            "Neutral (%)": round(neu_pct, 2),
            "Negative (%)": round(neg_pct, 2),
            "Dominant Sentiment": dominant,
            "Avg NPS": round(avg_nps, 2) if not pd.isna(avg_nps) else "N/A",
            "Promoters": promoters,
            "Passives": passives,
            "Detractors": detractors,
            "Sample Quotes": unique_quotes,
        })
    return pd.DataFrame(summary_rows).sort_values(by="Total Mentions", ascending=False)

def benchmark_kpis(df_summary, df_detail=None):
    total_mentions = df_summary["Total Mentions"].sum()
    pos_total = df_summary["Positive"].sum()
    neu_total = df_summary["Neutral"].sum()
    neg_total = df_summary["Negative"].sum()
    data = {
        "Total Mentions": f"{total_mentions:,}",
        "Positive Mentions (%)": f"{pos_total / total_mentions * 100:.2f}%" if total_mentions else "0.0%",
        "Neutral Mentions (%)": f"{neu_total / total_mentions * 100:.2f}%" if total_mentions else "0.0%",
        "Negative Mentions (%)": f"{neg_total / total_mentions * 100:.2f}%" if total_mentions else "0.0%",
    }
    if df_detail is not None and "NPS_Score" in df_detail.columns:
        nps_vals = df_detail["NPS_Score"].dropna()
        if not nps_vals.empty:
            avg = nps_vals.mean()
            promoters = nps_vals[(nps_vals >= 9) & (nps_vals <= 10)].count()
            passives = nps_vals[(nps_vals >= 7) & (nps_vals <= 8)].count()
            detractors = nps_vals[(nps_vals >= 0) & (nps_vals <= 6)].count()
            nps_score = (promoters - detractors) / len(nps_vals) * 100
            data.update({
                "Average NPS Score": f"{avg:.2f}",
                "NPS Score (%)": f"{nps_score:.1f}%",
                "Promoters": f"{promoters}",
                "Passives": f"{passives}",
                "Detractors": f"{detractors}",
            })
    return pd.DataFrame([{"KPI": k, "Value": v} for k, v in data.items()])

def create_aspect_bar_chart(df_summary):
    fig, ax = plt.subplots(figsize=(8, 4))
    top5 = df_summary.head(5)
    top5.plot(
        kind='bar',
        x='Aspect',
        y=['Positive', 'Neutral', 'Negative'],
        stacked=True,
        color=['#53b944', '#f7c948', '#ed3b39'],
        ax=ax
    )
    ax.set_ylabel("Mentions")
    ax.set_title("Customer sentiment by key aspects", fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

################################
# --- PDF Reporting --- #
################################

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.first_page = True
        try:
            base = os.path.dirname(__file__)
            fonts_path = os.path.join(base, "fonts")
            if not os.path.exists(fonts_path):
                fonts_path = base
            self.add_font('DejaVu', '', os.path.join(fonts_path, 'DejaVuSans.ttf'), uni=True)
            self.add_font('DejaVu', 'B', os.path.join(fonts_path, 'DejaVuSans-Bold.ttf'), uni=True)
            self.font_family = 'DejaVu'
        except Exception:
            self.font_family = 'Arial'

    def header(self):
        if self.page_no() == 1:
            self.set_font(self.font_family, 'B', 16)
            self.set_text_color(45, 65, 155)
            self.cell(0, 10, "CUSTOMER REVIEW SENTIMENT ANALYSIS", 0, 1, 'C')
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_family, 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, 'C')

    def section(self, title, size=14, after_space=1):
        self.set_font(self.font_family, 'B', size)
        self.set_text_color(15, 45, 90)
        self.cell(0, self.font_size_pt + 4, clean_text_for_pdf(title.upper()), 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.ln(after_space)

    def subheading(self, text, size=12, after_space=1):
        self.set_font(self.font_family, 'B', size)
        self.set_text_color(40, 40, 120)
        self.cell(0, self.font_size_pt + 3, clean_text_for_pdf(text), 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.ln(after_space)

    def add_paragraph(self, text, size=11, line_height=5, after_space=1):
        self.set_font(self.font_family, '', size)
        cleaned = clean_text_for_pdf(text)
        if not cleaned:
            cleaned = "[Empty or invalid text]"
        self.multi_cell(0, line_height, cleaned)
        self.ln(after_space)

    def add_table(self, df, title=None, fontsize=9, col_title_fontsize=9, truncate_columns=None):
        if title:
            self.set_font(self.font_family, 'B', fontsize + 2)
            self.cell(0, fontsize + 6, clean_text_for_pdf(title), 0, 1)
        num_cols = len(df.columns)
        available_width = self.epw - (num_cols + 1) * self.c_margin
        col_widths = [available_width / num_cols] * num_cols
        row_height = 6

        self.set_font(self.font_family, 'B', col_title_fontsize)
        for i, col in enumerate(df.columns):
            colname = col
            if truncate_columns and col in truncate_columns:
                colname = (str(col)[:20] + "...") if len(str(col)) > 20 else str(col)
            self.cell(col_widths[i], row_height, clean_text_for_pdf(colname), border=1, align='C')
        self.ln(row_height)

        self.set_font(self.font_family, '', fontsize)
        for _, row in df.iterrows():
            if self.get_y() + row_height > self.h - self.b_margin:
                self.add_page()
                self.set_font(self.font_family, 'B', col_title_fontsize)
                for i, col in enumerate(df.columns):
                    colname = col
                    if truncate_columns and col in truncate_columns:
                        colname = (str(col)[:20] + "...") if len(str(col)) > 20 else str(col)
                    self.cell(col_widths[i], row_height, clean_text_for_pdf(colname), border=1, align='C')
                self.ln(row_height)
                self.set_font(self.font_family, '', fontsize)

            for i, col in enumerate(df.columns):
                val = row[col]
                val_str = str(val)
                if truncate_columns and col in truncate_columns and len(val_str) > 40:
                    val_str = val_str[:40] + "..."
                self.cell(col_widths[i], row_height, clean_text_for_pdf(val_str), border=1)
            self.ln(row_height)

    def add_image(self, buf, width=None, caption=None, caption_center=True):
        if caption:
            self.set_font(self.font_family, 'I', 10)
            self.cell(0, 8, clean_text_for_pdf(caption), 0, 1, 'C' if caption_center else 'L')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(buf.getvalue())
            tmp.flush()
            image_path = tmp.name
        self.image(image_path, w=width if width else self.epw)
        os.remove(image_path)
        self.ln()

################################
# --- Recommendations & Negative Reviews --- #
################################

def build_recommendations_for_aspect(aspect, neg_reviews):
    if not Rake or not neg_reviews or all(not rev.strip() for rev in neg_reviews):
        return [f"No actionable recommendations for '{aspect}' at this time."]
    try:
        rake = Rake(min_length=3, max_length=6)
        cleaned_reviews = [re.sub(r'\bpage\s*\d+\b', '', rev, flags=re.I).strip() for rev in neg_reviews]
        combined_text = " ".join(cleaned_reviews)
        rake.extract_keywords_from_text(combined_text)
        phrases = [
            phr for phr in rake.get_ranked_phrases()
            if len(phr.strip()) > 6 and len(phr.split()) > 2 
            and not re.search(r'(kitten|page 3|\.\.|broken|s\b)', phr.lower())
        ]
    except Exception:
        phrases = []
    seen = set()
    key_points = []
    aspect_lower = aspect.lower()
    for phrase in phrases:
        phrase_clean = phrase.strip().capitalize()
        phrase_clean = phrase_clean.rstrip(' .,;:!?')
        if phrase_clean.lower() not in seen and len(phrase_clean.split()) > 2:
            seen.add(phrase_clean.lower())
            key_points.append(phrase_clean)
        if len(key_points) >= 5:
            break
    recommendations = []
    for point in key_points:
        point_lower = point.lower()
        if point_lower.startswith(aspect_lower):
            trimmed = point[len(aspect):].lstrip(' .,:;-–—')
            formatted = trimmed[0].upper() + trimmed[1:] if trimmed else point
        else:
            formatted = point
        if not any(formatted.endswith(punct) for punct in ['.', '!', '?']):
            formatted += '.'
        recommendations.append(formatted)
    if not recommendations:
        recommendations = [f"No strong actionable recommendations found for '{aspect}'."]
    return recommendations

def extract_top_negative_reviews_by_aspect(detail_df, aspects, max_reviews=5):
    def normalize(text):
        text = re.sub(r'page\s*\d+', '', text, flags=re.I)
        text = re.sub(r'^\d+\.\s*', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    reviews = {}
    unique_reviews = detail_df[['Review_ID', 'Review']].drop_duplicates()
    overall_scores = {
        row['Review_ID']: sia.polarity_scores(row['Review'])['compound']
        for _, row in unique_reviews.iterrows()
    }
    for aspect in aspects:
        filtered = detail_df[
            (detail_df['Aspect'] == aspect) &
            (detail_df['Aspect_Sentiment'] == 'Negative')
        ].copy()
        if filtered.empty:
            reviews[aspect] = []
            continue
        filtered['Overall_Score'] = filtered['Review_ID'].map(overall_scores)
        filtered['Context_Score'] = filtered['Aspect_Context'].apply(
            lambda x: sia.polarity_scores(x)['compound']
        )
        filtered['Is_Overall_Negative'] = filtered['Overall_Score'] <= -0.3
        filtered = filtered.sort_values(['Is_Overall_Negative', 'Context_Score'], ascending=[False, True])
        deduped_texts = []
        seen_texts = set()
        for _, row in filtered.iterrows():
            norm = normalize(row['Review'])
            if not norm or norm in seen_texts:
                continue
            seen_texts.add(norm)
            cleaned_review = clean_text_for_pdf(row['Review'])
            if cleaned_review:
                deduped_texts.append(cleaned_review)
            if len(deduped_texts) >= max_reviews:
                break
        reviews[aspect] = deduped_texts
    return reviews

def create_pdf_report(detail_df, summary_df):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    uploaded_count = detail_df['Review_ID'].max() if 'Review_ID' in detail_df else 'N/A'
    analyzed_count = len(detail_df['Review_ID'].unique()) if 'Review_ID' in detail_df else 'N/A'

    pdf.section("EXECUTIVE SUMMARY & KEY INSIGHTS")
    if summary_df.empty:
        pdf.add_paragraph("No data to summarize.")
    else:
        top = summary_df.iloc[0]
        pdf.add_paragraph(
            f"Uploaded {uploaded_count:,} reviews; analyzed {analyzed_count:,} reviews. "
            f"{len(summary_df)} unique aspects found. "
            f"Top mentioned aspect: {top['Aspect']} ({top['Total Mentions']} mentions)."
        )

    pdf.section("KEY METRICS & KPI OVERVIEW")
    kpi_df = benchmark_kpis(summary_df, detail_df)
    pdf.add_table(kpi_df, title="Sentiment & NPS Score KPIs")

    pdf.section("SENTIMENT DISTRIBUTION BY TOP ASPECTS")
    pdf.add_image(create_aspect_bar_chart(summary_df))

    pdf.section("DETAILED METRICS")
    cols = [
        "Aspect", "Total Mentions", "Positive", "Neutral", "Negative",
        "Positive (%)", "Neutral (%)", "Negative (%)",
        "Dominant Sentiment", "Avg NPS", "Promoters", "Passives", "Detractors"
    ]
    rename_columns = {
        "Aspect": "Aspect",
        "Total Mentions": "T.M.",
        "Positive": "Pos.",
        "Neutral": "Neu.",
        "Negative": "Neg.",
        "Positive (%)": "Pos%",
        "Neutral (%)": "Neu%",
        "Negative (%)": "Neg%",
        "Dominant Sentiment": "Dom.",
        "Avg NPS": "Avg NPS",
        "Promoters": "Promo.",
        "Passives": "Passi.",
        "Detractors": "Detrac."
    }
    subset_df = summary_df[cols].copy()
    subset_df.rename(columns=rename_columns, inplace=True)
    pdf.add_table(subset_df.head(10), fontsize=8, col_title_fontsize=8)

    pdf.section("RECENT NEGATIVE REVIEWS BY ASPECT")
    top_aspects = summary_df['Aspect'].head(5).tolist()
    neg_reviews = extract_top_negative_reviews_by_aspect(detail_df, top_aspects, max_reviews=5)
    for asp in top_aspects:
        pdf.subheading(f"{asp} - Negative Reviews")
        reviews = neg_reviews.get(asp, [])
        if not reviews:
            pdf.add_paragraph("No negative reviews found for this aspect.", size=9)
            continue
        for i, rev in enumerate(reviews, 1):
            pdf.add_paragraph(f"{i}. {safe_quote(rev)}", size=9)

    pdf.section("ACTIONABLE RECOMMENDATIONS")
    for asp in top_aspects:
        pdf.subheading(f"{asp} - Recommendations")
        recs = build_recommendations_for_aspect(asp, neg_reviews.get(asp, []))
        for i, rec in enumerate(recs, 1):
            pdf.add_paragraph(f"{i}. {rec}", size=9)

    output_pdf = pdf.output(dest='S')
    if isinstance(output_pdf, str):
        output_pdf = output_pdf.encode('utf-8', 'replace')
    if isinstance(output_pdf, bytearray):
        output_pdf = bytes(output_pdf)
    return output_pdf
