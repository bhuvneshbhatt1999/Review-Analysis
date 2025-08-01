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
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
from fpdf import FPDF
import gradio as gr

#####################
# Dependency checks #
#####################
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError as err:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. Install via 'python -m spacy download en_core_web_sm'."
    ) from err

sia = SentimentIntensityAnalyzer()

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
    """Safely clean text for PDF output while preserving readability."""
    if not isinstance(text, str):
        text = str(text)

    # Handle common Unicode replacements
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

    # Allow printable Unicode (e.g., accented characters), not just ASCII
    text = ''.join(ch for ch in text if ch.isprintable() or ch in ['\n', '\t', '\r'])

    return text.strip()


def safe_quote(q):
    q = str(q)
    if len(q) > 300:
        q = q[:297] + "..."
    q = re.sub(r"(\w{20,})", lambda m: ' '.join(textwrap.wrap(m.group(0), 20)), q)
    q = re.sub(r'\s+', ' ', q)
    return q


def perfectly_format_numbered_reviews(text, width=85, indent=' '):
    text = re.sub(r'\n{2,}', '\n', text)
    numbered = re.split(r'(?m)^\s*(\d+\.)', text)
    output = []
    i = 1
    while i < len(numbered):
        num = numbered[i].strip()
        rest = numbered[i + 1].strip()
        lines = rest.split('\n')
        wrapped_first = textwrap.fill(f"{num} {lines[0]}", width=width) if lines else f"{num} "
        other_lines = [line for line in lines[1:] if line.strip()]
        wrapped_others = [
            textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent)
            for line in other_lines
        ]
        combined = [wrapped_first] + wrapped_others
        output.append('\n'.join(combined))
        i += 2
    return '\n'.join(output)


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


def detect_language_of_reviews(df, review_col):
    try:
        from langdetect import detect
    except ImportError:
        raise ImportError(
            "Please install 'langdetect'; pip install langdetect"
        )
    sample_texts = df[review_col].dropna().astype(str).sample(min(20, len(df)), random_state=42)
    lang_counts = Counter(detect(t) for t in sample_texts if t.strip())
    return lang_counts.most_common(1)[0][0] if lang_counts else "unknown"


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


def limit_large_df(df):
    return df


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
        raise ValueError(f"All entries in review column '{review_col}' are empty.")

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
            asp["Review"] = text  # Ensure full text is preserved
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


class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.first_page = True
        try:
            base = os.path.dirname(__file__)
            fonts_path = os.path.join(base, "dejavu-fonts-ttf-2.37", "ttf")
            if not os.path.exists(fonts_path):
                fonts_path = os.path.join(base, "fonts")
            if not os.path.exists(fonts_path):
                fonts_path = base
            self.add_font('DejaVu', '', os.path.join(fonts_path, 'DejaVuSans.ttf'), uni=True)
            self.add_font('DejaVu', 'B', os.path.join(fonts_path, 'DejaVuSans-Bold.ttf'), uni=True)
            self.add_font('DejaVu', 'I', os.path.join(fonts_path, 'DejaVuSans-Oblique.ttf'), uni=True)
            self.add_font('DejaVu', 'BI', os.path.join(fonts_path, 'DejaVuSans-BoldOblique.ttf'), uni=True)
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


def build_recommendations_for_aspect(aspect, neg_reviews):
    if not neg_reviews or all(not rev.strip() for rev in neg_reviews):
        return [f"No actionable recommendations for '{aspect}' at this time."]
    try:
        from rake_nltk import Rake
        rake = Rake(min_length=3, max_length=6)  # Prefer slightly longer phrases
        # Preprocess text: remove obviously irrelevant strings like "page 3"
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
            if trimmed:
                formatted = trimmed[0].upper() + trimmed[1:]
            else:
                formatted = point
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
        text = re.sub(r'^\d+\.\s*', '', text)  # Remove numbering
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
    # === The key fix below for Streamlit compatibility ===
    if isinstance(output_pdf, str):
        output_pdf = output_pdf.encode('utf-8', 'replace')
    if isinstance(output_pdf, bytearray):
        output_pdf = bytes(output_pdf)
    return output_pdf


# ========== Global State ==========
global_detail_df = pd.DataFrame()
global_summary_df = pd.DataFrame()
global_top_neg_reviews = {}
global_nps_col = None


def run_analysis(csv_file, review_column=None, nps_column=None, progress=gr.Progress()):
    global global_detail_df, global_summary_df, global_top_neg_reviews, global_nps_col

    if csv_file is None:
        return "Upload CSV file.", None, None, None, None

    try:
        df = safe_read_csv(csv_file)
        if nps_column and nps_column in df.columns:
            df[nps_column] = pd.to_numeric(df[nps_column], errors='coerce')

        uploaded_count = len(df)
        review_col = review_column if review_column and review_column in df.columns else auto_detect_review_column(df)
        nps_col = nps_column if nps_column and nps_column in df.columns else auto_detect_nps_column(df)

        if nps_col:
            nps_msg = f"Auto-selected NPS column: **{nps_col}**."
        else:
            nps_msg = "NPS column not detected, skipping NPS analysis."

        lang = detect_language_of_reviews(df, review_col)
        if lang != 'en':
            return f"Detected language: {lang}. Only English is supported.", None, None, None, None

        detail_df = analyze_review_structured(df, review_col, nps_col, progress.update)
        summary_df = generate_sentiment_summary(detail_df)

        global_detail_df = detail_df
        global_summary_df = summary_df
        global_top_neg_reviews = extract_top_negative_reviews_by_aspect(detail_df, summary_df['Aspect'].head(5).tolist())

        summary_md = summary_df.head(10).to_markdown(index=False)
        detail_md = detail_df.head(20).to_markdown(index=False)
        pdf_bytes = create_pdf_report(detail_df, summary_df)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            temp_pdf_path = tmp.name

        msg = f"✅ Analysis complete. {uploaded_count} reviews uploaded, {len(detail_df['Review_ID'].unique())} analyzed."
        if nps_msg:
            msg += f" {nps_msg}"

        return msg, summary_md, detail_md, temp_pdf_path, gr.update(visible=True)

    except Exception as e:
        import traceback
        return f"Error: {e}\n{traceback.format_exc()}", None, None, None, gr.update(visible=False)


def chatbot_query(message, history):
    if global_summary_df.empty or not global_top_neg_reviews:
        return "", history + [[message, "Please upload and analyze data first."]]

    msg_lower = message.lower()
    response = ""

    if any(word in msg_lower for word in ["thank", "bye"]):
        response = "You're welcome! Ask more questions anytime."

    elif any(word in msg_lower for word in ["main negative", "top problems"]):
        neg_aspects = global_summary_df[global_summary_df['Dominant Sentiment'] == 'Negative']
        if neg_aspects.empty:
            response = "No dominant negative aspects found."
        else:
            response = "Main negative aspects:\n"
            for _, row in neg_aspects.head(3).iterrows():
                avg_nps = row['Avg NPS'] if isinstance(row['Avg NPS'], (int, float)) else "N/A"
                response += f"- **{row['Aspect']}** ({row['Total Mentions']} mentions, {row['Negative (%)']}% negative, Avg NPS: {avg_nps})\n"

    elif re.search(r"(sentiment|nps) for (.+)", msg_lower):
        match = re.search(r"(sentiment|nps) for (.+)", msg_lower)
        aspect_name = match.group(2).strip().title()
        row = global_summary_df[global_summary_df['Aspect'].str.lower() == aspect_name.lower()]
        if row.empty:
            row = global_summary_df[global_summary_df['Aspect'].str.contains(aspect_name, case=False)]
        if row.empty:
            response = f"No data for aspect '{aspect_name}'."
        else:
            row = row.iloc[0]
            avg_nps = row['Avg NPS'] if isinstance(row['Avg NPS'], (int, float)) else "N/A"
            response = (
                f"For **{row['Aspect']}**:\n"
                f"- Positive: {row['Positive (%)']}%\n"
                f"- Neutral: {row['Neutral (%)']}%\n"
                f"- Negative: {row['Negative (%)']}%\n"
                f"- Dominant: **{row['Dominant Sentiment']}**\n"
                f"- Avg NPS: **{avg_nps}**"
            )

    elif re.search(r"(recommendations?|suggestions?) for (.+)", msg_lower):
        match = re.search(r"(recommendations?|suggestions?) for (.+)", msg_lower)
        aspect_name = match.group(2).strip().title()
        rows = global_summary_df[global_summary_df['Aspect'].str.lower() == aspect_name.lower()]
        if rows.empty:
            rows = global_summary_df[global_summary_df['Aspect'].str.contains(aspect_name, case=False)]
        if rows.empty:
            response = f"No data for '{aspect_name}'."
        else:
            actual = rows.iloc[0]['Aspect']
            reviews = global_top_neg_reviews.get(actual, [])
            if not reviews:
                response = f"No negative reviews for '{actual}' to generate recommendations."
            else:
                recs = build_recommendations_for_aspect(actual, reviews)
                response = f"Recommendations for '{actual}':\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(recs, 1))

    else:
        response = (
            "Try asking:\n"
            "- What are the main negative aspects?\n"
            "- Sentiment for Food\n"
            "- Recommendations for Service"
        )

    history.append([message, response])
    return "", history


# ==== Gradio UI ====
with gr.Blocks() as demo:
    gr.Markdown("# Customer Reviews and NPS Sentiment Analysis")
    gr.Markdown("Upload your CSV/Excel file to analyze customer feedback and generate insights.")

    with gr.Row():
        with gr.Column(scale=1):
            csv_input = gr.File(label="Upload File (.csv or .xlsx)")
            review_input = gr.Textbox(label="Review Column (Optional)", placeholder="Leave blank to auto-detect")
            nps_input = gr.Textbox(label="NPS Column (Optional)", placeholder="e.g., nps_score")
            analyze_btn = gr.Button("🚀 Analyze Reviews")
            status_text = gr.Markdown("📊 Upload a file to begin.")
            pdf_output_file = gr.File(label="📥 Download Full PDF Report", visible=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Ask Insights", height=400)
            chatbox = gr.Textbox(label="Ask a question:", placeholder="E.g., 'What are the main negative aspects?'")
            clear_btn = gr.Button("🗑️ Clear Chat")

    gr.Markdown("---")
    gr.Markdown("## Results")
    with gr.Tabs():
        with gr.TabItem("Summary (Top 10)"):
            summary_output = gr.Markdown("Summary will appear here.")
        with gr.TabItem("Raw Mentions (First 20)"):
            detail_output = gr.Markdown("Detailed data will appear here.")

    analyze_btn.click(
        fn=run_analysis,
        inputs=[csv_input, review_input, nps_input],
        outputs=[status_text, summary_output, detail_output, pdf_output_file, pdf_output_file]
    )
    chatbox.submit(fn=chatbot_query, inputs=[chatbox, chatbot], outputs=[chatbox, chatbot])
    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
