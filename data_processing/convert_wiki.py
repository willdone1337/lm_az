import argparse
import json
import re
import random
import string
import unicodedata

from tqdm import tqdm
from corus import load_wiki
import razdel

from data_processing.lang_detector import FasttextLanguageDetector
from data_processing.util import PlainArchive, TextProcessor

RE_MARKUP = re.compile(
    r'<br( [^>]+)?>|'               # e.g. <br>, <br style="clear: both">
    r'<chem>[^<]*</chem>|'          # e.g. <chem>NH3 </chem>

    # e.g. <ref name="..."></ref>, <ref group="..." name="..."></ref>,
    # <ref name="Kath26/07/2011"> , "I Kathimeriní", .</ref>,
    # <ref name="2015/07/29 powersearch">Article "..." de Michael Lipka, paru ...</ref>,
    # sometimes opening/closing tags are on separate lines
    r'<ref\b[^>]*/ref>|'  # weird one line ref, e.g. <ref[oanda.com, March 9, 2022]/ref>
    r'<ref [^>]+>[^<]*</ref>|'          # one line
    r'<ref [^>]+>[^<]*$|^[^<]*</ref>|'  # two lines

    # Remnants of tables:
    r'^!.*=.*\||'                   # e.g. '! colspan="5" style=background:;|'
    # Catch anything that passes the sub-expression above:
    # - common unquoted attribute values:
    r'\bhidden=1\b|'
    # - quoted values, e.g. align="left",
    # - unquoted values upto "|" on the same line,
    #   e.g. 'frame-style = border: 1px solid rgb(200,200,200); |',
    # - unquoted values consisting of a single word, e.g. 'align=left'
    r'\b(rowspan|colspan|width|style|bgcolor|align|valign|frame-style|title-style|'
    r'content-style)\s*=\s*("[^"]*"|.*\||\w+|)|'

    # Code and formula placeholders:
    r'(codice|formula)_[0-9]+|'     # e.g. 'DNAformula_20', '様にcodice_1 の'

    # <ins>text</ins>, <del>text</del>,
    # randomly appearing <math>/</math> tags
    # <onlyinclude>/</onlyinclude>/<onlyinclude/> in pages only linking to another page:
    r'</?(ins|del|math|onlyinclude)>|<onlyinclude/>|'

    # Sometimes there are blocks of the following (desambiuation/redirection, etc.)
    r'<ns>.*?</ns>|'
    r'<parentid>.*?</parentid>|'
    r'<revision>|'
    r'<timestamp>.*?</timestamp>|'
    r'</?contributor>|'
    r'<username>.*?</username>|'
    r'<minor />|'
    r'<comment>.*?</comment>|'
    r'<model>.*?</model>|'
    r'<format>.*?</format>'
)

RE_HEADERS = re.compile(r"(=+)\s*([^=]+?)\s*\1", flags=re.MULTILINE)
RE_BRACKETS = re.compile(r"\([^\)]*\)", flags=re.MULTILINE)
TEXT_PROCESSOR = TextProcessor()


def count_punct_part(sentence):
    punct_count = 0.0
    all_count = 0.0
    for ch in sentence:
        if ch in string.punctuation:
            punct_count += 1.0
        all_count += 1.0
    return punct_count / all_count


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_text(text):
    for _ in range(10):
        text = text.replace("::", " ")

    text = TEXT_PROCESSOR(text)
    if not text:
        return None
    paragraphs = text.split("\n")

    # Remove title duplicates
    if len(paragraphs) <= 1:
        return None
    first_paragraph = strip_accents(paragraphs[0])
    second_paragraph = strip_accents(paragraphs[1])
    if second_paragraph.startswith(first_paragraph):
        paragraphs = paragraphs[1:]
    elif not second_paragraph.strip():
        paragraphs = paragraphs[2:]
    paragraphs = [p if p[-1] in string.punctuation else p + "." for p in paragraphs]
    text = "\n".join(paragraphs)

    # remove templates
    text = re.sub(r"\[\d+?\]", " ", text)
    text = re.sub(r"\{\{+[^{}]+?\}\}+", " ", text)

    text = RE_MARKUP.sub(" ", text)
    text = text.replace("*", " ")
    text = " ".join(text.split())

    # split by headers
    headers = RE_HEADERS.finditer(text)
    for header in headers:
        header = header.group()
        if len(header) > 100:
            continue
        text = text.replace(header, "\n")
    text = text.replace("=", " ")

    # remove bracketed texts
    brackets = RE_BRACKETS.finditer(text)
    for bracket in brackets:
        bracket = bracket.group()
        if len(bracket) > 200:
            continue
        text = text.replace(bracket, " ")

    # remove footnotes
    text = re.sub(r" \^ .+", " ", text)

    text = TEXT_PROCESSOR(text)
    if not text:
        return None

    # remove bad sentences/paragraphs
    paragraphs = text.split("\n")
    fixed_paragraphs = []
    for paragraph in paragraphs:
        if count_punct_part(paragraph) > 0.2:
            continue
        sentences = [s.text for s in razdel.sentenize(paragraph)]
        sentences = [s for s in sentences if len(s) > 5]
        sentences = [s for s in sentences if count_punct_part(s) < 0.1]
        paragraph = " ".join(sentences)
        if ": :" in paragraph or "}}" in paragraph or "//" in paragraph:
            continue
        if len(paragraph) < 20:
            continue
        if paragraph[0] in string.punctuation:
            continue
        fixed_paragraphs.append(paragraph)
    text = "\n".join(fixed_paragraphs)
    return text


def main(
    input_path,
    output_path,
    sample_rate
):
    records = load_wiki(input_path)
    archive = PlainArchive(output_path)
    for record in tqdm(records):
        if random.random() > sample_rate:
            continue
        title = record.title
        text = record.text
        rid = record.id
        text = preprocess_text(text)
        if not text:
            continue
        if len(text) < 300:
            continue
        archive.add_data(
            text=text,
            meta={
                "source": "wiki",
                "url": f"https://ru.wikipedia.org/?curid={rid}"
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    args = parser.parse_args()
    main(**vars(args))
