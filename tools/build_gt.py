"""Rebuild the reference MusicXML the paper evaluates against.

This repository ships no ground truth. The symbolic source is the CCARH MuseData edition of the
Beethoven symphonies, published at https://bitbucket.org/musedata/beethoven -- publicly readable,
but under copyright ((C) Center for Computer Assisted Research in the Humanities), which is why we
fetch it rather than redistribute it. Acknowledge CCARH if you repost anything derived from it.

    # the ten pages scripts/run_sample.sh uses
    python tools/build_gt.py --symphony 1 --pages 1-10

    # a whole symphony, or all 1062 pages
    python tools/build_gt.py --symphony 1
    python tools/build_gt.py --all

Output: data/sample/gt/musicxml/{symphony}_{movement}_{page}.musicxml, where `page` restarts at 1
in every movement -- the numbering stage 3 expects.

Two conversions happen here. `md22musicxml.py` turns MuseData stage-2 (.md2) into one MusicXML per
movement, keeping instrument layout, notes and beams and dropping articulation, dynamics and slurs;
the paper strips the same classes from every system's predictions before scoring. Then each movement
is cut at the measure where a new page begins in the Eulenburg print, which is what makes a page of
reference comparable to a page of prediction. Those measure numbers are in page_boundaries.json.

`md22musicxml.py` is vendored verbatim -- the paper's numbers were produced with exactly this file,
so do not "fix" it here without regenerating and rescoring everything. One known wart: a duplicated
`len(onlyNote) == 0` branch means single notes are emitted as one-note chords. It makes no
difference to the exported MusicXML, which is why it survived.

page_boundaries.json is preserved with the same care, and it has three entries where a page starts
at an *earlier* measure than the page before it: symphony 6, movement 3 page 14, movement 4 page 9,
and movement 5 page 12. These look like transcription slips in the boundary table, and they make
those pages come out short. They are kept as they are because the paper's reference set was cut
with exactly these numbers; correcting them would silently change the denominator of every TEDn and
OMR-NED score reported for symphony 6.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

MD2_URL = ("https://bitbucket.org/musedata/beethoven/raw/master/"
           "bhl/orch/sym{sym}/editions/public/score/mvt{mvt}.md2")


def load_boundaries():
    with open(os.path.join(HERE, "page_boundaries.json")) as f:
        return json.load(f)


def fetch_md2(sym, mvt, cache_dir):
    """Download one movement's MuseData, cached so repeat runs are offline."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"sym{sym}_mvt{mvt}.md2")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = MD2_URL.format(sym=sym, mvt=mvt)
    print(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    if len(data) < 1000 or data.lstrip()[:1] == b"<":
        raise RuntimeError(f"unexpected response for {url} ({len(data)} bytes)")
    with open(path, "wb") as f:
        f.write(data)
    return path


def parse_md2(md2_path):
    """Run the vendored parser over one .md2 and return a music21 Score.

    This is md22musicxml.py's own __main__ loop, with the file list and the output step lifted out
    so it can be called per movement. The parsing logic itself is untouched.
    """
    import md22musicxml as P
    from music21 import stream

    with open(md2_path, "r", encoding="utf-8", errors="replace") as f:
        allStrings = [line.rstrip("\n") for line in f]

    score = stream.Score()
    parserState = P.ParserState()
    skipUntilLine = -1
    isComment = False
    measureAccumList = []
    currentMeasureNumber = 1
    inMeasureTitle = ['A', 'B', 'C', 'D', 'E', 'F', 'G', ' ', 'b', 'i', 'r']
    ignoreList = ['S', '*', 'g', 'f', 'c']

    for line_num, line in enumerate(allStrings):
        if (line_num < skipUntilLine or len(line) < 1
                or line[0] == '@' or line[0] == 'P'
                or (isComment and line[0] != '&')):
            continue
        if line.startswith("/END"):
            score.append(parserState.getCurrPart())
            parserState.reset()
        elif line.startswith("&&&&&&"):
            skipUntilLine = line_num + 16
            currentMeasureNumber = 1
            parserState.setCurrentInstrument(allStrings[line_num + 11])
        elif line[0] == '&':
            isComment = not isComment
        elif line[0] == '$':
            parserState.updateParserState(line)
        elif line[0] in inMeasureTitle:
            measureAccumList.append(line)
        elif line[0] == 'm':
            parserState.addMeasureToPart(
                P.parseMeasure(measureAccumList, parserState, currentMeasureNumber))
            measureAccumList = []
            currentMeasureNumber += 1
        elif line[0] in ignoreList or line.startswith('/'):
            continue
    return score


def slice_score(score, start, end):
    """One page: measures [start, end] of every part. From cut_gt_*.py."""
    import copy
    from music21 import stream
    new_score = stream.Score()
    for part in score.parts:
        new_part = stream.Part()
        new_part.insert(0, part.getInstrument())
        for m in part.measures(start, end):
            new_part.append(copy.deepcopy(m))
        new_score.append(new_part)
    return new_score


def parse_pages(spec):
    if not spec:
        return None
    out = set()
    for chunk in spec.split(","):
        if "-" in chunk:
            a, b = chunk.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(chunk))
    return out


def build(sym, out_dir, cache_dir, wanted_global_pages=None, quiet=True):
    mvts = load_boundaries()[str(sym)]
    os.makedirs(out_dir, exist_ok=True)
    written, global_page = [], 0

    for mvt_idx, starts in enumerate(mvts, start=1):
        # which pages of this movement do we actually need?
        pages_here = []
        for i in range(len(starts)):
            global_page += 1
            if wanted_global_pages is None or global_page in wanted_global_pages:
                pages_here.append(i)
        if not pages_here:
            continue

        print(f"symphony {sym}, movement {mvt_idx}: {len(pages_here)} of {len(starts)} pages")
        md2 = fetch_md2(sym, mvt_idx, cache_dir)
        mvt_xml = os.path.join(cache_dir, f"{sym}_{mvt_idx}.musicxml")

        if not os.path.exists(mvt_xml):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf if quiet else sys.stdout):
                score = parse_md2(md2)      # the parser prints a line per input line
                from music21.musicxml.m21ToXml import ScoreExporter
                from xml.etree.ElementTree import tostring
                xml = tostring(ScoreExporter(score).parse(), encoding='unicode')
            with open(mvt_xml, "w", encoding="utf-8") as f:
                f.write(xml)

        # Re-read the movement from MusicXML before slicing. This round trip is load-bearing, not
        # housekeeping: slice_score() carries instrument names over with part.getInstrument(), and
        # on the freshly parsed score that Instrument has no name yet -- every <part-name> would
        # come out empty. Reading the MusicXML back populates it. The original two-script workflow
        # got this for free by writing a file and then re-parsing it.
        from music21 import converter
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf if quiet else sys.stdout):
            score = converter.parse(mvt_xml)
        n_parts = len(score.parts)
        n_meas = len(list(score.parts[0].getElementsByClass('Measure'))) if n_parts else 0
        print(f"  parsed: {n_parts} parts, {n_meas} measures")

        for i in pages_here:
            start = starts[i]
            end = (starts[i + 1] - 1) if i + 1 < len(starts) else None
            seg = slice_score(score, start, end)
            path = os.path.join(out_dir, f"{sym}_{mvt_idx}_{i + 1}.musicxml")
            seg.write('musicxml', path)
            written.append(path)
            print(f"  page {i + 1:>3} (measures {start}-{end if end else 'end'}) -> {os.path.basename(path)}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symphony", type=int, choices=range(1, 9),
                    help="which symphony (1-8)")
    ap.add_argument("--all", action="store_true", help="all 8 symphonies, 1062 pages")
    ap.add_argument("--pages", default=None,
                    help="page numbers within the symphony, counted across movements as stage 3 "
                         "counts them, e.g. 1-10 or 1,4,7-9. Default: every page.")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sample", "gt", "musicxml"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "musedata_cache"),
                    help="where the downloaded .md2 files are kept")
    ap.add_argument("--verbose", action="store_true", help="show the parser's per-line output")
    args = ap.parse_args()

    if not args.all and args.symphony is None:
        ap.error("give --symphony N or --all")
    if args.all and args.pages:
        ap.error("--pages applies to one symphony; drop --all")

    syms = range(1, 9) if args.all else [args.symphony]
    wanted = parse_pages(args.pages)
    total = []
    for s in syms:
        total += build(s, args.out, args.cache, wanted, quiet=not args.verbose)
    print(f"\n{len(total)} reference files in {args.out}")
    print("Source: CCARH MuseData (https://bitbucket.org/musedata/beethoven) -- "
          "acknowledge CCARH if you redistribute anything derived from it.")


if __name__ == "__main__":
    main()
