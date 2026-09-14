# Data

This repository ships **no score images and no ground truth**. Both derive from third-party
editions that are not ours to redistribute, so you obtain them yourself and place them here.

Everything below uses Beethoven Symphony No. 1 as the worked example, under the sheet name
`Bee_1_challenge`, which is what `scripts/run_sample.sh` expects.

## Expected layout

```
data/sample/
  images/                    Bee_1_challenge_001.png ... (3-digit, zero-padded)
  gt/
    musicxml/                1_1_1.musicxml ...      {symphony}_{movement}_{page}.musicxml
    metadata/                Bee_1_challenge_001.csv ... + Bee_1_challenge.json
    staffinfo/               Bee_1_challenge_001.json ...
```

Only `images/` is needed to run the pipeline. The `gt/` trees are needed only to score it:
`gt/musicxml/` for TEDn and OMR-NED, `gt/metadata/` for the stage-1 metadata metrics.

Of the three, only `gt/musicxml/` can be rebuilt automatically — see section 2. The images you
obtain yourself (section 1), and the staff layout annotations are ours rather than a public source
(section 3).

## 1. Page images

The paper uses the Eulenburg edition of the Beethoven symphonies (ed. Max Unger, Ernst Eulenburg,
Leipzig, n.d. [1938]), obtained through [IMSLP](https://imslp.org). Ten pages is enough to exercise the whole pipeline, which is what `scripts/run_sample.sh` runs.
Download the full-score PDF for the first movement and render it to PNG at roughly 2550x4200, one
file per page:

```bash
pdftoppm -png -r 300 score.pdf data/sample/images/Bee_1_challenge
```

Check IMSLP's copyright tagging for the specific file in your jurisdiction before using it.

Nothing in the pipeline is specific to this edition — any orchestral full-score scan works. The
sheet name is just a directory name.

## 2. Reference MusicXML (for TEDn / OMR-NED)

**This one is scripted.** `tools/build_gt.py` fetches the MuseData source, converts it, and cuts it
into pages for you:

```bash
# the ten pages scripts/run_sample.sh uses
python tools/build_gt.py --symphony 1 --pages 1-10

# a whole symphony, or all 1062 pages of symphonies 1-8
python tools/build_gt.py --symphony 1
python tools/build_gt.py --all
```

Output lands in `data/sample/gt/musicxml/` as `{symphony}_{movement}_{page}.musicxml`. Downloaded
`.md2` files and the per-movement MusicXML are cached under `data/musedata_cache/`, so re-running is
offline and cheap. Regenerating symphony 1 pages 1-10 takes about a minute.

The source is *The Nine Symphonies of Beethoven: A Digital Edition* (Sapp, Anthony, Bennion,
Correia, Hewlett, Selfridge-Field, Kornstadt), published by the
[Center for Computer Assisted Research in the Humanities (CCARH)](https://www.ccarh.org/publications/beethoven-symphonies/),
Stanford, in the [MuseData](https://www.musedata.org) format, and readable at
<https://bitbucket.org/musedata/beethoven>. It carries a CCARH copyright notice and no open
license, which is why this repository fetches it instead of redistributing it. **If you repost
anything derived from it, acknowledge CCARH as the original owner**, along with the editors named
on the title page.

Why a dedicated converter: the usual routes (`music21`'s MuseData reader, or the
`musedata2hum`/`hum2xml` chain) introduce numerous rhythmic errors. `tools/md22musicxml.py` parses
MuseData stage-2 directly, keeping instrument layout, notes and beams and dropping articulation,
dynamics and slurs; the same symbol classes are stripped from every system's predictions before
scoring, so the comparison stays fair.

Page boundaries come from `tools/page_boundaries.json` — the measure at which each page of the
Eulenburg print begins, which is what makes one page of reference comparable to one page of
prediction. That table is specific to the Eulenburg edition; a different edition needs its own.

If you would rather supply your own reference, any per-page MusicXML works, as long as it covers the
same pages as `images/`. `stage3_eval/tedn/make_manifest.py` pairs reference and prediction by page
number.

## 3. Staff layout metadata (for the stage-1 metrics)

One CSV per page, one row per staff, in reading order. The columns the evaluator uses:

| column | meaning |
|---|---|
| `ins1`, `ins2`, `ins3` | instrument(s) on this staff; `ins2`/`ins3` are non-empty only when one staff carries several (e.g. `Violoncello e Basso`) |
| `part` | part number within the instrument (Violin I vs II), optional |
| `tone` | transposition, optional |

plus `{sheet}.json`, the piece's instrument list in score order. `stage3_eval/metadata_metrics.py`
scores predictions against these; `--with-part-tone` additionally requires `part` and `tone` to
match.

You can also skip this: without `gt/metadata/` the pipeline still runs end to end, you just do not
get the Table 1 numbers.
