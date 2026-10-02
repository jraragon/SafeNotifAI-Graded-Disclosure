# Data provenance

This repository is the reproducibility package accompanying the manuscript
"Graded Disclosure for Privacy-Aware Parental Supervision: A Decision Framework with an Interpretable Fuzzy Policy".

## OffendES

The experimental evaluation is based on OffendES, an external research corpus
that was not created by the authors of this repository.

Original project repository:
https://github.com/fmplaza/OffendES

Original publication:
F. M. Plaza-del-Arco, M. D. Molina-González, L. A. Ureña-López, and
M. T. Martín-Valdivia, "OffendES: A New Corpus in Spanish for Offensive
Language Research", Proceedings of RANLP 2021.

Publication record:
https://aclanthology.org/2021.ranlp-1.123/

The original OffendES training, development, and test TSV files were used
during development and verification of the experimental pipeline. These source
files are external resources and are intentionally not redistributed in this
repository. Users who need the original corpus should obtain it from its
original distribution source and comply with the terms applicable there.

The directory `datos/offendes/` is excluded from version control to prevent
accidental redistribution of local source-data copies.

## Cached experimental signal matrix

The file

    datos/senales_offendes.csv

contains the cached signal-level representation used by the decision-policy
experiments reported in the manuscript.

The cached representation separates evaluation of the graded-disclosure
policies from repeated execution of the upstream NLP models. The main
decision-policy analyses can therefore be inspected and reproduced directly
from this artifact.

This distinction is important: the repository supports reproducibility of the
paper's policy-level experiments without claiming to redistribute the original
OffendES corpus.

## Lexical resource

The file

    datos/lexico_malsonante.csv

is a supporting lexical resource used by the signal-processing pipeline
associated with the experimental artifact.

## SafeNotifAI provenance

The work builds on SafeNotifAI, originally developed by Rocío Pedraz Galán.
The original SafeNotifAI source package used in the development of this work
is distributed under the GNU General Public License v3.0.

This reproducibility repository is not a complete redistribution of the
SafeNotifAI application. It contains the experimental code, cached
representation, and analysis artifacts required to inspect and reproduce the
graded-disclosure experiments described in the accompanying manuscript.

## Reproducibility boundary

The reproducibility target of this repository is the experimental evidence
reported in the manuscript from the cached signal representation onward.

Rebuilding the upstream NLP signals from the original text corpus is a
separate, provenance-dependent stage and is not required to reproduce the
decision-policy experiments from `datos/senales_offendes.csv`.

External datasets and resources retain their original authorship and applicable
terms and are not relicensed by this repository.
