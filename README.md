# Graded Disclosure for Privacy-Aware Parental Supervision

Reproducibility package for the manuscript:

**Graded Disclosure for Privacy-Aware Parental Supervision: A Decision Framework with an Interpretable Fuzzy Policy**

This repository contains the experimental code, cached signal representation,
preregistration material, and reported outputs supporting the analyses in the
manuscript.

## Scope

The paper studies parental supervision as a graded-disclosure decision problem.
Rather than treating supervision as a binary notification decision, the
experimental framework represents disclosure through ordered decision levels
and evaluates protection jointly with information exposure.

An interpretable fuzzy inference system (FIS) is studied as one policy
instantiation and is compared with alternative score-level policies under the
experimental protocol described in the manuscript.

## Repository contents

- `01_preregistration/` ? preregistration material.
- `datos/` ? cached signal matrix and supporting data resources.
- `salidas/01_preregistered/` ? preregistered comparison.
- `salidas/02_frontier/` ? disclosure-budget frontier.
- `salidas/03_ablation/` ? signal-zeroing ablation.
- `salidas/04_sensitivity/` ? sensitivity analysis.
- `salidas/05_cross_platform/` ? cross-platform robustness analysis.
- `salidas/06_objective_alignment/` ? post-hoc objective-aligned linear-policy analysis.
- `salidas/07_policy_structure/` ? policy-level distributions and structural inspection.

## Environment

The experiments were independently checked using Python 3.11.

Create a virtual environment:

    python -m venv .venv

On Windows PowerShell, activate it and install the dependencies:

    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

## Reproduction

Main preregistered comparison:

    python decisivo.py

Frontier, ablation, sensitivity, and cross-platform analyses:

    python experimentos_finales.py

Policy-level distributions and learned coefficients:

    python 1_niveles_y_coeficientes.py

Objective-aligned linear-policy analysis:

    python 3_lineal_ajustado_mismo_objetivo.py

Exploratory reconstruction of the high-protection frontier point:

    python 4_punto_frontera_0769.py

## Reported results

Canonical outputs are stored under `salidas/`. These directories contain the
results distributed with the paper rather than a history of intermediate
executions.

The objective-aligned linear analysis is post hoc. It is included to clarify
the interpretation of comparisons between aggregation families under different
optimization procedures.

## Data provenance

The evaluation uses OffendES, an external research resource. The cached signal
representation used by the decision-policy experiments is included as
`datos/senales_offendes.csv`.

See `DATA_PROVENANCE.md` for the distinction between external source data,
cached experimental signals, and the original SafeNotifAI project.

## Reproducibility boundary

The cached signal matrix permits reproduction of the decision-policy experiments
without repeatedly executing the upstream NLP models.

Byte-level differences in serialized files may occur because of encoding or
floating-point serialization. During preparation of this artifact, the reported
CSV outputs were reproduced with zero numerical difference, and the
objective-alignment JSON was reproduced with zero semantic differences.

## License

This repository is distributed under the GNU General Public License v3.0.
See `LICENSE`.

External datasets and resources retain their respective original terms and are
not relicensed by this repository.

## Citation

Citation metadata are provided in `CITATION.cff` and will be updated with the final
publication and archival DOI when available.
