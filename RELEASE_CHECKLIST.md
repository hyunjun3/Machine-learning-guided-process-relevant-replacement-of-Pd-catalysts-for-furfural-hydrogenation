# Public-release checklist

This repository remains private until every item below has been approved by the authors.

## Scientific content

- Confirm that all manuscript and Supplementary Figure numbers match the accepted files.
- Re-run the automated tests and every figure script in the documented environment.
- Visually compare the regenerated figures with the accepted manuscript and Supplementary Information.
- Confirm that `dataset/` is the approved public dataset snapshot.

## Release metadata

- Select and install the approved code license and, if needed, separate data terms.
- Replace version `0.1.0` in `CITATION.cff` with `1.0.0`.

## Confidentiality and repository hygiene

- Confirm that no manuscript PDF, Supplementary Information DOCX, submission correspondence, reviewer material, credentials, or internal network paths are tracked.
- Confirm that all required Git LFS objects can be fetched from a fresh clone.
- Confirm that GitHub Actions passes on the release commit.

## Public release

1. Create and push the signed or annotated tag `v1.0.0`.
2. Create a GitHub release from `v1.0.0` with a concise reproducibility summary.
3. If configured, archive the release with Zenodo or Figshare and add the DOI to the repository metadata.
4. Change repository visibility to public only after the preceding checks are complete.
