# ICRA 2027 PaperPlaza dry run

**Checked:** 2026-08-11 UTC  
**Scope:** anonymous contributed-paper initial submission; no paper or author record was created.

## Live requirements and taxonomy

The [official ICRA 2027 call](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
currently states an eight-page total limit, double-anonymous review, a September 15, 2026 11:59
PST paper deadline, and the August 5--September 9 and September 17--22 video windows.  It directs
authors to choose at least three entries from the live
[IEEE RAS ICRA keyword list](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra/icra-keywords/).

The three phrases proposed in the execution plan are not entries in that live list:

| Planned phrase | Exact live match | Recommended live replacement |
| --- | --- | --- |
| Motion Retargeting | no | Simulation and Animation |
| Human and Humanoid Motion Analysis and Synthesis | no | Humanoid Robots |
| Machine Learning for Robot Control | no | Learning and Adaptive Systems |

`Humanoid and Bipedal Locomotion` is a valid alternate if platform/locomotion routing is preferred
over simulation-and-motion routing.  No portal keyword selection has been made; the human owner
must confirm the recommended trio before submission.

## Public PDF compliance test

The current fallback PDF was uploaded to the public
[PaperPlaza PDF Test](https://ras.papercept.net/conferences/scripts/pdftest.pl) as an ICRA 2027
initial contributed paper.  PaperPlaza returned: **“The document may be uploaded. There are no
critical issues.”**

- Input: `/data/robotixx/snmr-research/paper-build/a4-final-fallback/main.pdf`
- Input SHA-256: `e293725b18f1a4398934705753584aa780c795838ee6e5306b31ed51430524b9`
- Reported file size/page count: 99.3 KB / 6 pages, against a 12 MB / 8-page test limit
- PDF version: 1.5, accepted range 1.4--1.7
- Page size: US Letter on all six pages
- Searchability: passed
- Fonts: all embedded, all subset, no Type 3 fonts, no oriental fonts
- Encryption: none
- Other errors or warnings: none
- Margins: passed as non-critical; PaperPlaza noted first-page margin impositions and supplied an
  overlay, which was visually checked with title and body content inside the marked content area

PaperPlaza reprocessed the input to remove bookmarks and annotations.  The returned copy is
`/data/robotixx/snmr-research/paper-build/a6-paperplaza/preprocessed.pdf`, SHA-256
`fb3347e906db34e2e778d73cac42e01fcdf49f9d0082c6f4577c28682f3b5b18`.  Extracted layout text is
identical to the input.  At 96 dpi, five pages render pixel-identically; page 2 differs at only 65
pixels (maximum channel delta 64) in link locations, with no visible content change.  The archived
margin overlay has SHA-256 `1033bca8261b079a5c5b1325a8d171115a0d3373226fee3f9a5c32b221e171c6`.

Local anonymity checks found one `Anonymous ICRA Submission` marker and zero email addresses,
host-local paths, configured git identities, or author metadata fields in the PDF.  The final B3
PDF must be retested because its generated three-seed figure changes the submitted file.

## Submission metadata worksheet

The identity-bearing values below remain out of the repository and PDF.  Complete them directly in
PaperPlaza from the human-owned author record after B3:

- [x] Submission type staged: `Contributed paper`, initial submission
- [x] Title staged: `What Crosses the Boundary? Measuring the Retarget-to-Track Interface for Humanoid Tracking`
- [x] Abstract source staged: use the final active branch rendered from `paper/main.tex`; do not
  copy the current seed-0 fallback after B3
- [ ] Confirm the three live keywords recommended above, or select the documented locomotion alternate
- [ ] Verify every author's single PaperPlaza PIN, legal name, email, and current affiliation
- [ ] Verify author order, corresponding author, and on-site presenter
- [ ] Paste the final title and abstract from the B3 PDF and compare them character-for-character
- [ ] Complete conflicts, subject areas, and reviewer-service declarations requested by the portal
- [ ] Human policy decision: determine whether the agent-assisted language work is only editing and
  grammar or requires the anonymous generative-AI disclosure described by the ICRA 2027 call
- [ ] Human performs the actual submission and records the confirmation ID in the execution report

The public PaperPlaza landing page says first submissions do not require a login, but author PINs
and identity data are still human-controlled.  No author metadata or credentials were entered in
this dry run.
