# Expected outputs

Schema-only snapshots of each stage's output, captured from a fresh
`examples/run.sh` run. A reader can diff their own outputs' schemas
against these JSON files to confirm column shapes match.

These are reference only — actual data values will differ because
the synthetic example uses a fixed seed but downstream stages
(embedding, training) are not bitwise deterministic.
