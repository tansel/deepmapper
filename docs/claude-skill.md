# Using DeepMapper with Claude

The repository ships a Claude Code skill at
`.claude/skills/deepmapper/SKILL.md`. When you open this repo in
[Claude Code](https://claude.com/claude-code), the skill is available and Claude can
drive a full DeepMapper analysis for you: load the data, run a fast linear check, run
the pipeline, read the gene chord, and reproduce an example figure.

## How it works

A Claude skill is a Markdown file with a short description in its front matter. Claude
reads the description to decide when the skill applies, then follows the workflow in
the body. The DeepMapper skill triggers when you ask Claude to:

- analyse a single-cell or omics matrix without dimension reduction,
- find the genes (or gene chords) that separate cell states,
- attribute a classifier back to named features,
- or reproduce one of the example analysis figures.

## Use it

1. Install Claude Code and open this repository in it.
2. Ask in plain language, for example:
   - "Analyse this h5ad with DeepMapper and tell me which genes separate the states."
   - "Find the gene chord for CD4 memory in this matrix."
   - "Reproduce the Figure 1 analysis from the bench scripts."
3. Claude loads the skill and walks the workflow: it checks the install, loads your
   data with `pydeepmapper.io`, runs a deterministic linear baseline first, then runs
   `pydeepmapper.runner.run`, and reads `findings.ranking(...)`.

You can also invoke it explicitly by name with `/deepmapper` if your Claude Code
setup lists it.

## What the skill will and will not do

The skill follows the same honest-analysis rules:

- It keeps every feature. It will not select highly variable genes or run PCA first.
- It runs the fast, deterministic linear baseline before the CNN, and expects the two
  rankings to roughly agree.
- It reports a finding as a module (a set of interchangeable genes), not a single
  fixed gene list, and it will not over-read a single run.
- It uses `cnn_small` by default and avoids `resnet18` on sparse single-cell data.

## Extending the skill

The skill is plain Markdown. Edit `.claude/skills/deepmapper/SKILL.md` to add your own
datasets, default configs, or house rules. Keep the front-matter `description` precise,
since that is what Claude matches against your request.
