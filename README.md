# ML Learning Notes

A personal blog for machine-learning study notes, written in **Markdown** and published as a static website with **[Hugo](https://gohugo.io/)** + the **[PaperMod](https://github.com/adityatelange/hugo-PaperMod)** theme.

🌐 **Live site:** https://omegazhang3.github.io/ml-blog/

Posts support **LaTeX math** (KaTeX), **syntax-highlighted code** with copy buttons, full-text **search**, **tags**, a **table of contents**, and automatic **light/dark mode**.

---

## How the system works

```
            write Markdown                 git push                GitHub Actions
  You  ─────────────────────▶  content/*.md  ─────────▶  GitHub repo  ─────────▶  Hugo builds  ─────────▶  GitHub Pages
         (preview locally                                 (master)                 (0.125.7)              https://omegazhang3.github.io/ml-blog/
          at localhost:1313)
```

1. You write posts as Markdown files under `content/posts/`.
2. A local Hugo dev server renders them instantly for preview.
3. On `git push` to `master`, a GitHub Actions workflow builds the site and deploys it to GitHub Pages — no manual build/upload needed.

---

## Repository layout

```
ml-blog/
├── bin/hugo                     # Pinned Hugo 0.125.7 binary (git-ignored, not committed)
├── hugo.toml                    # Site configuration
├── archetypes/default.md        # Template used by `hugo new content`
├── content/
│   ├── posts/                   # ← Your blog posts live here
│   │   ├── linear-regression.md
│   │   └── Softmax.md
│   └── search.md                # The search page
├── layouts/
│   └── partials/
│       └── extend_head.html     # Injects KaTeX (math rendering) on every page
├── themes/PaperMod/             # Theme (git submodule, pinned at tag v8.0)
├── .github/workflows/hugo.yml   # GitHub Actions: build + deploy to Pages
├── .gitignore                   # Ignores public/, resources/, bin/, lock file
└── .gitmodules                  # Declares the PaperMod submodule
```

Generated output (`public/`, `resources/`) and the `bin/` binary are **not** committed — they are rebuilt/re-downloaded as needed.

---

## ⚠️ Important: always use `./bin/hugo`

This project pins **Hugo 0.125.7 (extended)** in `./bin/hugo`. **Do not use the system-wide `hugo`** (which may be a newer, incompatible version).

> **Why:** PaperMod v8.0 uses the classic layout structure (`layouts/_default/`, `layouts/partials/`) and the old `{{ template "partials/..." }}` syntax. Hugo ≥ 0.146 changed layout lookup rules and removed that syntax, which breaks the theme ("no layout file found" errors). Hugo 0.125.7 is inside PaperMod v8.0's tested range (min 0.112.4). The CI workflow pins the same version so local and deployed builds match.

The pinned binary is git-ignored. If it's missing (e.g. fresh clone), re-download it:

```bash
mkdir -p bin && cd bin
curl -sSL -o hugo.tar.gz \
  https://github.com/gohugoio/hugo/releases/download/v0.125.7/hugo_extended_0.125.7_linux-amd64.tar.gz
tar xzf hugo.tar.gz hugo && rm hugo.tar.gz
cd ..
```

---

## Daily workflow

### 1. Start the local preview server

```bash
cd ml-blog
./bin/hugo server --buildDrafts
```

Open **http://localhost:1313/**. The browser auto-reloads whenever you save a Markdown file. `--buildDrafts` also shows posts marked `draft: true`.

### 2. Create a new post

```bash
./bin/hugo new content posts/my-topic.md
```

### 3. Write it

Each post is Markdown with a front-matter header:

```markdown
---
title: "深入理解交叉熵损失"
date: 2026-10-04
draft: false          # false = published; true = only visible with --buildDrafts
math: true            # enable LaTeX math ($...$ and $$...$$)
tags: ["机器学习", "损失函数"]
summary: "Cross-entropy, Softmax's best partner."
---

Inline math: the learning rate is $\alpha$.

$$ L = -\sum_i y_i \log \hat{y}_i $$

```python
import torch.nn.functional as F
loss = F.cross_entropy(logits, labels)
```
```

### 4. Publish

```bash
git add -A
git commit -m "New post: cross-entropy"
git push                 # → GitHub Actions auto-builds and deploys (~1–2 min)
```

Watch the deploy:

```bash
gh run watch   # or see the Actions tab on GitHub
```

---

## Front-matter reference

| Field     | Purpose                                                        |
|-----------|----------------------------------------------------------------|
| `title`   | Post title (shown in listings and the browser tab)             |
| `date`    | Publish date; controls ordering                                |
| `draft`   | `true` hides the post unless `--buildDrafts` is passed         |
| `math`    | `true` enables KaTeX math rendering for `$...$` / `$$...$$`     |
| `tags`    | List of tags; grouped on the Tags page                         |
| `summary` | Preview text shown on the home/list pages                      |

---

## Features & how they're wired

| Feature                 | Implementation                                                              |
|-------------------------|-----------------------------------------------------------------------------|
| **LaTeX math**          | KaTeX loaded via `layouts/partials/extend_head.html`; enabled per-post with `math: true` |
| **Code highlighting**   | Hugo's built-in Chroma highlighter (`[markup.highlight]` in `hugo.toml`), `github-dark` style |
| **Copy buttons**        | PaperMod `ShowCodeCopyButtons = true`                                       |
| **Search**              | PaperMod + Fuse.js; needs `home = [..., "JSON"]` in `[outputs]`             |
| **Tags**                | Hugo taxonomies + the `/tags/` menu entry                                   |
| **Table of contents**   | PaperMod `ShowToc = true`                                                   |
| **Light/dark mode**     | PaperMod `defaultTheme = "auto"`                                            |

All configured in `hugo.toml`.

---

## Deployment (GitHub Pages)

- **Workflow:** `.github/workflows/hugo.yml`
- **Trigger:** every push to `master` (and manual `workflow_dispatch`)
- **Steps:** install Hugo 0.125.7 → checkout with submodules → build with `hugo --gc --minify` → upload artifact → deploy to Pages
- **Pages source:** GitHub Actions (not a branch)

The `baseURL` in `hugo.toml` is set to `https://omegazhang3.github.io/ml-blog/` so links resolve correctly on the live site.

---

## Fresh-clone setup

```bash
git clone --recurse-submodules git@github.com:omegazhang3/ml-blog.git
cd ml-blog
# re-download the pinned Hugo binary (see section above)
./bin/hugo server --buildDrafts
```

If you forgot `--recurse-submodules`, pull the theme afterward:

```bash
git submodule update --init --recursive
```

---

## Tech stack

| Component     | Version / Choice                              |
|---------------|-----------------------------------------------|
| Site generator| Hugo **0.125.7** extended (pinned)            |
| Theme         | PaperMod **v8.0** (git submodule)             |
| Math          | KaTeX 0.16.11 (CDN)                           |
| Hosting       | GitHub Pages via GitHub Actions               |
| Content       | Markdown                                       |
