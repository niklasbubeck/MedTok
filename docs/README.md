# MedTok Project Page

This folder is the source for the [GitHub Pages](https://pages.github.com/) project site.

## Setup Instructions

1. **Enable GitHub Pages** (one-time):
   - Go to your repo: `https://github.com/YOUR_USERNAME/MedTok`
   - Click **Settings** → **Pages** (under "Code and automation")
   - Under "Build and deployment" → "Source": select **Deploy from a branch**
   - Branch: `main` (or your default branch)
   - Folder: **/docs**
   - Click **Save**

2. **Update links** in `index.html`:
   - Replace `YOUR_USERNAME` in the Code link with your GitHub username
   - Add the arXiv link when the paper is published

3. **Deploy**: Push changes to `main`. The site will be live at:
   ```
   https://YOUR_USERNAME.github.io/MedTok/
   ```

## Customization

- Edit `index.html` to update content, add figures, or change styling
- Add more pages (e.g., `results.html`, `citation.html`) and link to them
- Copy figures from `latex/figures/` into `docs/figures/` and reference them
