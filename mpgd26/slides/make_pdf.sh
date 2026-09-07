#!/usr/bin/env bash
# Regenerate mpgd26_talk_draft.pdf from index.html.
#
# Not a plain "print index.html" — Chrome's headless print pipeline (at least
# on chrome 151) blanks whichever slide is first whenever the print output has
# more than one page, regardless of that slide's own content or CSS (confirmed
# by swapping slide order: whichever slide is first goes blank). Printing each
# slide to its own single-page PDF never hits that multi-page code path, so
# instead this renders every slide separately and merges the results.
set -euo pipefail
cd "$(dirname "$0")"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ln -s "$(pwd)/assets" "$WORK/assets"

python3 - "$WORK" << 'PYEOF'
import re, sys
work = sys.argv[1]
with open('index.html') as f:
    c = f.read()
head = c[:c.index('<div class="stage">')]
sections = re.findall(r'<section class="slide.*?</section>', c, flags=re.S)
# Each slide prints alone, so the CSS slide counter would read "1" on every
# page; pre-load it with this slide's position in the full deck instead. An
# overlay build (.bstart + .bcont, see the <style> block) is ONE slide printed
# as several pages, numbered 6.1, 6.2, ...: its continuation frames do not
# increment the counter, so they print at the number the build started on. The
# ".n" half is the section's own data-frame attribute and needs nothing here.
s_no = 0
for i, sec in enumerate(sections, start=1):
    cont = 'bcont' in re.match(r'<section class="([^"]*)"', sec).group(1).split()
    if not cont:
        s_no += 1
    doc = (head
           + f'<style>.deck{{counter-reset:slide {s_no if cont else s_no - 1};}}</style>\n'
           + '<div class="stage"><div class="deck">\n' + sec + '\n</div></div>')
    # 3 digits, not 2: past 99 sections (crossed 2026-08-23, adding the
    # section-transition dividers), slide_100.html sorted lexicographically
    # BEFORE slide_11.html..slide_99.html in the shell glob below, and
    # pdfunite merged the pages in that scrambled order with no error.
    with open(f'{work}/slide_{i:03d}.html', 'w') as f:
        f.write(doc)
print(f'{len(sections)} slides', file=sys.stderr)
PYEOF

pdfs=()
for html in "$WORK"/slide_*.html; do
  pdf="${html%.html}.pdf"
  google-chrome --headless --disable-gpu --no-sandbox \
    --print-to-pdf="$pdf" --no-pdf-header-footer --virtual-time-budget=3000 \
    "file://$html" 2>/dev/null
  pdfs+=("$pdf")
done

pdfunite "${pdfs[@]}" mpgd26_talk_draft.pdf
echo "wrote $(pwd)/mpgd26_talk_draft.pdf ($(pdfinfo mpgd26_talk_draft.pdf | awk '/^Pages/{print $2}') pages)"
