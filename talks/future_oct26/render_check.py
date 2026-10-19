"""render_check.py <deck.html> <outdir> [i,j,..] -- render the deck headless and crop each .frame to s<NN>.png (positions read from the DOM)."""
import json, re, subprocess, sys
from pathlib import Path
from PIL import Image
html, out = Path(sys.argv[1]), Path(sys.argv[2])
only = set(sys.argv[3].split(',')) if len(sys.argv) > 3 else None
out.mkdir(exist_ok=True)
for f in out.glob('*.png'):
    f.unlink()
W = 1432
src = html.read_text()
probe = src.replace('</body>', '<script>window.addEventListener("load",()=>{setTimeout(()=>{const r=[...document.querySelectorAll(".frame")].map(f=>{const b=f.getBoundingClientRect();return [b.top+scrollY,b.height]});document.body.setAttribute("data-rects",JSON.stringify(r));},500)});</script></body>')
tmp = out / 'probe.html'
tmp.write_text(probe)
dom = subprocess.run(['google-chrome', '--headless=new', '--disable-gpu', '--virtual-time-budget=6000', f'--window-size={W},1000',
                      '--dump-dom', f'file://{tmp}'], capture_output=True, text=True).stdout
rects = json.loads(re.search(r'data-rects="([^"]*)"', dom).group(1).replace('&quot;', '"'))
H = int(rects[-1][0] + rects[-1][1] + 50)
full = out / 'full.png'
subprocess.run(['google-chrome', '--headless=new', '--disable-gpu', '--hide-scrollbars', '--virtual-time-budget=6000',
                f'--window-size={W},{H}', f'--screenshot={full}', f'file://{tmp}'], capture_output=True)
im = Image.open(full).convert('RGB')
for k, (y, h) in enumerate(rects):
    if only and str(k) not in only:
        continue
    im.crop((0, int(y), W, int(y + h))).save(out / f's{k:02d}.png')
full.unlink()
print(len(rects), 'frames')
