// Injected into a copy of the readout-stack 3D note (dylan-cern-site pages/notes/mx17-readout-stack-3d.html) by
// make_readout_stack_3d.py: hides the page around the viewer, white scene, "Peeled" preset, and writes the screen
// position of every layer label into <pre id="labjson"> (read back with --dump-dom).  ?shot=1 hides the labels.
(function () {
  document.documentElement.setAttribute('data-theme', 'light');
  const shot = /shot=1/.test(location.search);
  const q = new URLSearchParams(location.search), VW = +(q.get('w') || 760), VH = +(q.get('h') || 560);
  const css = document.createElement('style');
  css.textContent = `:root{--scene:#ffffff !important;--bg:#ffffff !important}
    header,.controls,aside,.pick,.legend{display:none !important}
    .app{display:block !important;padding:0 !important;height:100vh !important}
    .viewer-col{height:auto !important;display:block !important}
    .viewer{border:0 !important;border-radius:0 !important;width:${VW}px !important;height:${VH}px !important;min-height:0 !important;flex:none !important}
    ${shot ? '.lbl{visibility:hidden !important}' : ''}`;
  document.head.appendChild(css);
  window.addEventListener('load', () => {
    setTimeout(() => {
      document.getElementById('v-pap').click();
      setTimeout(() => {
        const vr = document.getElementById('viewer').getBoundingClientRect();
        const out = [...document.querySelectorAll('.lbl')].map(s => {
          const r = s.getBoundingClientRect();
          return { text: s.textContent, cls: s.className.replace('lbl', '').trim(), x: r.left - vr.left, y: r.top + r.height / 2 - vr.top };
        });
        const pre = document.createElement('pre'); pre.id = 'labjson';
        pre.textContent = JSON.stringify({ w: vr.width, h: vr.height, x0: vr.left, y0: vr.top, labels: out });
        document.body.appendChild(pre);
      }, 600);
    }, 300);
  });
})();
