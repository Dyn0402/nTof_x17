"""Build the detector-spec page: spec.py + template.html → out/mx17-detector-spec.html.

    python3 docs/detector_spec/build_page.py            # build
    python3 docs/detector_spec/build_page.py --publish  # build, then add it to the site's notes and deploy

Comments go to the Apps Script shared with the run-budget page (x17_facility_search/budget/feedback_apps_script.gs),
with page="detector"; set DETECTOR_FEEDBACK_ENDPOINT to point the page at a mock backend for testing.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec  # noqa: E402

SITE = Path.home() / "PycharmProjects" / "dylan-cern-site"


def check():
    ids = {e["id"] for e in spec.ELEMENTS}
    for e in spec.ELEMENTS:
        for p in e["params"]:
            assert p["st"] in spec.STATUS, (e["id"], p)
        for q in e.get("questions", []):
            assert q in {q["id"] for q in spec.QUESTIONS}, (e["id"], q)
    for q in spec.QUESTIONS:
        assert q["el"] in ids, q["id"]
    for row in spec.STACK:
        assert row[0] in ids, row


def build():
    check()
    data = dict(updated=spec.UPDATED, status=spec.STATUS, groups=spec.GROUPS, elements=spec.ELEMENTS,
                questions=spec.QUESTIONS, stack=spec.STACK, glossary=spec.GLOSSARY, chambers=spec.CHAMBERS,
                contact=spec.CONTACT,
                feedback_endpoint=os.environ.get("DETECTOR_FEEDBACK_ENDPOINT", spec.FEEDBACK_ENDPOINT))
    html = (HERE / "template.html").read_text()
    assert "/*__DATA__*/null" in html
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    out = HERE / "out" / f"{spec.SLUG}.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    total = sum(r[1] for r in spec.STACK)
    n = {k: sum(p["st"] == k for e in spec.ELEMENTS for p in e["params"]) for k in spec.STATUS}
    print(f"wrote {out}  ({len(spec.ELEMENTS)} parts, {sum(n.values())} numbers {n}, {len(spec.QUESTIONS)} questions,"
          f" stack {total:.3f} mm)")
    return out


if __name__ == "__main__":
    out = build()
    if "--publish" in sys.argv:
        subprocess.run([sys.executable, str(SITE / "scripts" / "add-note.py"), str(out), "--slug", spec.SLUG, "--force",
                        "--tags", "X17, micromegas, detector, geometry, review", "--deploy"], check=True)
