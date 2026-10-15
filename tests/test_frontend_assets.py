"""
The UI is plain ES modules with no build step, so a single wrong import path blanks the page.
These checks catch that (and accidental CDN links) without a browser.
"""

import os
import re
import shutil
import subprocess

import pytest

DIST = os.path.join(os.path.dirname(__file__), "..", "easyml", "unified_frontend", "dist")
STATIC = os.path.join(DIST, "static")
JS_ROOT = os.path.join(STATIC, "js")

IMPORT_RE = re.compile(r"""(?:import|export)\s+(?:[^'"]*?\sfrom\s+)?['"](\.{1,2}/[^'"]+)['"]|import\(\s*['"](\.{1,2}/[^'"]+)['"]\s*\)""")
ALLOWED_EXTERNAL = ("https://pypi.org/", "https://pytorch.org/", "https://www.linkedin.com/", "https://github.com/",
                    "https://download.pytorch.org/",
                    "http://127.0.0.1:")  # localhost addresses in copyable code examples


def _read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _own_files(root, ext):
    for folder, _, files in os.walk(root):
        if "vendor" in folder.split(os.sep):
            continue
        for name in files:
            if name.endswith(ext):
                yield os.path.join(folder, name)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js for the syntax check")
def test_modules_have_valid_syntax():
    # A syntax error in any module stops the whole app from mounting.
    for path in _own_files(JS_ROOT, ".js"):
        result = subprocess.run(["node", "--input-type=module", "--check"], input=_read(path),
                                capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, f"{os.path.relpath(path, JS_ROOT)}: {result.stderr.strip()[:500]}"


def test_index_references_existing_static_files():
    html = _read(os.path.join(DIST, "index.html")).replace("__VUE_BUILD__", "vue.global.prod.js")
    refs = re.findall(r'(?:src|href)="(/static/[^"?]+)', html)
    assert refs, "index.html should reference bundled assets"
    for ref in refs:
        assert os.path.isfile(os.path.join(DIST, ref.lstrip("/").replace("/", os.sep))), ref
    assert os.path.isfile(os.path.join(STATIC, "vendor", "vue.global.js")), "dev build used by EASYML_FRONTEND_DEBUG"


def test_every_module_import_resolves():
    seen, pending = set(), [os.path.join(JS_ROOT, "main.js")]
    while pending:
        path = os.path.normpath(pending.pop())
        if path in seen:
            continue
        seen.add(path)
        assert os.path.isfile(path), f"missing module {path}"
        for static_import, dynamic_import in IMPORT_RE.findall(_read(path)):
            target = static_import or dynamic_import
            pending.append(os.path.join(os.path.dirname(path), target.replace("/", os.sep)))
    # Every module in js/ should be reachable from main.js (no dead files shipped).
    all_modules = {os.path.normpath(p) for p in _own_files(JS_ROOT, ".js")}
    assert all_modules <= seen, f"unreachable modules: {sorted(all_modules - seen)}"


def test_css_urls_resolve():
    for css in list(_own_files(STATIC, ".css")) + [os.path.join(STATIC, "vendor", "fonts.css")]:
        for url in re.findall(r"url\(['\"]?([^'\")]+)['\"]?\)", _read(css)):
            if url.startswith("data:"):
                continue
            assert os.path.isfile(os.path.join(os.path.dirname(css), url)), f"{css}: {url}"


@pytest.mark.parametrize("ext", [".js", ".css", ".html"])
def test_no_remote_assets(ext):
    files = list(_own_files(DIST, ext))
    for path in files:
        text = _read(path)
        for url in re.findall(r"""(?:src=|@import\s+|url\()['"]?(https?://[^'")\s]+)""", text):
            pytest.fail(f"{path} loads a remote asset: {url}")
        for url in re.findall(r"""https?://[^\s'"<>)`]+""", text):
            assert url.startswith(ALLOWED_EXTERNAL), f"{path} links to an unexpected host: {url}"


def test_templates_do_not_use_js_interpolation():
    # Component templates are template literals; `${` inside them would be evaluated as JavaScript.
    for path in _own_files(JS_ROOT, ".js"):
        for template in re.findall(r"template:\s*/\* html \*/\s*`(.*?)`", _read(path), flags=re.S):
            assert "${" not in template, path


def test_index_and_modules_are_served(client):
    html = client.get("/").get_data(as_text=True)
    assert 'type="module"' in html and "/static/js/main.js?v=" in html
    response = client.get("/static/js/main.js")
    assert response.status_code == 200 and response.mimetype == "text/javascript"
    assert client.get("/static/css/tokens.css").mimetype == "text/css"
