#!/usr/bin/env python3

import argparse
import os

from bs4 import BeautifulSoup

base_scripts = """
<script type="text/javascript" id="documentation_options" data-url_root="./"
src="/js/documentation_options.js"></script>
<script type="text/javascript" src="/js/jquery.js"></script>
<script type="text/javascript" src="/js/underscore.js"></script>
<script type="text/javascript" src="/js/doctools.js"></script>
<script type="text/javascript" src="/js/language_data.js"></script>
<script type="text/javascript" src="/js/searchtools.js"></script>
"""  # noqa: E501

search_js_scripts = """
<script type="text/javascript">
    jQuery(function() { Search.loadIndex("/js/searchindex.js"); });
</script>

<script type="text/javascript" id="searchindexloader"></script>
"""

katex_scripts = """
<script src="https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/contrib/auto-render.min.js"></script>
<script src="/js/katex_autorenderer.js"></script>
<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.css" />
<link rel="stylesheet" type="text/css" href="/css/katex-math.css" />
"""  # noqa: E501


def parse_sphinx(input_dir, output_dir):
    for cur, _, files in os.walk(input_dir):
        for fname in files:
            if fname.endswith(".html"):
                with open(os.path.join(cur, fname), "r") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    doc = soup.find("div", {"class": "document"})
                    wrapped_doc = doc.wrap(
                        soup.new_tag("div", **{"class": "sphinx wrapper"})
                    )
                # add scripts that sphinx pages need
                if fname == "search.html":
                    out = (
                        base_scripts
                        + search_js_scripts
                        + katex_scripts
                        + str(wrapped_doc)
                    )
                else:
                    out = base_scripts + katex_scripts + str(wrapped_doc)
                output_path = os.path.join(output_dir, os.path.relpath(cur, input_dir))
                os.makedirs(output_path, exist_ok=True)
                with open(os.path.join(output_path, fname), "w") as fout:
                    fout.write(out)

    # update reference in JS file
    with open(os.path.join(input_dir, "_static/searchtools.js"), "r") as js_file:
        js = js_file.read()
    js = js.replace(
        "DOCUMENTATION_OPTIONS.URL_ROOT + '_sources/'", "'_sphinx-sources/'"
    )
    with open(os.path.join(input_dir, "_static/searchtools.js"), "w") as js_file:
        js_file.write(js)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip HTML body from Sphinx docs.")
    parser.add_argument(
        "-i",
        "--input_dir",
        metavar="path",
        required=True,
        help="Input directory for Sphinx HTML.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        metavar="path",
        required=True,
        help="Output directory in Docusaurus.",
    )
    args = parser.parse_args()
    parse_sphinx(args.input_dir, args.output_dir)
