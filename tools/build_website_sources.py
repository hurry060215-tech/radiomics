"""Render the maintained docs into Sphinx sources without duplicating their text.

No numerical API is imported. Run before sphinx-build. Generated files are local
build products; unknown pre-existing files are never silently overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    'api.md', 'support-matrix.md', 'image-native.md', 'features-native.md',
    'native-extraction.md', 'table-io-native.md', 'learning-native.md',
    'models-native.md', 'preprocessing-native.md', 'evaluation-native.md',
    'clinical-native.md', 'baseline-native.md', 'reliability-native.md',
    'pathology-pixels.md', 'pathology-aggregate.md', 'interop.md',
    'research-native.md', 'cli-workflows-native.md', 'pathology-cli-native.md',
    'technical-report.md', 'radiomics-benchmark-v2-results.md',
    'learning-pathology-benchmark-v2-results.md', 'deployment-benchmark-v2-results.md',
    'python-native.md', 'delivery-index.md',
    'onekey-workflow-gap-audit.md', 'native-performance-diagnosis.md',
    'rust-backend-decision.md', 'scalar-accuracy-audit.md',
    'sample-constructor-optimization.md',
)
LINK = re.compile(r'(?P<head>\!?\[[^\]\n]*\]\()(?P<target>[^)\s]+)(?P<tail>\))')


def source_revision(root):
    """Pin source links to this checkout, never an enclosing repository."""
    if not (root / '.git').exists():
        return 'main'
    result = subprocess.run(['git','rev-parse','HEAD'], cwd=root,
                            capture_output=True, text=True, check=False)
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and re.fullmatch(r'[0-9a-f]{40,64}',revision) else 'main'


def build(root=ROOT, source_ref=None):
    root = Path(root).resolve()
    source_ref = source_revision(root) if source_ref is None else source_ref
    docs, website = root / 'docs', root / 'website'
    # Base sdists omit the executed public CT notebook's NC-derived image.
    # Recreate its code-only source, without executing/downloading anything.
    if not (website / 'tutorials/07_public_ct.ipynb').exists():
        import subprocess
        import sys
        subprocess.run([sys.executable, str(root / 'tools/build_public_ct_notebook.py')], check=True)
    output = website / 'reference'
    sources = [docs / name for name in PAGES]
    sources += sorted((docs / 'user-guide').glob('*.md'))
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    destinations = {p.resolve(): output / p.relative_to(docs) for p in sources}
    manifest_path = output / '_sources.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    owned = set(previous.get('files', {}))
    for target in destinations.values():
        if target.exists() and target.relative_to(output).as_posix() not in owned:
            raise FileExistsError('Refusing to overwrite an unmanaged website source: ' + str(target))
    records = {}
    for source, target in destinations.items():
        text = source.read_text(encoding='utf-8')

        def rewrite(match):
            href = match['target']
            if href.startswith(('#', '/', 'http:', 'https:', 'mailto:', 'data:')) or '://' in href:
                return match.group(0)
            path, marker, fragment = href.partition('#')
            resolved = (source.parent / path).resolve()
            if resolved in destinations:
                href = Path(os.path.relpath(destinations[resolved], target.parent)).as_posix()
            elif resolved.is_relative_to(website):
                href = Path(os.path.relpath(resolved, target.parent)).as_posix()
            elif resolved.is_relative_to(root):
                href = ('https://github.com/hurry060215-tech/radiomics/blob/'
                        + quote(source_ref,safe='') + '/'
                        + quote(resolved.relative_to(root).as_posix()))
            else:
                raise ValueError('A relative documentation link escapes the repository: ' + str(source) + ': ' + href)
            return match['head'] + href + (marker + fragment if marker else '') + match['tail']

        text = LINK.sub(rewrite, text)
        # Rustdoc attributes are not Pygments lexer names; Mermaid is a directive.
        text = re.sub(r'^```rust,ignore\s*$', '```rust', text, flags=re.MULTILINE)
        text = re.sub(r'^```csv\s*$', '```text', text, flags=re.MULTILINE)
        text = re.sub(r'^```mermaid\s*$', '```{mermaid}', text, flags=re.MULTILINE)
        # Pages omitted from the shorter navigation remain available via links.
        text = '---\norphan: true\n---\n\n' + text
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode('utf-8'))
        records[target.relative_to(output).as_posix()] = {
            'source': source.relative_to(root).as_posix(),
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    stale = owned - records.keys()
    for name in stale:
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError('Invalid managed source path: ' + name)
        if path.is_file():
            path.unlink()
    manifest_path.write_bytes((json.dumps({'schema': 1, 'source_ref':source_ref,'files': records}, indent=2) + '\n').encode())
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-ref', help='Optional explicit Git source ref for repository links.')
    args = parser.parse_args()
    print('Prepared', len(build(args.root,args.source_ref)), 'reference pages from maintained Markdown sources.')
