"""Publication checks must distinguish retired routes from stale deployments."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import subprocess
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'documentation_publication', ROOT / 'tools/check_documentation_publication.py'
)
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


@pytest.fixture
def server():
    active = []

    def start(routes):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                status, body = routes.get(self.path, (404, b''))
                self.send_response(status)
                if 300 <= status < 400:
                    self.send_header('Location', '/genuinely-missing.html')
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        active.append((http, thread))
        return 'http://127.0.0.1:' + str(http.server_port) + '/'

    yield start
    for http, thread in active:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


def test_historical_routes_exclude_current_and_reintroduced_pages(tmp_path):
    subprocess.run(['git', 'init', '--quiet'], cwd=tmp_path, check=True)
    builder = tmp_path / publication.BUILDER
    builder.parent.mkdir()

    def commit(pages):
        builder.write_text('PAGES = ' + repr(pages) + '\n', encoding='utf-8')
        subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
        subprocess.run(
            ['git', '-c', 'user.name=Publication test', '-c',
             'user.email=publication-test@example.invalid', 'commit', '--quiet',
             '-m', 'Maintain page list'], cwd=tmp_path, check=True,
        )

    commit(('retired.md', 'kept.md', 'restored.md'))
    commit(('kept.md',))
    commit(('kept.md', 'restored.md'))
    assert publication.retired_routes(tmp_path) == ['reference/retired.html']


def test_publication_verifies_revision_and_a_retired_route(server, monkeypatch):
    revision = 'a' * 40
    base = server({'/': (200, ('data-build-revision="' + revision + '"').encode())})
    monkeypatch.setattr(publication, 'retired_routes', lambda root: ['reference/retired.html'])
    publication.check_publication(base, revision, attempts=1, retry_seconds=0)


@pytest.mark.parametrize('status', [200, 301, 302, 500])
def test_only_404_passes_retirement_even_if_a_redirect_target_is_missing(server, status):
    base = server({'/reference/retired.html': (status, b'')})
    with pytest.raises(RuntimeError, match='HTTP ' + str(status)):
        publication.require_response(
            base + 'reference/retired.html', 404, attempts=1, retry_seconds=0
        )


def test_stale_revision_retries_are_bounded(monkeypatch):
    calls = []

    def stale_response(url):
        calls.append(url)
        return 200, b'data-build-revision="stale"'

    monkeypatch.setattr(publication, 'response', stale_response)
    with pytest.raises(RuntimeError, match='expected build revision absent'):
        publication.require_response(
            'https://example.invalid/', 200, b'current', attempts=2, retry_seconds=0
        )
    assert len(calls) == 2
