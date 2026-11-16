"""Check the deployed revision and require retired generated pages to return 404.

Retired routes come from the checked-out branch's historical maintained page
lists. Reintroduced pages are excluded. Redirects and server errors never count
as successful retirement. No optional numerical package is imported.
"""
import argparse
import ast
from pathlib import Path, PurePosixPath
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
BUILDER = 'tools/build_website_sources.py'


def page_list(source):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'PAGES'
            for target in node.targets
        ):
            pages = ast.literal_eval(node.value)
            if not isinstance(pages, (list, tuple)) or not all(
                isinstance(page, str) and page.endswith('.md') for page in pages
            ):
                raise ValueError('Expected a literal list of maintained Markdown pages.')
            for page in pages:
                path = PurePosixPath(page)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Historical page paths must remain repository-relative.')
            return set(pages)
    return set()


def git(root, *arguments):
    return subprocess.check_output(
        ['git', *arguments], cwd=root, text=True, encoding='utf-8'
    ).strip()


def retired_routes(root=ROOT):
    if git(root, 'rev-parse', '--is-shallow-repository') != 'false':
        raise ValueError('Retired-page checks require full Git history (fetch-depth: 0).')
    current = page_list((root / BUILDER).read_text(encoding='utf-8'))
    historical = set()
    for revision in git(root, 'log', '--format=%H', '--', BUILDER).splitlines():
        historical.update(page_list(git(root, 'show', revision + ':' + BUILDER)))
    return [
        'reference/' + str(PurePosixPath(page).with_suffix('.html'))
        for page in sorted(historical - current)
    ]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def response(url):
    opener = build_opener(NoRedirect())
    try:
        with opener.open(
            Request(url, headers={'Cache-Control': 'no-cache'}), timeout=30
        ) as result:
            return result.status, result.read()
    except HTTPError as error:
        return error.code, b''


def require_response(url, expected_status, marker=None, attempts=6, retry_seconds=5):
    if attempts < 1 or retry_seconds < 0:
        raise ValueError('Retry attempts must be positive and delays nonnegative.')
    last = 'not requested'
    for attempt in range(attempts):
        try:
            status, body = response(url)
            last = 'HTTP ' + str(status)
            if status == expected_status and (marker is None or marker in body):
                print(url + ': ' + last, flush=True)
                return
            if status == expected_status and marker is not None:
                last += ', expected build revision absent'
        except (URLError, TimeoutError, OSError) as error:
            last = str(error)
        if attempt + 1 < attempts:
            time.sleep(retry_seconds)
    raise RuntimeError(url + ': expected HTTP ' + str(expected_status) + '; ' + last)


def check_publication(base_url, revision, *, root=ROOT, attempts=6, retry_seconds=5):
    parsed = urlsplit(base_url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError('An absolute HTTP(S) publication URL is required.')
    base_url = base_url.rstrip('/') + '/'
    marker = ('data-build-revision="' + revision + '"').encode('utf-8')
    require_response(
        base_url, 200, marker, attempts=attempts, retry_seconds=retry_seconds
    )
    routes = retired_routes(root)
    for route in routes:
        require_response(
            urljoin(base_url, route), 404,
            attempts=attempts, retry_seconds=retry_seconds,
        )
    print('Verified current revision and ' + str(len(routes)) + ' retired page(s).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--attempts', type=int, default=6)
    parser.add_argument('--retry-seconds', type=float, default=5)
    arguments = parser.parse_args()
    check_publication(
        arguments.base_url, arguments.revision,
        attempts=arguments.attempts, retry_seconds=arguments.retry_seconds,
    )
