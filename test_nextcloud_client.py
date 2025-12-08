import os
import sys
import types
from unittest import TestCase, mock

if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace(Session=lambda *args, **kwargs: None)

from nextcloud_client import NextcloudClient


class DummyResponse:
    def __init__(self, *, status_code=200, text="", ok=True, content=b"", iter_chunks=None):
        self.status_code = status_code
        self.text = text
        self.ok = ok
        self.content = content
        self._iter_chunks = iter_chunks or [content]

    def iter_content(self, chunk_size=1):
        yield from self._iter_chunks


class DummySession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.request_calls = []
        self.get_calls = []
        self.auth = None

    def request(self, method, url, headers=None, data=None):
        self.request_calls.append((method, url, headers, data))
        return self.responses.pop(0)

    def get(self, url, stream=False):
        self.get_calls.append((url, stream))
        if self.responses:
            return self.responses.pop(0)
        return DummyResponse(ok=False, status_code=500, text="no response")

    def put(self, url, data=None):
        self.get_calls.append((url, data))
        if self.responses:
            return self.responses.pop(0)
        return DummyResponse(ok=False, status_code=500, text="no response")


class NextcloudClientTests(TestCase):
    def setUp(self):
        # Avoid noise during tests
        self._orig_debug = os.environ.get("NEXTCLOUD_DEBUG")
        os.environ["NEXTCLOUD_DEBUG"] = "0"

    def tearDown(self):
        if self._orig_debug is None:
            os.environ.pop("NEXTCLOUD_DEBUG", None)
        else:
            os.environ["NEXTCLOUD_DEBUG"] = self._orig_debug

    @mock.patch("nextcloud_client.requests.Session")
    def test_login_and_list_directory(self, session_cls):
        xml = """
        <d:multistatus xmlns:d='DAV:'>
          <d:response>
            <d:href>http://example/remote.php/dav/files/user/RAGdocuments/</d:href>
            <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype><d:getetag>\"root\"</d:getetag></d:prop></d:propstat>
          </d:response>
          <d:response>
            <d:href>http://example/remote.php/dav/files/user/RAGdocuments/file.pdf</d:href>
            <d:propstat><d:prop><d:getcontentlength>5</d:getcontentlength><d:getlastmodified>Wed, 21 Oct 2015 07:28:00 GMT</d:getlastmodified><d:resourcetype/><d:getetag>\"etag2\"</d:getetag></d:prop></d:propstat>
          </d:response>
        </d:multistatus>
        """
        dummy = DummySession([DummyResponse(status_code=207), DummyResponse(status_code=207, text=xml)])
        session_cls.return_value = dummy

        client = NextcloudClient("http://example", "user", "token")
        self.assertTrue(client.login())

        entries = client.list_directory("/RAGdocuments", depth=1)
        self.assertEqual(len(entries), 1)
        file_entry = entries[0]
        self.assertEqual(file_entry["path"], "/RAGdocuments/file.pdf")
        self.assertEqual(file_entry["size"], 5)
        self.assertFalse(file_entry["is_dir"])

        self.assertEqual(dummy.request_calls[0][0], "PROPFIND")

    @mock.patch("nextcloud_client.requests.Session")
    def test_walk_yields_files(self, session_cls):
        xml = """
        <d:multistatus xmlns:d='DAV:'>
          <d:response>
            <d:href>http://example/remote.php/dav/files/user/RAGdocuments/</d:href>
            <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat>
          </d:response>
          <d:response>
            <d:href>http://example/remote.php/dav/files/user/RAGdocuments/example.txt</d:href>
            <d:propstat><d:prop><d:getcontentlength>3</d:getcontentlength><d:resourcetype/><d:getetag>\"etag\"</d:getetag></d:prop></d:propstat>
          </d:response>
        </d:multistatus>
        """
        dummy = DummySession([DummyResponse(status_code=207, text=xml)])
        session_cls.return_value = dummy

        client = NextcloudClient("http://example", "user", "token")
        files = list(client.walk("/RAGdocuments"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["path"], "/RAGdocuments/example.txt")

    @mock.patch("nextcloud_client.requests.Session")
    def test_download_to_temp(self, session_cls):
        content = b"data123"
        dummy = DummySession([DummyResponse(ok=True, iter_chunks=[content])])
        session_cls.return_value = dummy

        client = NextcloudClient("http://example", "user", "token")
        path = client.download_to_temp("/RAGdocuments/file.bin", suffix=".bin")
        self.addCleanup(path.unlink)

        with path.open("rb") as fh:
            self.assertEqual(fh.read(), content)
        self.assertTrue(any("file.bin" in call[0] for call in dummy.get_calls))
