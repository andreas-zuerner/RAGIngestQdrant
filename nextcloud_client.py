import base64
import os
import posixpath
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional

import requests


class NextcloudError(Exception):
    pass


class NextcloudClient:
    """Minimal WebDAV client for Nextcloud file access."""

    def __init__(self, base_url: str, user: str, token: str):
        if not token:
            raise NextcloudError("Missing Nextcloud token")
        self.base_url = base_url.rstrip("/")
        self.user = user
        self.session = requests.Session()
        self.session.auth = (user, token)
        self._dav_base = f"{self.base_url}/remote.php/dav/files/{self.user}"

    # ---------- Helpers ----------
    def _url(self, remote_path: str) -> str:
        clean = posixpath.normpath("/" + (remote_path or "")).lstrip("/")
        return f"{self._dav_base}/{clean}"

    def _ensure_parent(self, remote_path: str):
        parts = posixpath.normpath("/" + remote_path).split("/")
        if len(parts) <= 1:
            return
        cumulative = []
        for part in parts[:-1]:
            if not part:
                continue
            cumulative.append(part)
            current = "/" + "/".join(cumulative)
            self.mkdir(current)

    def _parse_prop(self, response_text: str, base_path: str) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        ns = {"d": "DAV:"}
        try:
            root = ET.fromstring(response_text)
        except ET.ParseError as exc:
            raise NextcloudError(f"Failed to parse WebDAV response: {exc}") from exc

        for resp in root.findall("d:response", ns):
            href_elem = resp.find("d:href", ns)
            if href_elem is None or not href_elem.text:
                continue
            href = href_elem.text
            # Normalize: strip base + trailing slash
            if href.startswith(self._dav_base):
                rel = href[len(self._dav_base) :]
            else:
                rel = href
            rel = rel.rstrip("/") or "/"
            if rel == base_path.rstrip("/"):
                is_root = True
            else:
                is_root = False

            propstat = resp.find("d:propstat", ns)
            prop = propstat.find("d:prop", ns) if propstat is not None else None
            if prop is None:
                continue
            res_type = prop.find("d:resourcetype", ns)
            is_dir = res_type is not None and res_type.find("d:collection", ns) is not None
            length_elem = prop.find("d:getcontentlength", ns)
            size = int(length_elem.text) if length_elem is not None and length_elem.text else 0
            modified_elem = prop.find("d:getlastmodified", ns)
            mtime = self._to_timestamp(modified_elem.text if modified_elem is not None else None)
            etag_elem = prop.find("d:getetag", ns)
            etag = (etag_elem.text or "").strip('"') if etag_elem is not None else None

            entries.append(
                {
                    "path": rel,
                    "is_dir": is_dir,
                    "size": size,
                    "mtime": mtime,
                    "etag": etag,
                    "is_root": is_root,
                }
            )
        return entries

    def _to_timestamp(self, value: Optional[str]) -> int:
        if not value:
            return 0
        try:
            dt = parsedate_to_datetime(value)
            return int(dt.timestamp())
        except Exception:
            return 0

    # ---------- Operations ----------
    def login(self) -> bool:
        url = self._url("/")
        r = self.session.request("PROPFIND", url, headers={"Depth": "0"})
        if not r.ok:
            raise NextcloudError(f"Login failed: {r.status_code} {r.text[:200]}")
        return True

    def mkdir(self, remote_path: str):
        url = self._url(remote_path)
        r = self.session.request("MKCOL", url)
        if r.status_code in (201, 405):
            return
        if r.status_code == 409:
            # Parent missing – try to create it
            parent = posixpath.dirname(remote_path.rstrip("/"))
            if parent and parent != "/":
                self.mkdir(parent)
                self.mkdir(remote_path)
                return
        if not r.ok:
            raise NextcloudError(f"MKCOL failed for {remote_path}: {r.status_code} {r.text[:200]}")

    def list_directory(self, remote_path: str, depth: int = 1) -> List[Dict[str, object]]:
        url = self._url(remote_path)
        body = (
            "<?xml version='1.0' encoding='UTF-8'?>"
            "<d:propfind xmlns:d='DAV:'>"
            "<d:prop><d:getlastmodified/><d:getcontentlength/><d:resourcetype/><d:getetag/></d:prop>"
            "</d:propfind>"
        )
        r = self.session.request("PROPFIND", url, headers={"Depth": str(depth)}, data=body)
        if not r.ok:
            raise NextcloudError(f"PROPFIND failed for {remote_path}: {r.status_code} {r.text[:200]}")
        entries = self._parse_prop(r.text, posixpath.normpath(remote_path) or "/")
        return [e for e in entries if not e.get("is_root")]

    def walk(self, remote_path: str) -> Generator[Dict[str, object], None, None]:
        stack = [remote_path]
        while stack:
            current = stack.pop()
            try:
                children = self.list_directory(current, depth=1)
            except NextcloudError:
                continue
            for entry in children:
                if entry.get("is_dir"):
                    stack.append(entry["path"])
                else:
                    yield entry

    def read_file(self, remote_path: str) -> bytes:
        url = self._url(remote_path)
        r = self.session.get(url, stream=True)
        if not r.ok:
            raise NextcloudError(f"GET failed for {remote_path}: {r.status_code} {r.text[:200]}")
        return r.content

    def download_file(self, remote_path: str, target: Path) -> Path:
        url = self._url(remote_path)
        r = self.session.get(url, stream=True)
        if not r.ok:
            raise NextcloudError(f"Download failed for {remote_path}: {r.status_code} {r.text[:200]}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
        return target

    def upload_file(self, remote_path: str, local_path: Path):
        self._ensure_parent(remote_path)
        url = self._url(remote_path)
        with local_path.open("rb") as fh:
            r = self.session.put(url, data=fh)
        if not r.ok:
            raise NextcloudError(f"Upload failed for {remote_path}: {r.status_code} {r.text[:200]}")

    def upload_bytes(self, remote_path: str, data: bytes):
        self._ensure_parent(remote_path)
        url = self._url(remote_path)
        r = self.session.put(url, data=data)
        if not r.ok:
            raise NextcloudError(f"Upload failed for {remote_path}: {r.status_code} {r.text[:200]}")

    def write_text(self, remote_path: str, text: str, encoding: str = "utf-8"):
        self.upload_bytes(remote_path, text.encode(encoding))

    def download_to_temp(self, remote_path: str, suffix: str = "") -> Path:
        tmp = Path(tempfile.mkstemp(prefix="nxc-", suffix=suffix)[1])
        return self.download_file(remote_path, tmp)


def env_client() -> NextcloudClient:
    base_url = os.environ.get("NEXTCLOUD_BASE_URL", "http://192.168.177.133:8080").rstrip("/")
    user = os.environ.get("NEXTCLOUD_USER", "andreas")
    token = os.environ.get("TOKEN") or os.environ.get("NEXTCLOUD_TOKEN", "")
    client = NextcloudClient(base_url, user, token)
    client.login()
    return client
