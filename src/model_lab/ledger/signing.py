"""SSH-signature wrapper around `ssh-keygen -Y sign/verify/find-principals`.

MODEL-1 never generates operator or validator keys (D2) — this module is
used in production only to *verify* detached `.sig` files the operator/
validator drop in `operator-inbox/`, and in tests only with throwaway keys
created and deleted inside the test run (`generate_throwaway_keypair`).

Signature format: OpenSSH SSHSIG (the `ssh-keygen -Y sign` output format),
namespaced per purpose (e.g. `model-lab-transition`, `model-lab-checkpoint`),
verified against a role-restricted `allowed_signers` file
(`principal namespaces="ns1,ns2" <public key line>`), exactly the pattern
used for `git`'s own SSH-signed-commit verification.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


class SigningToolUnavailableError(RuntimeError):
    pass


class SignatureVerificationError(ValueError):
    pass


def _require_ssh_keygen() -> str:
    path = shutil.which("ssh-keygen")
    if path is None:
        raise SigningToolUnavailableError("ssh-keygen is not on PATH")
    return path


@dataclass(frozen=True)
class ThrowawayKeypair:
    private_key_path: Path
    public_key_path: Path
    public_key_text: str


def generate_throwaway_keypair(dest_dir: Path, *, comment: str = "model-lab-throwaway") -> ThrowawayKeypair:
    """Create an ed25519 keypair for TEST USE ONLY. Callers are responsible
    for deleting `dest_dir` (or the two key files) once the test completes —
    this function does not register any cleanup itself, to keep it usable
    both as a bare function and inside a pytest fixture with its own
    teardown."""
    ssh_keygen = _require_ssh_keygen()
    dest_dir.mkdir(parents=True, exist_ok=True)
    priv = dest_dir / "id_ed25519"
    subprocess.run(
        [ssh_keygen, "-t", "ed25519", "-N", "", "-C", comment, "-f", str(priv), "-q"],
        check=True,
        capture_output=True,
    )
    pub = priv.with_suffix(".pub")
    return ThrowawayKeypair(
        private_key_path=priv,
        public_key_path=pub,
        public_key_text=pub.read_text(encoding="utf-8").strip(),
    )


def sign_bytes(data: bytes, private_key_path: Path, namespace: str) -> str:
    """Return the SSHSIG signature text (contents of the `<file>.sig`
    ssh-keygen produces) for `data`, signed under `namespace`."""
    ssh_keygen = _require_ssh_keygen()
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "payload"
        target.write_bytes(data)
        subprocess.run(
            [ssh_keygen, "-Y", "sign", "-f", str(private_key_path), "-n", namespace, str(target)],
            check=True,
            capture_output=True,
        )
        sig_path = target.with_suffix(target.suffix + ".sig")
        return sig_path.read_text(encoding="utf-8")


def write_allowed_signers(
    path: Path, entries: list[tuple[str, list[str], str]]
) -> None:
    """`entries`: list of (principal, namespaces, public_key_line). Mirrors
    the `git`/OpenSSH `allowed_signers` file format:
        <principal> namespaces="ns1,ns2" <public-key-line>
    """
    lines = []
    for principal, namespaces, pubkey_line in entries:
        ns = ",".join(namespaces)
        lines.append(f'{principal} namespaces="{ns}" {pubkey_line}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_signature(
    data: bytes,
    signature_text: str,
    *,
    allowed_signers_path: Path,
    principal: str,
    namespace: str,
) -> bool:
    """Verify `signature_text` over `data` for `principal` in `namespace`
    against `allowed_signers_path`. Returns True/False rather than raising,
    for a plain fail-closed check callers can branch on directly; callers
    that need the failure detail should inspect the subprocess output
    themselves (kept out of the return value so verification never leaks
    stderr content into a signature/decision record)."""
    ssh_keygen = _require_ssh_keygen()
    with tempfile.TemporaryDirectory() as td:
        payload = Path(td) / "payload"
        payload.write_bytes(data)
        sig_path = Path(td) / "payload.sig"
        sig_path.write_text(signature_text, encoding="utf-8")
        result = subprocess.run(
            [
                ssh_keygen,
                "-Y",
                "verify",
                "-f",
                str(allowed_signers_path),
                "-I",
                principal,
                "-n",
                namespace,
                "-s",
                str(sig_path),
            ],
            input=data,
            capture_output=True,
        )
        return result.returncode == 0


def find_principals(signature_text: str, allowed_signers_path: Path) -> list[str]:
    """`ssh-keygen -Y find-principals` — list which principals in the
    allowed_signers file could have produced this signature (does not by
    itself verify the signature over any particular message)."""
    ssh_keygen = _require_ssh_keygen()
    with tempfile.TemporaryDirectory() as td:
        sig_path = Path(td) / "sig.sig"
        sig_path.write_text(signature_text, encoding="utf-8")
        result = subprocess.run(
            [ssh_keygen, "-Y", "find-principals", "-f", str(allowed_signers_path), "-s", str(sig_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
