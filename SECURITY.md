# Security Policy

## Supported Versions

Security updates and cryptographic verification audits are applied to the following releases:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

---

## Reporting a Vulnerability

We take the security and integrity of our research software, dataset bundles, and cryptographic provenance very seriously.

If you discover a potential security vulnerability, credential leak, or integrity flaw in `spatial-graph-bench`, please do not open a public issue. Instead, report it through one of the following channels:

- **Private Security Advisory**: Open a draft security advisory on GitHub under the [Security Advisories](https://github.com/ToruOkadaOi/spatial-graph-bench/security/advisories) tab.
- **Maintainer Contact**: Contact **Aman Nalakath** via GitHub profile [@ToruOkadaOi](https://github.com/ToruOkadaOi).

Please include:
1. A description of the issue and potential impact.
2. Step-by-step reproduction instructions or proof-of-concept.
3. Relevant environment details (OS, Python version, dependency versions).

We will acknowledge receipt within 48 hours and provide a fix or remediation timeline.

---

## Cryptographic Artifact Integrity Model

`spatial-graph-bench` distributes large benchmark artifacts (preprocessed PCA feature matrices and spatial graph topologies) via GitHub Releases rather than tracking large binary files in Git.

To protect against data tampering and man-in-the-middle attacks across distributed worker nodes:
1. Every release asset is accompanied by an immutable `sha256sums_*.txt` manifest.
2. The [`manage_release_artifacts.py`](file:///Users/aman/Documents/spatial-graph-bench/scripts/manage_release_artifacts.py) CLI automatically verifies SHA-256 hashes before unpacking any `.tar.gz` archive.
3. The extraction process uses safe archive filters (`filter="data"`) to prevent path traversal attacks.
4. If a checksum mismatch is detected, execution aborts immediately with an error.
