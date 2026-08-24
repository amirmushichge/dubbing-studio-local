# Security policy

## Supported version

Security fixes currently target the latest `0.1.x` public alpha release only.

## Reporting a vulnerability

Report vulnerabilities privately through **GitHub → Security → Advisories → New draft security advisory**. Do not open a public issue for a suspected vulnerability.

Never attach source videos, voice samples, project folders, setup logs, tokens, system usernames or other personal data to a public issue. A minimal reproduction made from synthetic media is preferred.

## Local-network boundary

The application listens on `127.0.0.1` only. Do not change it to `0.0.0.0` unless you understand and secure the network exposure. Project data and model caches are not encrypted at rest, so operating-system accounts with access to the data directory can read them.

Model repositories and weights are pinned by revision and downloaded from their documented upstream sources. Do not replace them with untrusted model files. Clone a voice only with its owner’s permission, and never use the result for deception, impersonation, fraud or identity-verification bypass.

Seed-VC uses `descript-audio-codec`, whose optional training stack declares an obsolete protobuf ceiling. The production installer applies the audited protobuf version from `requirements/seedvc-overrides.txt` after installing the upstream-compatible packages. The inference imports used by Dubbing Studio are covered by CI and the local release checks.
