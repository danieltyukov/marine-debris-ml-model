# Security policy

mdebris downloads data and model weights from the network, loads trained models from
disk, and can run as an HTTP service (`mdebris serve`). Problems in those areas matter
more than anywhere else in the project.

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub security advisories: open the repository's
Security tab and choose "Report a vulnerability", or go directly to
https://github.com/danieltyukov/marine-debris-ml-model/security/advisories/new.
Include the version (`mdebris --version`), the operating system, how it was
installed, the steps to reproduce, and what an attacker could gain. If you have a
fix, a draft pull request attached to the advisory is welcome.

## Scope

In scope:

- Model loading. `SpectralClassifier.load` uses joblib, which is pickle-based and
  runs code from the file it loads. The project never downloads or ships a `.joblib`
  file for that reason. A path where the code loads a model file it did not create,
  or a pickle reaching the repository or a release, is in scope.
- The HTTP API behind `mdebris serve`: request validation, file paths or URLs built
  from request fields, and anything that lets a request read files it should not or
  make the server fetch an arbitrary URL.
- Downloads: MARIDA (checked against its md5 before use), Sentinel-2 assets read over
  STAC, OpenStreetMap queries, and model weights fetched from Hugging Face on first
  use. Anything that
  lets a mirror or a crafted STAC item make the code write outside the cache
  directory, skip a checksum, or execute content.
- Credentials: the optional `PL_API_KEY` for Planet, and settings read from `.env`.
  Keys appearing in logs, error messages, outputs or the API's responses are in
  scope.
- The release and Pages workflows in `.github/workflows/`.

Out of scope:

- Vulnerabilities in Planetary Computer, earth-search, Zenodo, Overpass, Hugging
  Face or Planet themselves. Report those to their operators.
- Problems in third-party dependencies with no demonstrated impact on this project.
  Those are still useful to hear about, and a normal issue is fine.
- Issues that need an attacker who already controls the user's account on the
  machine.
- The accuracy of a detection. A wrong answer is a bug or a limitation, not a
  vulnerability; open an issue.

## What to expect

You should get an acknowledgement within seven days. The project is maintained by
one person, so please allow some slack; if you have heard nothing after two weeks,
comment on the advisory. Confirmed problems are fixed in a release and described in
`CHANGELOG.md` and the advisory once the fix is available. Credit is given to the
reporter unless they ask otherwise.

## Supported versions

Only the latest release receives fixes.
