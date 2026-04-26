# Fixture Policy

Fixtures are the compatibility contract for persisted data and archive formats.
Because Cerememory is pre-release, incompatible fixture changes are allowed, but
they must be intentional and reviewed in the same commit as the code change.

Add fixtures when changing:

- CMA archive header, footer, checksum, encryption, or bundle semantics.
- Store encryption payload format or migration behavior.
- redb table names, key encodings, or value encodings.
- Public protocol JSON shapes that are consumed outside this workspace.

Each fixture set should include:

- A short README describing how the fixture was produced.
- At least one load/import/migration test that reads the fixture from disk.
- Expected counts or checksums so silent partial reads fail loudly.

Do not rewrite existing fixtures to make tests pass unless the change is a
deliberate breaking format reset. In that case, document the reset in
`CHANGELOG.md`.
