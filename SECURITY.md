# Security and research-data handling

Do not post patient images, clinical tables, identifying paths, credentials or
institution secrets in public issues, notebooks or pull requests. Use a small
synthetic reproduction, sanitized headers and explicit configuration.

For a software vulnerability, use GitHub's private vulnerability reporting option
if available on the repository Security tab. If unavailable, open an issue
requesting a private contact without exploit details or data. No response-time
guarantee or formal security certification is claimed.

Native `.rad` files contain typed numerical state and a versioned manifest.
Unknown schemas/features/types are rejected. Legacy optional Python workflows
use pickle/joblib-style persistence: load only trusted files. Typed formats do
not remove the need to validate file sizes, provenance and OS permissions.

This is research software. It does not establish clinical safety, diagnostic
validity, regulatory clearance or patient-data authorization. Apply institutional
access and de-identification controls. Dataset licenses remain separate from
the MIT software license.
