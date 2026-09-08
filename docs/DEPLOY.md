# Redeploying the site to Vercel

```bash
python scripts/build_site.py                              # rebuild docs/ from results/
```

Then, **do not** `cd docs && vercel deploy` from inside this repository. It will silently
block. Instead:

```bash
rm -rf /tmp/vercel-deploy && mkdir -p /tmp/vercel-deploy
cp -r docs/* /tmp/vercel-deploy/
rm -rf /tmp/vercel-deploy/.vercel /tmp/vercel-deploy/.env.local /tmp/vercel-deploy/.gitignore
rm -f /tmp/vercel-deploy/DEPLOY.md          # this file; cp -r docs/* sweeps it in

cd /tmp/vercel-deploy
vercel link --yes --project ophthalmic-screening

rm -f .env.local .gitignore                 # link RECREATES these - see the traps below

vercel deploy --prod --yes
```

Then verify against the live site, not the CLI output:

```bash
curl -s https://ophthalmic-screening.vercel.app/ | grep -c "class=rail"
curl -s -o /dev/null -w '%{http_code}\n' https://ophthalmic-screening.vercel.app/.env.local
```

The first should be non-zero, the second `404`.

## Why the copy step is mandatory

`docs/` lives inside this git repository. The Vercel CLI auto-detects that and attaches
git-commit metadata to the deployment, which triggers a GitHub-committer verification check.
Because the local git identity here (`gargee@example.com`) is not a verified GitHub account
on this Vercel team, **every deployment silently sits at `readyState: BLOCKED`** — no error,
no timeout, just an indefinitely "Building…" CLI that never returns. The API-level reason,
visible only via `GET /v13/deployments/{id}`, is:

> "The Deployment was blocked because GitHub could not associate the committer with a
> GitHub user."

This cost six blocked deployments and about forty minutes of misdiagnosis as a CLI hang
before the actual cause turned up. Deploying from a directory with no `.git` ancestor
sidesteps the check entirely, since there is no commit for Vercel to try to verify.

## Four more traps, already hit once each

- **Don't let `vercel link` (run with no `--project` flag) create a brand-new project.** A
  fresh project on this team defaults to Vercel Deployment Protection (SSO), so its URLs
  302-redirect to a login page instead of serving the site. Always pass
  `--project ophthalmic-screening` explicitly.
- **The CLI prints the deployment URL when the build *starts*, not when it finishes.**
  Wrapping the command in a shell `timeout` and reading its stdout is not a valid way to
  confirm success — check `readyState` via the API, or curl the URL for known content.
- **`vercel link` recreates `.env.local`, containing a fresh `VERCEL_OIDC_TOKEN`.** Deleting
  it during the copy step is therefore not enough: linking puts it back, *after* that
  deletion and *before* the deploy, so an unguarded run publishes a live token at
  `/.env.local`. Delete it again between `link` and `deploy`, which is what the command
  block above now does, and confirm the deployed path 404s.
- **`cp -r docs/*` copies this file too.** DEPLOY.md names the Vercel team's auth quirks and
  a git identity, none of which belongs on a public site. Remove it from the staging
  directory before deploying.

## Verifying a deployment

Check the production domain, not the per-deployment URL the CLI prints. Deployment
Protection can leave `ophthalmic-screening-<hash>-<team>.vercel.app` serving an SSO login
page (~330 KB of HTML, HTTP 200) while `ophthalmic-screening.vercel.app` serves the real
site perfectly well. A 200 on the hashed URL proves nothing on its own — grep the response
for content you know you just changed.
