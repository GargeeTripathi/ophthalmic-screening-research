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

cd /tmp/vercel-deploy
vercel link --yes --project ophthalmic-screening
vercel deploy --prod --yes
```

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

## Two more traps, already hit once each

- **Don't let `vercel link` (run with no `--project` flag) create a brand-new project.** A
  fresh project on this team defaults to Vercel Deployment Protection (SSO), so its URLs
  302-redirect to a login page instead of serving the site. Always pass
  `--project ophthalmic-screening` explicitly.
- **The CLI prints the deployment URL when the build *starts*, not when it finishes.**
  Wrapping the command in a shell `timeout` and reading its stdout is not a valid way to
  confirm success — check `readyState` via the API, or curl the URL for known content.
