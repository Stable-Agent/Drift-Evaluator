#!/usr/bin/env python
"""Source OOD bugs for prereg_corrector_file_loc_v1 (leak-free v2).

Differences from ood_source.py (v1), per the pre-registration:
  - problem statement = linked ISSUE title+body (pre-fix), never the PR body;
    the issue must have been opened before the PR and must not contain a
    diff/patch block
  - adds FILE arm input: full content of the (single) primary source file at
    the PR's BASE commit (pre-fix), <= 60 KB
  - keeps the v1 gold-hunk window as the labeled WINDOW (oracle) arm
  - records changed test files + base sha so the execution harness can run
    the PR's own tests at the pre-fix commit
  - buildable repos only; every exclusion is logged with a reason

Output: reports/ood_cases_v2.jsonl (resumable; one line per kept case)
        reports/ood_cases_v2_excluded.jsonl (audit trail)
"""
from __future__ import annotations
import csv, io, json, re, sys, pathlib, urllib.parse, urllib.request

sys.path.insert(0, "Drift-Evaluator/scripts_tmp")
from ood_source import sh, parse_diff, gold_and_code, is_test, is_src

OUT = pathlib.Path("Drift-Evaluator/reports/ood_cases_v2.jsonl")
EXCL = pathlib.Path("Drift-Evaluator/reports/ood_cases_v2_excluded.jsonl")
CUTOFF = "2026-02-15"            # margin past the Jan-2026 training cutoff
REPOS = ["sympy/sympy", "django/django", "pydata/xarray",
         "pylint-dev/pylint", "pytest-dev/pytest", "scrapy/scrapy"]
PER_REPO = 12                    # diversity cap; target >= 45 total
MAX_FILE_BYTES = 60_000
MAX_GOLD_CHARS = 6_000

_RE_PATCHY = re.compile(r"```diff|diff --git|^\+\+\+ |^--- ", re.M)


def gh_json(args):
    raw = sh(["gh"] + args)
    try:
        return json.loads(raw) if raw else None
    except Exception:
        return None


def file_at_ref(repo: str, path: str, ref: str) -> str | None:
    """Raw file content at a commit, via the contents API."""
    q = urllib.parse.quote(path)
    return sh(["gh", "api", "-H", "Accept: application/vnd.github.raw",
               f"repos/{repo}/contents/{q}?ref={ref}"]) or None


def trac_issue(pr_title: str) -> dict | None:
    """django tracks bugs in Trac, not GitHub issues. PR titles follow
    'Fixed #NNNNN -- ...'; the ticket definitionally predates the fix PR,
    and the Trac CSV 'description' field is the reporter's original (pre-fix)
    bug report only — comments (where fix discussion lives) are not included.
    Logged as prereg amendment 2026-06-09."""
    m = re.match(r"Fixed #(\d+)", pr_title)
    if not m:
        return None
    num = m.group(1)
    url = f"https://code.djangoproject.com/ticket/{num}?format=csv"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            row = next(csv.DictReader(io.StringIO(r.read().decode("utf-8-sig"))))
    except Exception:
        return None
    if not row.get("summary") or not row.get("description"):
        return None
    return {"title": row["summary"], "body": row["description"],
            "createdAt": "", "url": f"https://code.djangoproject.com/ticket/{num}"}


def main():
    seen = set()
    if OUT.exists():
        seen = {json.loads(l)["iid"] for l in open(OUT) if l.strip()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fp, fx = OUT.open("a"), EXCL.open("a")

    def skip(iid, reason):
        fx.write(json.dumps({"iid": iid, "reason": reason}) + "\n")
        fx.flush()

    kept_total = 0
    for repo in REPOS:
        # django's convention is 'Fixed #NNNNN --'; 'fix' doesn't match it
        query = "Fixed in:title" if repo == "django/django" else "fix in:title"
        raw = sh(["gh", "search", "prs", query, "--repo", repo,
                  "--merged-at", f">{CUTOFF}", "--limit", "100",
                  "--json", "number,title,url"])
        if not raw:
            print(f"{repo}: no search results")
            continue
        n_repo = 0
        for pr in json.loads(raw):
            iid = f"{repo.split('/')[-1]}#{pr['number']}"
            if iid in seen:
                n_repo += 1
                continue
            title = pr["title"]
            if re.match(r"(?i)^(backport|merge|\[?backport)", title):
                continue
            if not re.search(r"(?i)\b(fix(ed|es)?|bug|regression|incorrect|error|broken)\b", title):
                continue
            url = pr["url"]
            meta = gh_json(["pr", "view", url, "--json",
                            "createdAt,baseRefOid,closingIssuesReferences"])
            if not meta:
                skip(iid, "pr view failed"); continue

            # --- pre-fix problem text: linked issue opened before the PR ---
            issue = None
            if repo == "django/django":
                issue = trac_issue(title)
            else:
                for ref in (meta.get("closingIssuesReferences") or []):
                    num = ref.get("number")
                    if not num:
                        continue
                    iv = gh_json(["issue", "view", str(num), "--repo", repo,
                                  "--json", "title,body,createdAt,url"])
                    if iv and iv.get("createdAt", "9999") < meta["createdAt"]:
                        issue = iv
                        break
            if issue is None:
                skip(iid, "no linked pre-PR issue"); continue
            body = issue.get("body") or ""
            if _RE_PATCHY.search(body):
                skip(iid, "issue body contains diff/patch"); continue
            problem = (issue["title"] + "\n\n" + body).strip()
            if len(problem) < 80:
                skip(iid, "issue text too short"); continue

            # --- gold diff: exactly one primary source file + >=1 test file ---
            diff = sh(["gh", "pr", "diff", url])
            if not diff:
                skip(iid, "no diff"); continue
            files = parse_diff(diff)
            test_files = [f for f in files if is_test(f)]
            src_files = [f for f in files if is_src(f)]
            if not test_files:
                skip(iid, "no test change"); continue
            if len(src_files) != 1:
                skip(iid, f"{len(src_files)} source files (need exactly 1)"); continue
            f, gold, window_code = gold_and_code(files)
            if not f or not gold:
                skip(iid, "gold reconstruction failed"); continue
            if len(gold) > MAX_GOLD_CHARS:
                skip(iid, "gold too large"); continue

            # --- FILE arm input: full file at the base (pre-fix) commit ---
            base = meta["baseRefOid"]
            content = file_at_ref(repo, f, base)
            if not content:
                skip(iid, "base-commit file fetch failed"); continue
            if len(content) > MAX_FILE_BYTES:
                skip(iid, f"file > {MAX_FILE_BYTES} bytes"); continue

            fp.write(json.dumps({
                "iid": iid, "repo": repo, "url": url, "title": title,
                "issue_url": issue["url"], "problem": problem[:2500],
                "file": f, "file_content": content,
                "window_code": (window_code or "")[:2500],
                "gold": gold, "test_files": test_files,
                "base_sha": base, "has_test": True}) + "\n")
            fp.flush()
            kept_total += 1
            n_repo += 1
            if n_repo >= PER_REPO:
                break
        print(f"{repo}: kept {n_repo}")
    fp.close(); fx.close()
    print(f"\ntotal v2 OOD cases: {kept_total} new ({len(seen)} pre-existing)")
    print(f"kept -> {OUT}\nexclusions -> {EXCL}")
    print("Next: freeze the case list, then run the three arms per "
          "docs/prereg_corrector_file_loc_v1.md")


if __name__ == "__main__":
    main()
