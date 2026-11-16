import os
import re
import time
import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
gh = requests.Session()                       # all GitHub calls share these headers
gh.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
})

# M0 = newest month, M11 = oldest
MONTHS = list(pd.period_range("2025-07", "2026-06", freq="M"))[::-1]
WIN_START = MONTHS[-1].start_time.tz_localize("UTC")
WIN_END = MONTHS[0].end_time.tz_localize("UTC")
PRED_START = pd.Timestamp("2026-07-01", tz="UTC")
PRED_END = pd.Timestamp("2026-09-30", tz="UTC")     
OUT = "dataset_initial.csv"

PACKAGES = ["debug-js/debug","mochajs/mocha", "chaijs/chai", "sinonjs/sinon", "fastify/fastify",
    "koajs/koa", "sveltejs/svelte", "preactjs/preact",
     "rollup/rollup", "evanw/esbuild",
    "postcss/postcss", "winstonjs/winston", "pinojs/pino", "expressjs/multer",
    "expressjs/morgan", "handlebars-lang/handlebars.js", "node-fetch/node-fetch",
    "sindresorhus/got", "expressjs/express",
    "axios/axios",
    "lodash/lodash",
    "webpack/webpack",
    "vitejs/vite",
    "colinhacks/zod",
    "chalk/chalk",
    "date-fns/date-fns",
    "moment/moment",
    "uuidjs/uuid",
    "markedjs/marked",
    "motdotla/dotenv",
    "nodemailer/nodemailer",
    "Automattic/mongoose",
    "sequelize/sequelize",
    "prisma/prisma",
    "typeorm/typeorm",
    "nestjs/nest",
    "remix-run/react-router",
    "tanstack/query"
]         

STABLE = re.compile(r"^\d+\.\d+\.\d+$")


# ---------------------------------------------------------------- http
def pages(url, params=None):
    """Yield every item of a paginated GitHub list endpoint."""
    params = {**(params or {}), "per_page": 100}
    while url:
        r = gh.get(url, params=params, timeout=30)
        r.raise_for_status()
        yield from r.json()
        url, params = r.links.get("next", {}).get("url"), None


# ---------------------------------------------------------------- raw github data (GraphQL)
def load_issue_tables(repo):
    owner, name = repo.split("/", 1)
    query = """
    query($owner: String!, $name: String!, $cursorIssues: String, $cursorPRs: String) {
      repository(owner: $owner, name: $name) {
        issues(first: 100, after: $cursorIssues, orderBy: {field: CREATED_AT, direction: ASC}) {
          pageInfo { hasNextPage endCursor }
          nodes {
            number
            createdAt
            timelineItems(first: 50, itemTypes: [CLOSED_EVENT, REOPENED_EVENT]) {
              nodes {
                __typename
                ... on ClosedEvent { id createdAt }
                ... on ReopenedEvent { id createdAt }
              }
            }
          }
        }
        pullRequests(first: 100, after: $cursorPRs, orderBy: {field: CREATED_AT, direction: ASC}) {
          pageInfo { hasNextPage endCursor }
          nodes {
            number
            createdAt
            mergedAt
            timelineItems(first: 50, itemTypes: [CLOSED_EVENT, REOPENED_EVENT]) {
              nodes {
                __typename
                ... on ClosedEvent { id createdAt }
                ... on ReopenedEvent { id createdAt }
              }
            }
          }
        }
      }
    }
    """

    issue_rows, event_rows = [], []
    c_issues, c_prs = None, None
    has_issues, has_prs = True, True
    page_num = 0

    while has_issues or has_prs:
        page_num += 1
        print(f"\r     -> Lade GitHub GraphQL Seite {page_num} ({len(issue_rows)} Issues/PRs)...", end="", flush=True)

        variables = {
            "owner": owner,
            "name": name,
            "cursorIssues": c_issues if has_issues else None,
            "cursorPRs": c_prs if has_prs else None,
        }
        r = gh.post("https://api.github.com/graphql",
                    json={"query": query, "variables": variables}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            print()
            raise RuntimeError(data["errors"])

        repo_data = data["data"]["repository"]

        if has_issues:
            iss_block = repo_data["issues"]
            for node in iss_block["nodes"]:
                num = node["number"]
                issue_rows.append({"number": num, "is_pr": False, "ts": node["createdAt"]})
                for ev in node["timelineItems"]["nodes"]:
                    ev_name = "closed" if ev["__typename"] == "ClosedEvent" else "reopened"
                    event_rows.append({
                        "id": ev["id"], "number": num, "is_pr": False,
                        "event": ev_name, "ts": ev["createdAt"]
                    })
            has_issues = iss_block["pageInfo"]["hasNextPage"]
            c_issues = iss_block["pageInfo"]["endCursor"]

        if has_prs:
            pr_block = repo_data["pullRequests"]
            for node in pr_block["nodes"]:
                num = node["number"]
                issue_rows.append({"number": num, "is_pr": True, "ts": node["createdAt"]})
                if node.get("mergedAt"):
                    event_rows.append({
                        "id": f"merge_{num}", "number": num, "is_pr": True,
                        "event": "merged", "ts": node["mergedAt"]
                    })
                for ev in node["timelineItems"]["nodes"]:
                    ev_name = "closed" if ev["__typename"] == "ClosedEvent" else "reopened"
                    event_rows.append({
                        "id": ev["id"], "number": num, "is_pr": True,
                        "event": ev_name, "ts": ev["createdAt"]
                    })
            has_prs = pr_block["pageInfo"]["hasNextPage"]
            c_prs = pr_block["pageInfo"]["endCursor"]

    print(f"\r     -> GitHub fertig: {len(issue_rows)} Issues/PRs, {len(event_rows)} Events geladen.        ")

    issues = pd.DataFrame(issue_rows, columns=["number", "is_pr", "ts"]).drop_duplicates(subset=["number"])
    events = pd.DataFrame(event_rows, columns=["id", "number", "is_pr", "event", "ts"]).drop_duplicates(subset=["id"])

    for df in (issues, events):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return issues, events


def github_monthly(repo):
    issues, events = load_issue_tables(repo)
    merged_keys = set(zip(events.loc[events.event == "merged", "number"],
                          events.loc[events.event == "merged", "ts"]))
    events["unmerged_close"] = [
        e == "closed" and (n, t) not in merged_keys
        for n, e, t in zip(events.number, events.event, events.ts)]

    def open_at(end, pr):
        made = issues[(issues.is_pr == pr) & (issues.ts <= end)]
        ev = events[(events.is_pr == pr) & events.event.isin(["closed", "reopened"])
                    & (events.ts <= end)].sort_values("ts")
        last = ev.groupby("number").event.last()
        return int((~made.number.isin(last[last == "closed"].index)).sum())

    out = {}
    for i, m in enumerate(MONTHS):
        lo, hi = m.start_time.tz_localize("UTC"), m.end_time.tz_localize("UTC")

        def n_created(pr):
            return int(((issues.is_pr == pr) & issues.ts.between(lo, hi)).sum())

        def n_event(name, pr, mask=None):
            sel = (events.is_pr == pr) & (events.event == name) & events.ts.between(lo, hi)
            if mask is not None:
                sel &= events[mask]
            return int(sel.sum())

        out.update({
            f"issues_opened_count_M{i}": n_created(False),
            f"issues_closed_count_M{i}": n_event("closed", False),
            f"issues_reopened_count_M{i}": n_event("reopened", False),
            f"issues_open_at_month_end_M{i}": open_at(hi, False),
            f"pull_requests_opened_M{i}": n_created(True),
            f"pull_requests_merged_M{i}": n_event("merged", True),
            f"pull_requests_closed_unmerged_M{i}": n_event("closed", True, "unmerged_close"),
            f"pull_requests_reopened_M{i}": n_event("reopened", True),
            f"pull_requests_open_at_month_end_M{i}": open_at(hi, True),
        })
    return out


# ---------------------------------------------------------------- advisories
def duplicates(da):
    """Filtert Advisories heraus, deren Summary mit 'duplicate' beginnt."""
    return [
        item for item in da
        if not (item.get("summary") or "").strip().lower().startswith("duplicate")
    ]


def advisory_table(pkg):
    raw_advisories = list(pages("https://api.github.com/advisories",
                                {"ecosystem": "npm", "affects": pkg}))
    clean_advisories = duplicates(raw_advisories)
    
    rows = [{"id": a.get("ghsa_id"), "ts": a["published_at"], "sev": a.get("severity") or "unknown"}
            for a in clean_advisories]
            
    df = pd.DataFrame(rows, columns=["id", "ts", "sev"]).drop_duplicates(subset=["id"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["sev"] = df.sev.replace({"medium": "moderate"})
    return df


def sev_counts(df, prefix, suffix=""):
    counts = df.sev.value_counts()
    return {f"{prefix}_{s}_count{suffix}": int(counts.get(s, 0))
            for s in ("low", "moderate", "high", "critical", "unknown")}


def ghsa_features(pkg):
    adv = advisory_table(pkg)
    out = sev_counts(adv[adv.ts < PRED_START], "ghsa")
    for i, m in enumerate(MONTHS):
        sel = adv[adv.ts.between(m.start_time.tz_localize("UTC"), m.end_time.tz_localize("UTC"))]
        out.update(sev_counts(sel, "ghsa", f"_M{i}"))
    out.update(sev_counts(adv[(adv.ts >= PRED_START) & (adv.ts < PRED_END)],
                          "target_ghsa", "_90d"))
    return out


# ---------------------------------------------------------------- npm
def npm_monthly(pkg, meta):
    stable_dates = pd.Series(
        [pd.Timestamp(t) for v, t in meta["time"].items() if STABLE.match(v)],
        dtype="datetime64[ns, UTC]")
    url = (f"https://api.npmjs.org/downloads/range/"
           f"{WIN_START:%Y-%m-%d}:{WIN_END:%Y-%m-%d}/{pkg}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    daily = pd.DataFrame(resp.json().get("downloads", []), columns=["day", "downloads"])
    daily["month"] = pd.to_datetime(daily.day).dt.to_period("M")
    out = {}
    for i, m in enumerate(MONTHS):
        d = daily[daily.month == m]
        out[f"package_downloads_M{i}"] = int(d.downloads.sum())
        out[f"package_downloads_days_observed_M{i}"] = len(d)
        out[f"package_stable_release_count_M{i}"] = int(
            stable_dates.between(m.start_time.tz_localize("UTC"),
                                 m.end_time.tz_localize("UTC")).sum())
    return out


def version_rows(meta, shared):
    rows = []
    for ver, ts in meta["time"].items():
        if not STABLE.match(ver):
            continue
        published = pd.Timestamp(ts)
        if not (WIN_START <= published < PRED_START):
            continue
        info = meta["versions"].get(ver, {})
        dist = info.get("dist", {})
        bundled = info.get("bundleDependencies") or info.get("bundledDependencies") or []
        rows.append({
            "purl": f"pkg:npm/{meta['name']}@{ver}",
            "created_at": meta["time"].get("created"),
            "published_at": ts,
            "unpacked_size": dist.get("unpackedSize"),
            "file_count": dist.get("fileCount"),
            "dependency_count": len(info.get("dependencies") or {}),
            "optional_dependency_count": len(info.get("optionalDependencies") or {}),
            "peer_dependency_count": len(info.get("peerDependencies") or {}),
            "dev_dependency_count": len(info.get("devDependencies") or {}),
            "bundled_dependency_count": 0 if bundled is True else len(set(bundled)),
            **shared,
        })
    return rows


# ---------------------------------------------------------------- main
def main():
    total = len(PACKAGES)
    start_total = time.time()
    header = not os.path.exists(OUT)
    done = set()
    if not header:                           
        done = set(pd.read_csv(OUT, usecols=["purl"]).purl
                   .str.rsplit("@", n=1).str[0].str.removeprefix("pkg:npm/"))

    for idx, repo in enumerate(PACKAGES, start=1):
        pkg = repo.rsplit("/", 1)[-1]
        print(f"\n[{idx}/{total}] == {repo} -> {pkg} ==")
        if pkg in done:
            print("     -> bereits in CSV vorhanden, übersprungen.")
            continue

        t0 = time.time()
        try:
            print("     -> Lade npm Metadaten & Downloads...")
            reg = requests.get(f"https://registry.npmjs.org/{pkg.replace('/', '%2F')}",
                               timeout=30)
            reg.raise_for_status()
            meta = reg.json()
            npm_data = npm_monthly(meta["name"], meta)

            gh_data = github_monthly(repo)

            print("     -> Lade Security Advisories...")
            ghsa_data = ghsa_features(meta["name"])

            shared = {**npm_data, **gh_data, **ghsa_data}
            rows = version_rows(meta, shared)
        except Exception as exc:
            print(f"     -> ÜBERSPRUNGEN wegen Fehler: {exc}")
            continue

        elapsed = time.time() - t0
        if rows:
            pd.DataFrame(rows).to_csv(OUT, mode="a", index=False, header=header,
                                      na_rep="null", encoding="utf-8")
            header = False
            print(f"     -> Fertig in {elapsed:.1f}s ({len(rows)} Versionen gespeichert)")
        else:
            print(f"     -> Fertig in {elapsed:.1f}s (0 Versionen im Zeitfenster gefunden)")

    total_elapsed = time.time() - start_total
    print(f"\nAlle {total} Pakete abgeschlossen in {total_elapsed:.1f}s!")


if __name__ == "__main__":
    main()