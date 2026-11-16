#!/usr/bin/env python3
"""Tests for predict.py.      Usage:  python test_predict.py dataset_initial.csv

Run train.py first. The test runs predict.py like the course would (as a separate
program) and checks: output format (floats), the 10 second limit, one input file per
package, and many odd inputs. Odd inputs are tested twice: with XGBoost and with the
simple fallback rule.
"""
import csv
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICT = os.path.join(HERE, "predict.py")
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "dataset_initial.csv")
EXPECTED_HEADER = ["target_ghsa_%s_count_90d" % s
                   for s in ("low", "moderate", "high", "critical", "unknown")]
failures = []
tmp = tempfile.mkdtemp()


def run(args, env_extra=None):
    env = dict(os.environ)
    env.update(env_extra or {})
    start = time.time()
    p = subprocess.run([sys.executable, PREDICT] + args, capture_output=True, text=True,
                       timeout=10, env=env)
    return p, time.time() - start


def parse_output(text, label):
    """Header with the 5 target columns, exactly one record, floats >= 0 (like 0.2000)."""
    rows = list(csv.reader(text.splitlines()))
    ok = (len(rows) == 2 and rows[0] == EXPECTED_HEADER and len(rows[1]) == 5
          and all(re.fullmatch(r"\d+\.\d+", v) for v in rows[1]))
    if not ok:
        failures.append("%s: bad output %r" % (label, rows))
        return None
    return [float(v) for v in rows[1]]


def read_output(path, label):
    with open(path, newline="", encoding="utf-8") as f:
        return parse_output(f.read(), label)


def write_csv(path, header, rows, delimiter=",", bom=False):
    with open(path, "w", newline="", encoding="utf-8-sig" if bom else "utf-8") as f:
        w = csv.writer(f, delimiter=delimiter, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


with open(DATA, newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = list(reader)
purl_i = header.index("purl")
by_package = {}
for r in data:
    by_package.setdefault(r[purl_i].rsplit("@", 1)[0], []).append(r)
target_idx = [header.index(c) for c in EXPECTED_HEADER]

# ------------------------------------------------------------------ real packages
print("== Real packages, one input file per package (XGBoost) ==")
print("%-18s %-34s %-18s %s" % ("package", "predicted (low,mod,high,crit,unk)", "actual", "seconds"))
slowest, used_fallback = 0.0, False
for name, rows in sorted(by_package.items()):
    inp, out = os.path.join(tmp, "in.csv"), os.path.join(tmp, "out.csv")
    write_csv(inp, header, rows)
    p, secs = run([inp, out])
    slowest = max(slowest, secs)
    if p.returncode != 0 or not os.path.isfile(out):
        reason = ("exit code %s" % p.returncode if p.returncode != 0
                  else "predict.py returned success but did not create the output file")
        details = "\n".join(
            label + value for label, value in (("stdout: ", p.stdout.strip()),
                                                ("stderr: ", p.stderr.strip())) if value
        )
        failures.append("%s: %s%s" % (name, reason, ("\n" + details) if details else ""))
        print("%-18s FAILED (%s)" % (name.replace("pkg:npm/", ""), reason))
        continue
    if "WARNING" in p.stderr:
        used_fallback = True
    pred = read_output(out, name)
    actual = [int(float(rows[0][i])) for i in target_idx]
    print("%-18s %-34s %-18s %.2f" % (name.replace("pkg:npm/", ""), pred, actual, secs))
if used_fallback:
    print("\n!! WARNING: XGBoost was NOT used (the simple fallback rule ran). Reason:")
    print("   ", p.stderr.strip())
    print("   Fix: pip install xgboost numpy   and   python train.py dataset_initial.csv")
else:
    print("\nXGBoost was used for all packages (no fallback warnings).")

# ------------------------------------------------------------------ odd inputs
MODES = [("XGBoost", {}), ("fallback rule", {"PCOM_NO_XGBOOST": "1"})]
sample = by_package.get("pkg:npm/fastify") or max(by_package.values(), key=len)

for mode, env in MODES:
    print("\n== Odd inputs, mode: %s ==" % mode)

    def case(label, make_input, expect=None):
        inp, out = os.path.join(tmp, "case_in.csv"), os.path.join(tmp, "case_out.csv")
        if os.path.exists(out):
            os.remove(out)
        make_input(inp)
        p, secs = run([inp, out], env)
        if p.returncode != 0 or not os.path.isfile(out):
            reason = ("exit code %s" % p.returncode if p.returncode != 0
                      else "predict.py did not create the output file")
            details = "\n".join(
                label + value for label, value in (("stdout: ", p.stdout.strip()),
                                                    ("stderr: ", p.stderr.strip())) if value
            )
            failures.append("[%s] %s: %s%s" %
                            (mode, label, reason, ("\n" + details) if details else ""))
            print("  FAIL", label)
            return None
        got = read_output(out, "[%s] %s" % (mode, label))
        if expect is not None and got != expect:
            failures.append("[%s] %s: expected %s got %s" % (mode, label, expect, got))
        print("  ok  ", label, "->", got)
        return got

    def with_junk_targets(p):
        rows = [list(r) for r in sample]
        for r in rows:
            for i in target_idx:
                r[i] = "999"
        write_csv(p, header, rows)

    ref = case("reference package (all versions)", lambda p: write_csv(p, header, sample))
    no_data = case("header only (no records)", lambda p: write_csv(p, header, []))
    case("completely empty file", lambda p: open(p, "w").close(), no_data)
    case("input file does not exist", lambda p: None, no_data)
    case("only the purl column", lambda p: write_csv(p, ["purl"], [["pkg:npm/foo@1.0.0"]]), no_data)
    case("NULL / empty / text instead of numbers",
         lambda p: write_csv(p, ["purl", "ghsa_high_count_M0", "ghsa_low_count_M1", "package_downloads_M0"],
                             [["pkg:npm/foo@1.0.0", "NULL", "", "abc"]]), no_data)
    case("semicolon separated + BOM", lambda p: write_csv(p, header, sample, ";", True), ref)
    case("versions in random order", lambda p: write_csv(p, header, list(reversed(sample))), ref)
    case("target columns filled with junk are ignored", with_junk_targets, ref)
    case("scoped package purl",
         lambda p: write_csv(p, ["purl", "ghsa_high_count_M0"],
                             [["pkg:npm/%40scope/foo@2.0.0", "8"], ["pkg:npm/%40scope/foo@10.0.0", "8"]]))
    if mode == "fallback rule":  # here the size of the numbers is predictable
        got = case("newest version wins by semver (1.10.0 > 1.9.0)",
                   lambda p: write_csv(p, ["purl", "ghsa_high_count_M0"],
                                       [["pkg:npm/foo@1.10.0", "8"], ["pkg:npm/foo@1.9.0", ""],
                                        ["pkg:npm/foo@1.2.0", "0"]]))
        if got is not None and got[2] < 1:
            failures.append("newest version was not used, got %s" % got)
        got = case("many recent advisories",
                   lambda p: write_csv(p, ["purl", "ghsa_high_count_M0", "ghsa_moderate_count_M3"],
                                       [["pkg:npm/foo@1.0.0", "40", "20"]]))
        if got is not None and not (got[2] >= 1 and got[1] >= 1):
            failures.append("many recent advisories should give values above 1, got %s" % got)

# ------------------------------------------------------------------ output options
print("\n== Output options ==")
inp = os.path.join(tmp, "n_in.csv")
write_csv(inp, header, sample)
nested = os.path.join(tmp, "new_folder", "pred.csv")
p, _ = run([inp, nested])
if p.returncode != 0 or not os.path.isfile(nested):
    reason = ("exit code %s" % p.returncode if p.returncode != 0
              else "predict.py did not create the output file")
    failures.append("output into a new folder failed (%s)\nstdout: %s\nstderr: %s"
                    % (reason, p.stdout.strip(), p.stderr.strip()))
else:
    print("  ok   output into a folder that does not exist yet")
p, _ = run([inp])
if p.returncode != 0 or parse_output(p.stdout, "print mode") is None:
    failures.append("print-to-screen mode failed")
else:
    print("  ok   no output path: the CSV is printed to the screen")

print("\nSlowest real run: %.2f seconds (limit: 10)" % slowest)
if slowest >= 10:
    failures.append("too slow")
if failures:
    print("\nFAILED:")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("\nALL TESTS PASSED" + ("  (but see the XGBoost warning above)" if used_fallback else ""))