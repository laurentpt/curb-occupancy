"""Write requirements.lock.txt: the installed versions of requirements.in's packages and their
dependencies (only those: optional packages installed in the env, such as ultralytics, stay
out of the lock). Run inside the env: `make lock`."""
import re
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main():
    direct, index = [], []
    for line in (ROOT / "requirements.in").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line.startswith("--"):
            index.append(line)
        elif line:
            direct.append(norm(Requirement(line).name))
    dists = {norm(d.metadata["Name"]): d for d in metadata.distributions()}
    missing = [d for d in direct if d not in dists]
    if missing:
        raise SystemExit(f"not installed: {missing} (pip install -r requirements.in)")
    seen, todo = set(), list(direct)
    while todo:
        n = todo.pop()
        if n in seen or n not in dists:
            continue
        seen.add(n)
        for r in dists[n].requires or []:
            req = Requirement(r)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                todo.append(norm(req.name))
    lines = index + [f"{dists[n].metadata['Name']}=={dists[n].version}" for n in sorted(seen)]
    (ROOT / "requirements.lock.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(seen)} packages -> requirements.lock.txt")


if __name__ == "__main__":
    main()
