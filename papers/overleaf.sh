#!/usr/bin/env bash
# overleaf.sh -- keep papers/<paper>/ in this repo and an Overleaf project in step (git subtree).
#
#   papers/overleaf.sh link <paper> <overleaf-git-url>   once per paper
#   papers/overleaf.sh pull <paper>    merge Overleaf's edits (Stephan's, Dylan's) into this repo
#   papers/overleaf.sh push <paper>    send this repo's commits under papers/<paper>/ to Overleaf
#   papers/overleaf.sh sync <paper>    pull, then push
#
# The Overleaf project's git URL is under Menu > Integrations > Git; auth is user "git" and a
# token from Account Settings > Git integration. Never forces: if Overleaf moved on, push is
# refused and you pull first. Commit (or stash) your own changes before pull or sync.
set -euo pipefail
cd "$(dirname "$0")/.."
cmd=${1:-}; paper=${2:-}
[[ -n $cmd && -d papers/$paper ]] || { sed -n 2,11p "$0"; exit 1; }
remote=overleaf-$paper
prefix=papers/$paper

clean() {
  git diff --quiet && git diff --cached --quiet || { echo "commit or stash your changes first"; exit 1; }
}

case $cmd in
link)
  url=${3:?overleaf git url}
  clean
  git remote get-url "$remote" >/dev/null 2>&1 || git remote add "$remote" "$url"
  git fetch "$remote"
  split=$(git subtree split -q --prefix="$prefix")
  if git rev-parse -q --verify "$remote/master" >/dev/null; then
    # A fresh Overleaf project has its own first commit: record it as a parent and keep our tree,
    # so the push is a fast-forward and nothing on either side is forced.
    tip=$(git commit-tree "$split^{tree}" -p "$split" -p "$remote/master" \
          -m "Link $prefix to Overleaf (replaces the blank project's files)")
  else
    tip=$split
  fi
  git push "$remote" "$tip:refs/heads/master"
  git fetch -q "$remote"
  # Join the two histories as `git subtree add` would (same tree, so nothing changes on disk);
  # later subtree pulls and splits find each other through this commit's trailers.
  join=$(git commit-tree "HEAD^{tree}" -p HEAD -p "$tip" -m "Link $prefix to Overleaf

git-subtree-dir: $prefix
git-subtree-mainline: $(git rev-parse HEAD)
git-subtree-split: $tip")
  git merge -q --ff-only "$join"
  echo "linked $prefix <-> $remote ($url)"
  ;;
pull)
  clean
  git subtree pull -q --prefix="$prefix" "$remote" master -m "Merge Overleaf edits to $paper"
  ;;
push)
  git subtree push -q --prefix="$prefix" "$remote" master
  ;;
sync)
  "$0" pull "$paper" && "$0" push "$paper"
  ;;
*)
  sed -n 2,11p "$0"; exit 1 ;;
esac
