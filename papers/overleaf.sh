#!/usr/bin/env bash
# overleaf.sh -- keep papers/<paper>/ in this repo and an Overleaf project in step (git subtree).
#
#   papers/overleaf.sh link <paper> <overleaf-git-url>   once per paper
#   papers/overleaf.sh pull <paper>    merge Overleaf's edits (Stephan's, Dylan's) into this repo
#   papers/overleaf.sh push <paper>    send this repo's commits under papers/<paper>/ to Overleaf
#   papers/overleaf.sh sync <paper>    pull, then push
#   papers/overleaf.sh status <paper>  who is ahead, what Overleaf changed, any hold (read-only)
#   papers/overleaf.sh hold <paper> "why"   block every push (Stephan is editing); release lifts it
#   papers/overleaf.sh release <paper>
#   papers/overleaf.sh push <paper> --now   push even if Overleaf changed in the last 30 min
#
# The Overleaf project's git URL is under Menu > Integrations > Git; auth is user "git" and a
# token from Account Settings > Git integration. Never forces: if Overleaf moved on, push is
# refused and you pull first. Commit (or stash) your own changes before pull or sync.
#
# Protections for shared editing (Stephan in Overleaf, Claude in this repo):
#   - pull tags HEAD first (overleaf-backup/<paper>-<time>) and, on a conflict, aborts the merge so
#     the repo is left clean and the conflicted files are listed. Nobody resolves it blindly.
#   - push fetches first, refuses if Overleaf is ahead, refuses while a hold is set, refuses if
#     Overleaf's last commit is under 30 min old (someone is probably editing; --now overrides),
#     and prints exactly which files it is about to change.
#   - hold/release: a tracked flag file papers/.overleaf-hold-<paper>, so it holds on every machine.
set -euo pipefail
cd "$(dirname "$0")/.."
cmd=${1:-}; paper=${2:-}; flag=${3:-}
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
  if git rev-parse -q --verify "$remote/main" >/dev/null; then
    # A fresh Overleaf project has its own first commit: record it as a parent and keep our tree,
    # so the push is a fast-forward and nothing on either side is forced.
    tip=$(git commit-tree "$split^{tree}" -p "$split" -p "$remote/main" \
          -m "Link $prefix to Overleaf (replaces the blank project's files)")
  else
    tip=$split
  fi
  git push "$remote" "$tip:refs/heads/main"
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
  tag=overleaf-backup/$paper-$(date +%Y%m%d-%H%M%S)
  git tag "$tag"
  if ! git subtree pull -q --prefix="$prefix" "$remote" main -m "Merge Overleaf edits to $paper"; then
    echo "CONFLICT merging Overleaf edits to $paper in:"
    git diff --name-only --diff-filter=U | sed 's/^/  /'
    git merge --abort 2>/dev/null || git reset -q --hard "$tag"
    echo "Merge aborted, repo back to $tag. Do NOT resolve this automatically: tell Dylan, who decides"
    echo "per file (or have Stephan accept/reject in Overleaf first, then pull again)."
    exit 1
  fi
  echo "pulled; backup tag $tag"
  git diff --stat "$tag" HEAD -- "$prefix" | tail -n 20
  ;;
status)
  git fetch -q "$remote"
  echo "Overleaf-only commits (not in repo): $(git rev-list --count HEAD.."$remote"/main)"
  git log --format='  %h %an, %ar: %s' -n 5 "$remote"/main
  [[ -f papers/.overleaf-hold-$paper ]] && echo "HOLD: $(cat papers/.overleaf-hold-$paper)" || echo "no hold"
  split=$(git subtree split -q --prefix="$prefix" 2>/dev/null || true)
  [[ -n $split ]] && { echo "Files a push would change:"; git diff --stat "$remote"/main "$split" | sed 's/^/  /'; }
  ;;
hold)
  echo "${3:-held by $(git config user.name) $(date +%F\ %H:%M)}" > papers/.overleaf-hold-$paper
  git add papers/.overleaf-hold-$paper
  git commit -q -m "Hold pushes to Overleaf $paper" -- papers/.overleaf-hold-$paper
  echo "hold set (commit it to origin main so other machines see it): pushes are blocked until release"
  ;;
release)
  git rm -q -f papers/.overleaf-hold-$paper
  git commit -q -m "Release hold on Overleaf $paper" -- papers/.overleaf-hold-$paper
  echo "hold released"
  ;;
push)
  [[ ! -f papers/.overleaf-hold-$paper ]] || { echo "REFUSED: hold set: $(cat papers/.overleaf-hold-$paper)"; echo "(papers/overleaf.sh release $paper)"; exit 1; }
  git fetch -q "$remote"
  if [[ $(git rev-list --count HEAD.."$remote"/main) != 0 ]]; then
    echo "REFUSED: Overleaf has commits this repo lacks. Run: papers/overleaf.sh pull $paper"; exit 1
  fi
  age=$(( $(date +%s) - $(git log -1 --format=%ct "$remote"/main) ))
  if (( age < 1800 )) && [[ $flag != --now ]]; then
    echo "REFUSED: Overleaf's last commit is $((age/60)) min old ($(git log -1 --format='%an: %s' "$remote"/main))."
    echo "Someone is probably editing. Ask Dylan, or re-run with --now."; exit 1
  fi
  split=$(git subtree split -q --prefix="$prefix")
  echo "This push changes in Overleaf:"; git diff --stat "$remote"/main "$split" | sed 's/^/  /'
  git push -q "$remote" "$split":refs/heads/main
  ;;
sync)
  "$0" pull "$paper" && "$0" push "$paper" ${3:-}
  ;;
*)
  sed -n 2,11p "$0"; exit 1 ;;
esac
