#!/usr/bin/env bash
# overleaf.sh -- keep papers/<paper>/ in this repo and an Overleaf project in step (git subtree).
#
#   papers/overleaf.sh link <paper> <overleaf-git-url>   once per paper
#   papers/overleaf.sh pull <paper>    merge Overleaf's edits (Stephan's, Dylan's) into this repo
#   papers/overleaf.sh push <paper>    send this repo's commits under papers/<paper>/ to Overleaf
#   papers/overleaf.sh sync <paper>    pull, then push
#   papers/overleaf.sh status <paper>  who is ahead, what Overleaf changed, any hold (read-only)
#   papers/overleaf.sh adopt <paper>   one-off: record Overleaf's tip as merged when its text equals a repo version
#   papers/overleaf.sh hold <paper> "why"   block every push (Stephan is editing); release lifts it
#   papers/overleaf.sh release <paper>
#
# The Overleaf project's git URL is under Menu > Integrations > Git; auth is user "git" and a
# token from Account Settings > Git integration. Never forces: if Overleaf moved on, push is
# refused and you pull first. Commit (or stash) your own changes before pull or sync.
#
# Protections for shared editing (Stephan in Overleaf, Claude in this repo):
#   - pull tags HEAD first (overleaf-backup/<paper>-<time>) and, on a conflict, aborts the merge so
#     the repo is left clean and the conflicted files are listed. Nobody resolves it blindly.
#   - push fetches first, refuses if Overleaf has edits this repo has not merged (it lists them and the files they
#     changed), refuses while a hold is set, and prints exactly which files it is about to change. There is no time
#     guard any more (removed 2026-11-24: it measured the bridge's commit time, which a fetch resets, not the edit).
#   - push records the pushed commit as merged (a join commit, no file change), so the next pull starts from it.
#     Before 2026-11-24 it did not, and every pull after a push conflicted on the lines just pushed.
#   - hold/release: a tracked flag file papers/.overleaf-hold-<paper>, so it holds on every machine.
set -euo pipefail
cd "$(dirname "$0")/.."
cmd=${1:-}; paper=${2:-}; flag=${3:-}
[[ -n $cmd && -d papers/$paper ]] || { sed -n 2,12p "$0"; exit 1; }
remote=overleaf-$paper

# The remote comes from papers/overleaf_remotes.txt (tracked), so a fresh clone or another machine needs no setup.
url_of() { awk -v p="$paper" '$1 == p { print $2 }' papers/overleaf_remotes.txt 2>/dev/null; }
# An optional third column names the folder the project mirrors, when it is not papers/<paper>. Since 2026-11-23 the
# Paper I project shared with Stephan (paper1) carries the paper1_v2 text: pull and push for paper1 use papers/paper1_v2,
# and papers/paper1 stays in the repo only as the home of the figures and producers that v2 still uses.
dir_of() { awk -v p="$paper" '$1 == p && NF >= 3 { print $3 }' papers/overleaf_remotes.txt 2>/dev/null; }
prefix=papers/$(d=$(dir_of); echo "${d:-$paper}")
[[ $prefix == papers/$paper ]] || echo "($paper mirrors $prefix)"
if [[ $cmd != link ]] && ! git remote get-url "$remote" >/dev/null 2>&1; then
  u=$(url_of)
  [[ -n $u ]] || { echo "no Overleaf URL for $paper in papers/overleaf_remotes.txt (run: $0 link $paper <url>)"; exit 1; }
  git remote add "$remote" "$u" && echo "added remote $remote -> $u (from papers/overleaf_remotes.txt)"
fi

clean() {
  git diff --quiet && git diff --cached --quiet || { echo "commit or stash your changes first"; exit 1; }
}

# The commit a push sends: this repo's $prefix tree, with Overleaf's current tip as its only parent, so the push is
# always a fast-forward and never forces. The push is refused earlier unless every Overleaf commit is merged here, so
# nothing of Overleaf's is dropped. The message lists the repo commits since Overleaf's tip that touched $prefix.
# (This replaces `git subtree split`, which fails on this history; see join below.) Prints nothing if Overleaf
# already has this text.
overleaf_commit() {
  local tree; tree=$(git rev-parse "HEAD:$prefix")
  [[ $tree == $(git rev-parse "$remote/main^{tree}") ]] && return 0
  local msg; msg=$(printf 'Push from the repo (%s)\n\n%s' "$(git rev-parse --short HEAD)" \
    "$(git log --no-merges --format='- %s' "$remote"/main..HEAD -- "$prefix" | cut -c1-200)")
  git commit-tree "$tree" -p "$remote/main" -m "$msg"
}

# Record an Overleaf commit as merged into this repo's history, without changing any file: a commit with this repo's
# tree, parents HEAD and <split>, and the git-subtree trailers. Without it, the next pull merges from an older base and
# every line edited since then conflicts with itself (the "add/add bridge conflict" of 11-23 and 11-24). The branch is
# moved with update-ref, so a dirty working tree (another session's edits) is not touched.
# 2026-11-24 (evening): no git-subtree trailers in this message any more. A join is pushed to Overleaf with the next
# push (as a copy of the commit), so its trailers appeared twice in the history and `git subtree split` stopped with
# "fatal: cache for <commit> already exists". Pushes no longer use `git subtree split` (see overleaf_commit), and the
# pull only needs the Overleaf commit as a parent.
join() {
  local split=$1 head; head=$(git rev-parse HEAD)
  git merge-base --is-ancestor "$split" "$head" && return 0
  local j; j=$(git commit-tree "$head^{tree}" -p "$head" -p "$split" -m "Record Overleaf $paper at ${split:0:8} as merged (no file change)")
  git update-ref -m "overleaf.sh join $paper" "$(git symbolic-ref HEAD)" "$j" "$head"
  echo "recorded Overleaf ${split:0:8} as merged (${j:0:8}); push origin main to share it"
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
  [[ -n $(url_of) ]] || printf '%-10s %s\n' "$paper" "$url" >> papers/overleaf_remotes.txt
  echo "linked $prefix <-> $remote ($url); commit papers/overleaf_remotes.txt so other machines pick it up"
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
  [[ $(git rev-parse "HEAD:$prefix") == $(git rev-parse "$remote/main^{tree}") ]] \
    || { echo "Files a push would change:"; git diff --stat "$remote"/main "HEAD:$prefix" | sed 's/^/  /'; }
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
  # The check for edits: the fetch above brings every edit saved in Overleaf. If Overleaf has commits this repo has not
  # merged, someone edited there since the last pull, and the push is refused until they are pulled.
  if [[ $(git rev-list --count HEAD.."$remote"/main) != 0 ]]; then
    echo "REFUSED: Overleaf has edits this repo has not merged:"
    git log --format='  %h %an, %ar: %s' HEAD.."$remote"/main
    echo "Files they changed:"; git diff --stat HEAD..."$remote"/main | sed 's/^/  /'
    echo "Run: papers/overleaf.sh pull $paper"; exit 1
  fi
  split=$(overleaf_commit)
  [[ -n $split ]] || { echo "nothing to push: Overleaf already has this text"; exit 0; }
  echo "This push changes in Overleaf:"; git diff --stat "$remote"/main "$split" | sed 's/^/  /'
  git push -q "$remote" "$split":refs/heads/main
  join "$split"
  ;;
adopt)
  # One-off repair for a project pushed before push recorded its joins: if Overleaf's text is exactly a version this
  # repo already committed under $prefix (so Overleaf holds no edit of its own), record Overleaf's tip as merged.
  git fetch -q "$remote"
  tip=$(git rev-parse "$remote"/main); want=$(git rev-parse "$tip^{tree}")
  found=
  for c in $(git rev-list -n 200 HEAD -- "$prefix"); do
    [[ $(git rev-parse "$c:$prefix") == "$want" ]] && { found=$c; break; }
  done
  [[ -n $found ]] || { echo "REFUSED: Overleaf's text matches no recent repo version of $prefix: it has edits. Pull and resolve with Dylan."; exit 1; }
  echo "Overleaf ${tip:0:8} has the same text as repo commit ${found:0:8}: no Overleaf edits to lose."
  join "$tip"
  ;;
sync)
  "$0" pull "$paper" && "$0" push "$paper" ${3:-}
  ;;
*)
  sed -n 2,12p "$0"; exit 1 ;;
esac
