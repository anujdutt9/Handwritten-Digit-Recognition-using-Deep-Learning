# Run after master branch is ready for push
git reset --hard `git log master --format="%H" -n 1`
git push github-fork master --force --no-verify
