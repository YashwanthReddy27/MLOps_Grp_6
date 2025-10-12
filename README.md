# MLOps_Grp_6

## Contributing: Branching workflow

We contribute using a Git feature-branch workflow. The `main` branch stays stable; all changes go through pull requests.

- Create a branch from `main`.
	- Naming: `feature/<short-name>` for new work, `fix/<short-name>` for bug fixes, `docs/<short-name>` for documentation.
- Commit with clear messages; keep PRs small and focused.
- Push and open a PR to `main`; request at least one reviewer.
- Ensure checks/CI pass and address feedback.
- Use "Squash and merge" when approved, then delete the branch.

Quick start:

```powershell
git checkout main
git pull
git checkout -b feature/your-change
# make changes
git add .
git commit -m "feat: describe your change"
git push origin feature/your-change
# open a PR to main
```
