#!/usr/bin/env python3

"""
Automerge Script for LLVM Downstream Integration

This script automatically merges commits from an upstream LLVM branch into a
downstream branch, handling each commit individually as a separate merge commit.

Key Features:
- Merges commits one-by-one, preserving history and enabling easier bisection
- Prefixes each merge commit with "Automerge:" for easy identification
- Handles merge conflicts by creating pull requests for manual resolution
- Respects .automerge_ignore file for paths to exclude from merging

For a full list of available options, run:
    python3 automerge.py --help

Usage Examples:

1. Merge all new commits from a branch:
   python3 automerge.py \\
     --project-name arm/arm-toolchain \\
     --from-branch upstream/release/22.x \\
     --to-branch release/arm-software/22.x \\
     --repo-path /path/to/repo

2. Merge from a specific commit (useful for release points):
   python3 automerge.py \\
     --project-name arm/arm-toolchain \\
     --from-branch 4434dabb69916856b824f68a64b029c67175e532 \\
     --to-branch release/arm-software/22.x \\
     --repo-path /path/to/repo

3. Dry run to preview changes without pushing:
   python3 automerge.py \\
     --project-name arm/arm-toolchain \\
     --from-branch upstream/release/22.x \\
     --to-branch release/arm-software/22.x \\
     --repo-path /path/to/repo \\
     --dry-run \\
     --verbose

Notes:
- --from-branch accepts branch names, commit hashes, or tags
- When using a commit hash as --from-branch, all commits from the merge-base
  up to and including that commit will be merged
- The script pushes to the 'origin' remote unless --dry-run is specified
"""

import argparse
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


MERGE_CONFLICT_LABEL = "automerge_conflict"
AUTOMERGE_BRANCH = "automerge"
REMOTE_NAME = "origin"
MERGE_IGNORE_PATHSPEC_FILE = Path(__file__).parent / ".automerge_ignore"


class MergeConflictError(Exception):
    """
    An exception representing a failed merge from upstream due to a conflict.
    """

    def __init__(self, commit_hash: str) -> None:
        super().__init__()
        self.commit_hash = commit_hash


class Git:
    """
    A helper class for running Git commands on a repository that lives in a
    specific path.
    """

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path

    def get_repo_path(self) -> Path:
        return self.repo_path

    def run_cmd(self, args: list[str], check: bool = True) -> str:
        git_cmd = ["git", "-C", str(self.repo_path)] + args
        logger.debug("Running git command: %s", git_cmd)
        git_process = subprocess.run(
            git_cmd, check=check, capture_output=True, text=True
        )
        logger.debug("Stdout:\n%s\nStderr:\n%s", git_process.stdout, git_process.stderr)
        return git_process.stdout


def is_merge_in_progress(git_repo: Git) -> bool:
    """
    Check if a Git merge operation is currently in progress.

    Returns:
        bool: True if a merge is in progress, False otherwise.
    """
    # The `.git/MERGE_HEAD` file only exists when a merge operation is in progress.
    merge_head_path = Path(git_repo.repo_path) / ".git" / "MERGE_HEAD"
    return merge_head_path.exists()


def restore_changes_to_ignored_files(git_repo: Git, ignore_list: list[str]) -> None:
    """
    Restore files in the ignore list to their state in the target branch.

    This function handles files specified in .automerge_ignore by:
    1. Keeping the target branch version for conflicting files
    2. Ensuring deleted files remain deleted
    3. Restoring all other ignored files to their original state

    Args:
        git_repo: Git repository helper instance.
        ignore_list: List of file paths to ignore during merge.
    """
    if not ignore_list:
        return
    # First, deal with any conflicting changes to files in the ignore list,
    # keeping the version from the destination branch
    git_repo.run_cmd(["restore", "--ours", "--worktree"] + ignore_list)
    # Next, any files still unmerged are the ones deleted on the destination branch.
    # Make sure they stay deleted.
    ls_files_output = git_repo.run_cmd(
        ["diff", "--name-only", "--diff-filter=U", "--"] + ignore_list
    )
    deleted_by_us = ls_files_output.splitlines()
    if deleted_by_us:
        git_repo.run_cmd(["rm"] + deleted_by_us)
    # Finally, restore all other ignored files
    git_repo.run_cmd(["restore", "--staged", "--worktree"] + ignore_list)


def has_unresolved_conflicts(git_repo: Git) -> bool:
    """
    Check if there are any unresolved merge conflicts.

    Returns:
        bool: True if unresolved conflicts exist, False otherwise.
    """
    diff_output = git_repo.run_cmd(["diff", "--name-only", "--diff-filter=U"])
    diff_output = diff_output.strip()
    return bool(diff_output)


def prefix_current_commit_message(git_repo: Git) -> None:
    """
    Prefix the current commit message with "Automerge: " for easy identification.

    Args:
        git_repo: Git repository helper instance.
    """
    log_output = git_repo.run_cmd(
        ["log", "HEAD", "--max-count=1", "--pretty=format:%B"]
    )
    commit_msg = f"Automerge: {log_output}"
    git_repo.run_cmd(["commit", "--amend", "--message=" + commit_msg])


def merge_commit(
    git_repo: Git,
    to_branch: str,
    commit_hash: str,
    ignored_paths: list[str],
    dry_run: bool,
    verbose: bool,
) -> None:
    """
    Merge a single commit into the target branch.

    Performs a no-fast-forward merge of the specified commit, handles files in
    the ignore list, and pushes the result to the remote repository (unless
    running in dry-run mode).

    Args:
        git_repo: Git repository helper instance.
        to_branch: Name of the target branch to merge into.
        commit_hash: Hash of the commit to merge.
        ignored_paths: List of file paths to ignore during merge.
        dry_run: If True, skip pushing to remote repository.
        verbose: If True, log additional debug information.

    Raises:
        MergeConflictError: If the merge results in unresolved conflicts.
        RuntimeError: If git merge fails unexpectedly.
    """
    logger.info("Merging commit %s into %s", commit_hash, to_branch)
    git_repo.run_cmd(["switch", to_branch])
    if verbose:
        current_head = git_repo.run_cmd(
            ["log", "--no-walk", "HEAD", "--pretty=reference"]
        )
        logger.debug("Current HEAD of %s is %s", to_branch, current_head)
    # `git merge` will return a non-zero exit status if there's a conflict, but
    # the conflict might be resolved by applying our ignore list. We work around
    # that by not checking the exist status and validating that a merge is in
    # progress after `git merge` runs.
    git_repo.run_cmd(["merge", commit_hash, "--no-commit", "--no-ff"], check=False)
    if not is_merge_in_progress(git_repo):
        raise RuntimeError("Unexpected error occurred when running git merge")
    restore_changes_to_ignored_files(git_repo, ignored_paths)
    if has_unresolved_conflicts(git_repo):
        logger.info("Merge failed")
        git_repo.run_cmd(["merge", "--abort"])
        raise MergeConflictError(commit_hash)
    git_repo.run_cmd(["commit", "--reuse-message", commit_hash])
    prefix_current_commit_message(git_repo)
    if verbose:
        merge_reference = git_repo.run_cmd(
            ["log", "--no-walk", "HEAD", "--pretty=reference"]
        )
        logger.debug("Merge commit finalized: %s", merge_reference)
    if dry_run:
        logger.info("Dry run. Skipping push into remote repository.")
    else:
        git_repo.run_cmd(["push", REMOTE_NAME, to_branch])
    logger.info("Merge successful")


def create_pull_request(git_repo: Git, to_branch: str) -> None:
    """
    Create a pull request for a merge conflict using GitHub CLI.

    Args:
        git_repo: Git repository helper instance.
        to_branch: Base branch for the pull request.
    """
    logger.info("Creating Pull Request")
    log_output = git_repo.run_cmd(
        ["log", "HEAD", "--max-count=1", "--pretty=format:%s"]
    )
    pr_title = f"Automerge conflict: {log_output}"
    subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--head",
            AUTOMERGE_BRANCH,
            "--base",
            to_branch,
            "--fill",
            "--title",
            pr_title,
            "--label",
            MERGE_CONFLICT_LABEL,
        ],
        cwd=git_repo.get_repo_path(),
        check=True,
    )


def process_conflict(
    git_repo: Git, commit_hash: str, to_branch: str, dry_run: bool
) -> None:
    """
    Handle a merge conflict by creating a branch and pull request for manual resolution.

    Creates a new branch from the conflicting commit and, unless in dry-run mode,
    pushes it to the remote and creates a PR labeled for automerge conflicts.

    Args:
        git_repo: Git repository helper instance.
        commit_hash: Hash of the commit that caused the conflict.
        to_branch: Base branch for the pull request.
        dry_run: If True, skip pushing and creating the PR.
    """
    logger.info("Processing conflict for %s", commit_hash)
    git_repo.run_cmd(["switch", "--force-create", AUTOMERGE_BRANCH, commit_hash])
    if dry_run:
        logger.info("Dry run, skipping push and creation of PR.")
        return
    git_repo.run_cmd(["push", REMOTE_NAME, AUTOMERGE_BRANCH])
    logger.info("Publishing Pull Request for conflict")
    create_pull_request(git_repo, to_branch)


def get_merge_commit_list(git_repo: Git, from_branch: str, to_branch: str) -> list[str]:
    """
    Calculate the list of commits to be merged from source to target branch.

    This function:
    1. Finds the merge-base (common ancestor) between source and target branches
    2. Gets all commits from merge-base to from_branch
    3. Returns them in chronological order (oldest first)

    Args:
        git_repo: Git repository helper instance.
        from_branch: Source branch, tag, or commit hash to merge from. Can be:
                    - A branch name (e.g., "upstream/release/22.x")
                    - A commit hash (e.g., "4434dabb6991...")
                    - A tag name (e.g., "llvmorg-22.1.0")
        to_branch: Target branch to merge into.

    Returns:
        List of commit hashes in chronological order (oldest first).

    Examples:
        # Merge all commits from a branch
        commits = get_merge_commit_list(repo, "upstream/main", "my-branch")

        # Merge from a specific commit (all commits from merge-base to that commit)
        commits = get_merge_commit_list(repo, "4434dabb6991", "my-branch")
    """
    logger.info(
        "Calculating list of commits to be merged from %s to %s", from_branch, to_branch
    )
    merge_base_output = git_repo.run_cmd(["merge-base", from_branch, to_branch])
    merge_base_commit = merge_base_output.strip()
    log_output = git_repo.run_cmd(
        ["log", f"{merge_base_commit}..{from_branch}", "--pretty=format:%H"]
    )
    commit_list = log_output.strip()
    if not commit_list:
        logger.info("No commits to be merged")
        return []
    commit_list = commit_list.split("\n")
    commit_list.reverse()
    logger.info("Found %d commits to be merged", len(commit_list))
    return commit_list


def pr_exist_for_label(project_name: str, label: str) -> bool:
    """
    Check if any open pull requests exist with the specified label.

    Args:
        project_name: GitHub repository in OWNER/REPO format.
        label: Label to search for.

    Returns:
        bool: True if at least one open PR with the label exists.
    """
    logger.info("Fetching list of open PRs for label '%s'.", label)
    gh_process = subprocess.run(
        ["gh", "pr", "list", "--label", label, "--repo", project_name, "--json", "id"],
        check=True,
        capture_output=True,
        text=True,
    )
    return len(json.loads(gh_process.stdout)) > 0


def is_worktree_clean(git_repo: Git) -> bool:
    """
    Check if the Git worktree is clean (no uncommitted changes).

    Returns:
        bool: True if worktree is clean, False if there are uncommitted changes.
    """
    # `git status --porcelain` returns an empty result if worktree is clean
    status_output = git_repo.run_cmd(["status", "--porcelain"]).strip()
    return len(status_output) == 0


def main():
    arg_parser = argparse.ArgumentParser(
        prog="automerge",
        description="A script that automatically merges individual commits from one branch into another.",
    )
    arg_parser.add_argument(
        "--project-name",
        required=True,
        metavar="OWNER/REPO",
        help="The name of the project in GitHub.",
    )
    arg_parser.add_argument(
        "--from-branch",
        required=True,
        metavar="BRANCH_OR_COMMIT",
        help="Source branch, commit hash, or tag to merge from. When a commit hash is provided, all commits from the merge-base up to and including that commit will be merged.",
    )
    arg_parser.add_argument(
        "--to-branch",
        required=True,
        metavar="BRANCH_NAME",
        help="The target branch for merging incoming commits",
    )
    arg_parser.add_argument(
        "--repo-path",
        metavar="PATH",
        default=Path.cwd(),
        help="The path to the existing local checkout of the repository (default: working directory)",
    )
    arg_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process changes locally, but don't merge them into the remote repository and don't create PRs",
    )
    arg_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose log messages during automerge run",
    )

    args = arg_parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        if pr_exist_for_label(args.project_name, MERGE_CONFLICT_LABEL):
            logger.error("There are pending automerge PRs. Cannot continue.")
            sys.exit(1)
        logger.info("No pending merge conflicts. Proceeding with automerge.")

        git_repo = Git(args.repo_path)

        if not is_worktree_clean(git_repo):
            logger.error("The repository worktree is not clean. Cannot continue.")
            sys.exit(1)

        with open(MERGE_IGNORE_PATHSPEC_FILE) as ignored_paths_file:
            ignored_paths = ignored_paths_file.read().splitlines()

        merge_commits = get_merge_commit_list(
            git_repo, args.from_branch, args.to_branch
        )
        for commit_hash in merge_commits:
            merge_commit(
                git_repo,
                args.to_branch,
                commit_hash,
                ignored_paths,
                args.dry_run,
                args.verbose,
            )
    except MergeConflictError as conflict:
        process_conflict(
            git_repo,
            conflict.commit_hash,
            args.to_branch,
            args.dry_run,
        )
    except subprocess.CalledProcessError as error:
        logger.error(
            'Failed to run command: "%s"\nstdout:\n%s\nstderr:\n%s',
            shlex.join(error.cmd),
            error.stdout,
            error.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
