# Copyright 2023 Databricks, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import shutil
from dataclasses import dataclass
from typing import List

from transformers.trainer_callback import TrainerCallback
from watchdog.utils.dirsnapshot import DirectorySnapshot, EmptyDirectorySnapshot

logger = logging.getLogger(__name__)


@dataclass
class SnapshotDiff:
    """A diff between two states of a directory tree."""

    files_created: List[str]
    files_deleted: List[str]
    files_modified: List[str]
    dirs_created: List[str]
    dirs_deleted: List[str]


# Note: copied from dbx sync
def compute_snapshot_diff(ref: DirectorySnapshot, snapshot: DirectorySnapshot) -> SnapshotDiff:
    """Computes a diff between two directory snapshots at different points in time.

    See SnapshotDiff for what changes this detects.  This ignores changes to attributes of directories,
    such as the modification time.  It also does not attempt to detect moves.  A move is represented
    as a delete and create.

    This diff is useful for applying changes to a remote blob storage system that supports CRUD operations.
    Typically these systems don't support move operations, which is why detecting moves is not a priority.

    Args:
        ref (DirectorySnapshot): what we're comparing against (the old snapshot)
        snapshot (DirectorySnapshot): the new snapshot

    Returns:
        SnapshotDiff: record of what has changed between the snapshots
    """
    created = snapshot.paths - ref.paths
    deleted = ref.paths - snapshot.paths

    # any paths that exist in both ref and snapshot may have been modified, so we'll need to do some checks.
    modified_candidates = ref.paths & snapshot.paths

    # check if file was replaced with a dir, or dir was replaced with a file
    for path in modified_candidates:
        snapshot_is_dir = snapshot.isdir(path)
        ref_is_dir = ref.isdir(path)
        if (snapshot_is_dir and not ref_is_dir) or (not snapshot_is_dir and ref_is_dir):
            created.add(path)
            deleted.add(path)

    # anything created or deleted is by definition not modified
    modified_candidates = modified_candidates - created - deleted

    # check if any of remaining candidates were modified based on size or mtime changes
    modified = set()
    for path in modified_candidates:
        if ref.mtime(path) != snapshot.mtime(path) or ref.size(path) != snapshot.size(path):
            modified.add(path)

    dirs_created = sorted([path for path in created if snapshot.isdir(path)])
    dirs_deleted = sorted([path for path in deleted if ref.isdir(path)])
    dirs_modified = sorted([path for path in modified if ref.isdir(path)])

    files_created = sorted(list(created - set(dirs_created)))
    files_deleted = sorted(list(deleted - set(dirs_deleted)))
    files_modified = sorted(list(modified - set(dirs_modified)))

    return SnapshotDiff(
        files_created=files_created,
        files_deleted=files_deleted,
        files_modified=files_modified,
        dirs_created=dirs_created,
        dirs_deleted=dirs_deleted,
    )


class LocalToOutputDirSync(TrainerCallback):
    """A callback for the Trainer that copies from a local directory to an output directory."""

    def __init__(self, *, local_dir: str, output_dir: str):
        self.local_dir = local_dir
        self.output_dir = output_dir
        self.is_first_sync = True

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.sync()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            self.sync()

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.sync()

    def _get_output_path(self, local_full_path: str) -> str:
        """Converts a local path to an output path.

        Args:
            local_full_path (str): the local path where the change occurred

        Raises:
            ValueError: the input path was not actually a local path

        Returns:
            str: the output path for the target file to copy/delete/modify/etc.
        """
        if not local_full_path.startswith(self.local_dir):
            raise ValueError(f"Expected to start with {self.local_dir}")
        rel_path = local_full_path[len(self.local_dir) :].strip("/")
        return os.path.join(self.output_dir, rel_path)

    def sync(self):
        """
        Performs an incremental sync from the local directory to the output directory by performing a diff of the
        local directory state compared to the previous state.
        """
        logger.info("Syncing from %s to %s", self.local_dir, self.output_dir)
        if self.is_first_sync:
            self.last_snapshot = EmptyDirectorySnapshot()
            self.is_first_sync = False

        snapshot = DirectorySnapshot(self.local_dir)

        diff = compute_snapshot_diff(ref=self.last_snapshot, snapshot=snapshot)

        for p in diff.dirs_deleted:
            target_path = self._get_output_path(p)
            logger.info("* Deleting dir: %s", target_path)

        for p in diff.dirs_created:
            target_path = self._get_output_path(p)
            os.makedirs(target_path)
            logger.info("* Creating dir: %s", target_path)

        for p in diff.files_deleted:
            target_path = self._get_output_path(p)
            logger.info("* Deleting file: %s", target_path)
            os.remove(target_path)

        for p in diff.files_created:
            target_path = self._get_output_path(p)
            logger.info("* Creating file: %s", target_path)
            shutil.copyfile(p, target_path)

        for p in diff.files_modified:
            target_path = self._get_output_path(p)
            logger.info("* Updating file: %s", target_path)
            shutil.copyfile(p, target_path)

        self.last_snapshot = snapshot
