from pydriller import Repository

from config import C_FILE_EXTENSIONS, MAX_CHANGE
from helpers import get_logger
from .modifier_extractor import ModifierExtractor

logger = get_logger(__name__)

class CommitExtractor():
    def __init__(self, repo_dir: str, commit_id: str, extract=True):
        self._repository_dir = repo_dir
        self.commit_id = commit_id

        logger.info(f"extract commit: {self.commit_id}")
        self._pydriller_repo = Repository(self._repository_dir, single=self.commit_id)
        self.commit = next(self._pydriller_repo.traverse_commits())
        self.modifies = list()
        # if len(self.commit.modified_files) >= MAX_CHANGE:
        #     self.modifies = self.commit.modified_files
        #     return
        for modified in self.commit.modified_files:
            ext = modified.filename.rsplit('.', 1)
            if len(ext) < 2 or (len(ext) > 1 and ext[1] not in C_FILE_EXTENSIONS):
                continue
            self.modifies.append(ModifierExtractor(modified, commit_id, extract))
        self.meta_data = self.get_meta_data()

    def get_meta_data(self):
        pass
