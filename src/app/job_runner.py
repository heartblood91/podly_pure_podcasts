"""Run a single processing job in a short-lived child process.

Usage: python -m app.job_runner JOB_ID POST_GUID

The parent JobsManager holds the global processing lock while this runs and
fails the job if the child exits with a nonzero status (crash, OOM kill).
"""

import logging
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m app.job_runner JOB_ID POST_GUID", file=sys.stderr)
        return 2
    job_id, post_guid = args

    # pylint: disable=import-outside-toplevel
    from app import create_job_runner_app
    from app.extensions import db
    from app.jobs_manager import run_job
    from podcast_processor.processing_status_manager import ProcessingStatusManager

    logger = logging.getLogger("global_logger")
    app = create_job_runner_app()
    with app.app_context():
        logger.info("[JOB_RUNNER] Starting job_id=%s post_guid=%s", job_id, post_guid)
        status_manager = ProcessingStatusManager(db_session=db.session, logger=logger)
        run_job(job_id, post_guid, status_manager)
        logger.info("[JOB_RUNNER] Finished job_id=%s", job_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
