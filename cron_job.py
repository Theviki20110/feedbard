"""Entry point kept at repo root so an existing cron schedule pointing at
`python cron_job.py` keeps working unchanged. Actual logic lives in
feedbard/app.py."""

from feedbard.app import check_and_run

if __name__ == "__main__":
    check_and_run()
