import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.services.leaderboard import fetch_and_store_benchmarks

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


def start_scheduler():
    scheduler.add_job(
        fetch_and_store_benchmarks,
        trigger=IntervalTrigger(hours=4),
        id="fetch_benchmarks",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started: benchmark fetch every 4 hours")


def stop_scheduler():
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")
