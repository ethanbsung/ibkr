1. ~~Automate preflight check to each markets open~~ DONE 2026-06-09 — cron
   runs of preflight_check.py at 9:30 PM ET Sun–Thu (Asia) and 3:30 AM ET
   Mon–Fri (Eurex) in crontab_vps.txt (remember: `crontab crontab_vps.txt` on
   the VPS to install).
2. Implement carry backtest + eventually live
