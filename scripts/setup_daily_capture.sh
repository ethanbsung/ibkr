#!/bin/bash
# Setup automated daily capture at market close for accurate Sharpe ratio

echo "Setting up daily market close capture..."

# Create cron job for post-market capture (5:00 PM ET = 22:00 UTC)
CRON_JOB="0 22 * * 1-5 cd /Users/ethansung/quant/ibkr && python3 account_summary.py > /tmp/market_close_capture.log 2>&1"

# Add to crontab
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "Cron job added:"
echo "- Runs Monday-Friday at 5:00 PM ET (post-market)"
echo "- Captures consistent daily account values"
echo "- Logs to /tmp/market_close_capture.log"

# Alternative: Use launchd on macOS
PLIST_FILE="$HOME/Library/LaunchAgents/com.trading.marketclose.plist"

cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.trading.marketclose</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/ethansung/quant/ibkr/account_summary.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/ethansung/quant/ibkr</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>17</integer>
        <key>Minute</key>
        <integer>0</integer>
        <key>Weekday</key>
        <integer>1</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/tmp/market_close_capture.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/market_close_capture_error.log</string>
</dict>
</plist>
EOF

# Load the plist
launchctl load "$PLIST_FILE"

echo "LaunchAgent created and loaded:"
echo "- Runs daily at 5:00 PM ET on weekdays"
echo "- Provides consistent daily post-market captures"
echo "- Required for accurate Sharpe ratio calculations"

echo ""
echo "To check if it's working:"
echo "  tail -f /tmp/market_close_capture.log" 