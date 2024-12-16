# FlareSense-v2
python=3.11

# Website
Access logs with `sudo journalctl -u flaresense.service`


## Redeploy, when changing service
sudo systemctl restart flaresense.service
sudo systemctl daemon-reload