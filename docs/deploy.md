# Deployment

FlareSense has two Compose environments, same layout as heliotime-v2:

| | Dev | Prod |
|---|---|---|
| Command | `./du.sh dev deploy` | `./du.sh prod deploy` |
| Compose file | `docker/compose.dev.yml` | `docker/compose.prod.yml` |
| Image directory | `./data/burst_images` | `FLARESENSE_HOST_IMAGE_DIR` (required) |
| Restart | only while you leave it up | `unless-stopped` |
| Site | http://localhost:7860/ | reverse proxy to `WEB_PORT` |

`./du.sh` is a thin wrapper around `docker compose`. It loads `docker/compose.yml` and `docker/compose.<env>.yml`, and names the project `flaresense-dev` or `flaresense-prod`.

The container only serves the website. It does not run the model. `pred_live.py` keeps writing PNGs on the host (today: `flaresense.service`). Point `FLARESENSE_HOST_IMAGE_DIR` at that directory. Production mounts it read-only.

## Production

```sh
FLARESENSE_HOST_IMAGE_DIR=/srv/flaresense-data/burst_live_images
WEB_PORT=7860
```

Put that in `.env` in the repository root (Compose reads it), then:

```sh
./du.sh prod deploy
```

The path above is the one the app used before image paths moved into `paths.py` (March 2026). On the server, use the directory `paths.py` calls `SERVER_BURST_IMAGES_PATH`.

Until Compose is in front of nginx, the list fix is the systemd app: pull this tree and `sudo systemctl restart flaresense_app.service`. With no environment variables set, the process still reads `paths.py`. The predictor service does not need a restart for the list.

Only port 80/443 should be reachable from the internet. Proxy `WEB_PORT` on localhost. Example nginx:

```nginx
server {
   listen 443 ssl;
   server_name flaresense.ch;

   ssl_certificate /etc/letsencrypt/live/flaresense.ch/fullchain.pem;
   ssl_certificate_key /etc/letsencrypt/live/flaresense.ch/privkey.pem;
   include /etc/letsencrypt/options-ssl-nginx.conf;
   ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

   location / {
      proxy_pass http://127.0.0.1:7860;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto $scheme;
      proxy_http_version 1.1;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
      proxy_read_timeout 300s;
   }
}
```

Gradio uses a websocket, so the upgrade headers stay.

A second host, for example `dev.flaresense.ch`, is the same stack with `./du.sh dev deploy` and a different `WEB_PORT` plus its own `FLARESENSE_HOST_IMAGE_DIR`. Dev and prod are separate Compose projects, so both can run on one machine if the ports differ.

## What the website reads

Images are files named:

```text
{confidence}_{station}_{dd-mm-YYYY}_{HH-MM-SS}.png
```

under `{image_dir}/{year}/{month}/{day}/{station}/`. Confidence in the name is a percent (`86.50` means 86.5%). The Latest tab shows the last 3 days at 50% or above. One station is enough to list a time.
