# HTTPS deployment for IMS

## Topology

Internet / corporate network -> TCP 443 -> nginx -> `least_conn` -> HTTP `api-1:8080` / `api-2:8080` on the private Docker network.
TCP 80 is used only for HTTP-to-HTTPS redirect.
FastAPI diagnostics are loopback-only on `127.0.0.1:8081` and `127.0.0.1:8082`; neither API port is externally reachable.

## TLS files

Recommended host directory:

```text
/etc/ims/tls/
  fullchain.pem
  dmrc.key
```

`fullchain.pem` must contain the leaf/server certificate first and the intermediate CA certificate second. Do not append the root CA certificate.

Recommended permissions:

```bash
sudo chown -R root:root /etc/ims/tls
sudo chmod 700 /etc/ims/tls
sudo chmod 600 /etc/ims/tls/dmrc.key
sudo chmod 644 /etc/ims/tls/fullchain.pem
```

## Application settings

When the public site is HTTPS-only:

```env
APP_ENV=production
SESSION_COOKIE_SECURE=true
TLS_CERT_DIR=/etc/ims/tls
```

`SESSION_COOKIE_SECURE=true` is required so the IMS authentication cookie is never transmitted over plain HTTP.

## Networking

DNS must resolve the selected certificate-covered hostname to the VM's stable public IP. Allow inbound TCP 443. TCP 80 is optional but recommended for redirect. Do not expose 8081/8082, PostgreSQL, Valkey or inference ports publicly.

## Long-running query streams

`/api/v1/query/stream` has nginx buffering disabled and a one-hour upstream timeout so Research progress events continue to reach the browser promptly.
