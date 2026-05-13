# AWS EC2 Deployment

This is the lowest-friction AWS deployment for the Finance MCP server:

- One EC2 instance
- Docker Compose
- Caddy for HTTPS
- No load balancer, NAT gateway, or managed database

## Suggested AWS shape

Use `t4g.micro` or `t4g.small` on Amazon Linux 2023. `t4g.nano` is cheaper, but 512 MB RAM is tight for Docker builds plus Caddy.

For a stable URL without buying a domain, allocate an Elastic IP and use `sslip.io`:

```text
https://<elastic-ip>.sslip.io
```

Example:

```text
https://203.0.113.10.sslip.io
```

Use that exact base URL for `SERVER_BASE_URL`, and use the trailing-slash version as the Auth0 API Identifier / `OAUTH_AUDIENCE`.

## Security group

Allow inbound:

- TCP 22 from your IP only
- TCP 80 from anywhere
- TCP 443 from anywhere

## First deploy

SSH into the instance, then:

```bash
sudo dnf update -y
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
newgrp docker
sudo curl -SL https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-aarch64 \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

Clone the repo:

```bash
git clone <your-repo-url>
cd kd-ai-league/mcp-finance-server
```

Create `.env.production` on the instance. Do not commit it.

```env
PORT=8080
MCP_PATH=/mcp
SERVER_BASE_URL=https://<elastic-ip>.sslip.io

AUTH_BYPASS=false

OAUTH_ISSUER=https://mcp-finance-server-ai-league.us.auth0.com/
OAUTH_JWKS_URI=https://mcp-finance-server-ai-league.us.auth0.com/.well-known/jwks.json
OAUTH_AUDIENCE=https://<elastic-ip>.sslip.io/
OAUTH_AUTHORIZATION_SERVER=https://mcp-finance-server-ai-league.us.auth0.com/
OAUTH_TIER_CLAIM=https://kd-ai-league.example.com/tier
OAUTH_ROLES_CLAIM=https://kd-ai-league.example.com/roles

ALPHA_VANTAGE_API_KEY=
FINNHUB_API_KEY=
NEWSAPI_API_KEY=
GNEWS_API_KEY=
```

Create `.env` for Compose variable substitution:

```env
DOMAIN=<elastic-ip>.sslip.io
```

Start:

```bash
docker-compose -f docker-compose.aws.yml up -d --build
```

Verify:

```bash
curl https://<elastic-ip>.sslip.io/health
curl https://<elastic-ip>.sslip.io/.well-known/oauth-protected-resource
```

## Auth0

Create/update the Auth0 API Identifier to:

```text
https://<elastic-ip>.sslip.io/
```

Then update role permissions on that API and use the same Regular Web Application client ID/secret in Claude.

Claude MCP URL:

```text
https://<elastic-ip>.sslip.io/mcp
```
