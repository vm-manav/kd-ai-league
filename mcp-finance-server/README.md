# mcp-finance-server (PS1)

Production-style MCP server scaffold for **AI League #3 PS1: Financial Research Copilot**.

This implementation focuses on:
- MCP primitives (tools/resources/prompts) over Streamable HTTP
- OAuth resource-server behavior and protected resource metadata
- Tier-aware capability discovery (Free, Premium, Analyst)
- Scope checks, rate limits, cache, and audit logging
- Cross-source reasoning tools with explicit source citations

## What is implemented now

### Core Infrastructure
- Streamable HTTP MCP endpoint at `/mcp`
- Protected resource metadata at `/.well-known/oauth-protected-resource`
- Token validation path (JWT/JWKS) with audience + issuer checks
- Tier-based request rate limits:
  - Free: 30 calls/hour
  - Premium: 150 calls/hour
  - Analyst: 500 calls/hour
- In-memory TTL cache with stale fallback behavior
- Audit log for each tool invocation

### PS1 Tools (implemented)
- `get_stock_quote` (Free, `market:read`)
- `get_price_history` (Free, `market:read`, paginated shape)
- `get_index_data` (Free, `market:read`)
- `get_top_gainers_losers` (Free, `market:read`)
- `get_company_news` (Free, `news:read`, paginated)
- `get_market_news` (Free, `news:read`)
- `get_technical_indicators` (Premium, `technicals:read`)
- `get_financial_statements` (Premium, `fundamentals:read`)
- `get_key_ratios` (Premium, `fundamentals:read`)
- `get_shareholding_pattern` (Premium, `fundamentals:read`)
- `get_quarterly_results` (Premium, `fundamentals:read`)
- `search_mutual_funds` (Premium, `mf:read`)
- `get_fund_nav` (Premium, `mf:read`)
- `compare_funds` (Premium, `mf:read`)
- `get_news_sentiment` (Premium, `news:read`)
- `get_rbi_rates` (Premium, `macro:read`)
- `get_inflation_data` (Analyst, `macro:historical`)
- `get_corporate_filings` (Analyst, `filings:read`)
- `cross_reference_signals` (Analyst, `research:generate`)
- `generate_research_brief` (Analyst, `research:generate`)
- `compare_companies` (Analyst, `research:generate`)

### MCP Resources
- `market://overview`
- `macro://snapshot`
- `watchlist://{user_id}/stocks`
- `research://{ticker}/latest`

### MCP Prompts
- `quick_analysis` (all tiers)
- `deep_dive` (Premium+)
- `sector_scan` (Premium+)
- `morning_brief` (Analyst)

---

## Local Setup

### 1) Install
```bash
npm install
cp .env.example .env
```

### 2) Run development server
```bash
npm run dev
```

Health check:
```bash
curl http://localhost:8080/health
```

Protected resource metadata:
```bash
curl http://localhost:8080/.well-known/oauth-protected-resource
```

---

## Docker Compose

This repo includes a baseline one-command composition for:
- MCP server
- Redis
- Keycloak

```bash
cp .env.example .env
docker compose up --build
```

---

## Auth Notes

For fast local hacking, `.env.example` sets:
- `AUTH_BYPASS=true`

For judged demo, set:
- `AUTH_BYPASS=false`
- `OAUTH_ISSUER`
- `OAUTH_JWKS_URI`
- `OAUTH_AUDIENCE`
- `OAUTH_TIER_CLAIM`

And use real OAuth login with PKCE from your MCP client setup.

### Auth0 production tier setup

For Auth0, create an API with identifier `mcp-finance-server`, enable RBAC, and enable
"Add Permissions in the Access Token". Assign API permissions to roles such as:

- `finance-free`: `market:read`, `news:read`
- `finance-premium`: free permissions plus `fundamentals:read`, `technicals:read`, `mf:read`, `macro:read`
- `finance-analyst`: premium permissions plus `macro:historical`, `filings:read`, `filings:deep`, `research:generate`, `watchlist:read`, `watchlist:write`

Add an Auth0 Post Login Action that writes the user's tier to a namespaced access-token claim:

```js
exports.onExecutePostLogin = async (event, api) => {
  const namespace = "https://kd-ai-league.example.com";
  const roles = event.authorization?.roles || [];

  let tier = "free";
  if (roles.includes("finance-analyst")) tier = "analyst";
  else if (roles.includes("finance-premium")) tier = "premium";

  api.accessToken.setCustomClaim(`${namespace}/tier`, tier);
  api.accessToken.setCustomClaim(`${namespace}/roles`, roles);
};
```

Set `OAUTH_TIER_CLAIM` and `OAUTH_ROLES_CLAIM` to match those namespaced claims. The server reads scopes from `scope`, `scp`, or Auth0 RBAC `permissions`.

---

## Scope / Tier behavior

Discovery is tier-aware: the server only registers tool/prompt capabilities allowed for the authenticated tier session.

Runtime checks enforce both:
- Minimum tier requirement
- Scope requirement

On insufficient scope, tools return `insufficient_scope` errors.  
On unauthenticated requests, server returns `401` plus `WWW-Authenticate` metadata pointer.  
On limit breaches, server returns `429` with `Retry-After`.

---

## Data Providers

Provider layer:
- `src/providers/liveData.ts` -> real clients + graceful fallback
- `src/providers/mockData.ts` -> fallback payloads for resilience

Active integrations in current build:
- Yahoo Finance endpoints (quotes/history/indexes/ratios/statements/earnings/holders)
- MFapi.in (scheme search + NAV history)
- Finnhub (company news, when `FINNHUB_API_KEY` is set)
- NewsAPI (company + market news, when `NEWSAPI_API_KEY` is set)
- Alpha Vantage (technicals, when `ALPHA_VANTAGE_API_KEY` is set)
- World Bank indicator API (inflation time-series)

---

## Next recommended upgrades (before final submission)

1. Set provider keys in `.env` and verify live responses end-to-end.
2. Implement full PKCE login flow in your MCP client walkthrough.
3. Persist watchlist/research/audit/rate-limit counters to Redis/Postgres.
4. Add a production-grade BSE/NSE filings adapter with robust pagination.
5. Add architecture diagram and full API docs table for judging packet.
