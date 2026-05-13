import dotenv from "dotenv";

dotenv.config();

export const config = {
  port: Number(process.env.PORT ?? 8080),
  mcpPath: process.env.MCP_PATH ?? "/mcp",
  serverBaseUrl: process.env.SERVER_BASE_URL ?? "http://localhost:8080",
  providers: {
    alphaVantageApiKey: process.env.ALPHA_VANTAGE_API_KEY ?? "",
    finnhubApiKey: process.env.FINNHUB_API_KEY ?? "",
    newsApiKey: process.env.NEWSAPI_API_KEY ?? "",
    gnewsApiKey: process.env.GNEWS_API_KEY ?? "",
  },
  oauth: {
    issuer: process.env.OAUTH_ISSUER ?? "",
    jwksUri: process.env.OAUTH_JWKS_URI ?? "",
    audience: process.env.OAUTH_AUDIENCE ?? "mcp-finance-server",
    tierClaim: process.env.OAUTH_TIER_CLAIM ?? "https://kd-ai-league.example.com/tier",
    rolesClaim: process.env.OAUTH_ROLES_CLAIM ?? "https://kd-ai-league.example.com/roles",
    authorizationServer:
      process.env.OAUTH_AUTHORIZATION_SERVER ??
      "http://localhost:8081/realms/mcp-finance",
  },
  authBypass: process.env.AUTH_BYPASS === "true",
};
