import { randomUUID } from "node:crypto";
import { createMcpExpressApp } from "@modelcontextprotocol/sdk/server/express.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
import { config } from "./config.js";
import { protectedResourceMetadata, sendUnauthorized } from "./auth/httpAuth.js";
import { validateBearerToken } from "./auth/tokenValidator.js";
import { checkAndConsumeRateLimit } from "./core/rateLimit.js";
import { createPs1Server } from "./mcp/createPs1Server.js";

interface SessionState {
  transport: StreamableHTTPServerTransport;
  userId: string;
}

const sessions: Record<string, SessionState> = {};
const app = createMcpExpressApp({ host: "0.0.0.0" });

app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "authorization,content-type,accept,mcp-session-id,mcp-protocol-version",
  );
  res.setHeader(
    "Access-Control-Expose-Headers",
    "mcp-session-id,x-ratelimit-limit,x-ratelimit-remaining,retry-after,www-authenticate",
  );

  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }

  next();
});

app.get("/health", (_req, res) => {
  res.status(200).json({
    status: "ok",
    name: "mcp-finance-server",
    mode: config.authBypass ? "auth-bypass" : "oauth",
    timestamp: new Date().toISOString(),
  });
});

app.get("/.well-known/oauth-protected-resource", (_req, res) => {
  res.status(200).json(protectedResourceMetadata());
});

app.post(config.mcpPath, async (req, res) => {
  try {
    const auth = await validateBearerToken(req.header("authorization"));
    if (!auth) {
      sendUnauthorized(res);
      return;
    }

    const rate = checkAndConsumeRateLimit({ userId: auth.userId, tier: auth.tier });
    res.setHeader("X-RateLimit-Limit", String(rate.limit));
    res.setHeader("X-RateLimit-Remaining", String(rate.remaining));
    if (!rate.ok) {
      res.setHeader("Retry-After", String(rate.retryAfterSec ?? 60));
      res.status(429).json({
        error: "rate_limit_exceeded",
        message: "Tier rate limit exceeded.",
      });
      return;
    }

    const sessionId = req.header("mcp-session-id");
    let transport: StreamableHTTPServerTransport | undefined;

    if (sessionId && sessions[sessionId]) {
      if (sessions[sessionId].userId !== auth.userId) {
        sendUnauthorized(res);
        return;
      }
      transport = sessions[sessionId].transport;
      await transport.handleRequest(req, res, req.body);
      return;
    }

    if (!sessionId && isInitializeRequest(req.body)) {
      const server = createPs1Server(auth);
      transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: (newSessionId) => {
          sessions[newSessionId] = {
            transport: transport!,
            userId: auth.userId,
          };
        },
      });

      await server.connect(transport);
      await transport.handleRequest(req, res, req.body);
      return;
    }

    res.status(400).json({
      jsonrpc: "2.0",
      error: { code: -32000, message: "Bad Request: invalid session lifecycle." },
      id: null,
    });
  } catch (error) {
    if (!res.headersSent) {
      res.status(401).json({
        error: "invalid_token",
        message: error instanceof Error ? error.message : "Token validation failed",
      });
    }
  }
});

app.get(config.mcpPath, async (req, res) => {
  const sessionId = req.header("mcp-session-id");
  if (!sessionId || !sessions[sessionId]) {
    res.status(400).send("Invalid or missing session ID");
    return;
  }
  await sessions[sessionId].transport.handleRequest(req, res);
});

app.listen(config.port, (error?: Error) => {
  if (error) {
    console.error("Failed to start mcp-finance-server:", error);
    process.exit(1);
  }
  console.log(`mcp-finance-server listening on :${config.port}`);
});
