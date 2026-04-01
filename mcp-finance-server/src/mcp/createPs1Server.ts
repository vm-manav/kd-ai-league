import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { hasMinimumTier } from "../auth/tiers.js";
import { appendAudit } from "../core/auditLog.js";
import { withTtlCache } from "../core/cache.js";
import { AuthContext, Scope, Tier } from "../types.js";
import {
  compareFunds,
  crossReferenceSignals,
  getCompanyNews,
  getCorporateFilings,
  getFinancialStatements,
  getFundNav,
  getIndexData,
  getInflationData,
  getKeyRatios,
  getMarketNews,
  getNewsSentiment,
  getPriceHistory,
  getQuarterlyResults,
  getRbiRates,
  getShareholdingPattern,
  getStockQuote,
  getTechnicalIndicators,
  getTopGainersLosers,
  searchMutualFunds,
} from "../providers/liveData.js";
import { getResearch, getWatchlist, saveResearch } from "../userData/store.js";

type ToolHandlerArgs = Record<string, unknown>;

function toolError(message: string, code: string): { isError: true; content: { type: "text"; text: string }[] } {
  return {
    isError: true,
    content: [{ type: "text", text: JSON.stringify({ error: code, message }) }],
  };
}

function hasScopes(ctx: AuthContext, requiredScopes: Scope[]): boolean {
  return requiredScopes.every((scope) => ctx.scopes.has(scope));
}

function registerTieredTool(input: {
  server: McpServer;
  auth: AuthContext;
  name: string;
  description: string;
  minTier: Tier;
  requiredScopes: Scope[];
  inputSchema: z.ZodTypeAny;
  cacheTtlSec?: number;
  handler: (args: ToolHandlerArgs) => Promise<Record<string, unknown>>;
}): void {
  const { server, auth, name, description, minTier, requiredScopes, inputSchema, cacheTtlSec, handler } = input;

  if (!hasMinimumTier(auth.tier, minTier)) {
    return;
  }

  server.registerTool(
    name,
    {
      description,
      inputSchema,
    },
    async (args) => {
      if (!hasScopes(auth, requiredScopes)) {
        appendAudit({
          userId: auth.userId,
          tier: auth.tier,
          operation: name,
          status: "forbidden",
          timestamp: new Date().toISOString(),
          details: { requiredScopes },
        });
        return toolError("Insufficient scope", "insufficient_scope");
      }

      try {
        const data =
          cacheTtlSec && cacheTtlSec > 0
            ? (
                await withTtlCache(
                  `${name}:${JSON.stringify(args)}`,
                  cacheTtlSec,
                  async () => handler(args as ToolHandlerArgs),
                )
              ).data
            : await handler(args as ToolHandlerArgs);

        appendAudit({
          userId: auth.userId,
          tier: auth.tier,
          operation: name,
          status: "success",
          timestamp: new Date().toISOString(),
        });

        return {
          content: [{ type: "text", text: JSON.stringify(data) }],
        };
      } catch (error) {
        appendAudit({
          userId: auth.userId,
          tier: auth.tier,
          operation: name,
          status: "error",
          timestamp: new Date().toISOString(),
          details: { message: error instanceof Error ? error.message : "unknown" },
        });
        return toolError("Upstream data unavailable", "upstream_unavailable");
      }
    },
  );
}

export function createPs1Server(auth: AuthContext): McpServer {
  const server = new McpServer({
    name: "mcp-finance-server",
    version: "0.1.0",
  });

  registerTieredTool({
    server,
    auth,
    name: "get_stock_quote",
    description: "Latest NSE/BSE-style quote data for a ticker",
    minTier: "free",
    requiredScopes: ["market:read"],
    cacheTtlSec: 60,
    inputSchema: z.object({ ticker: z.string().min(1) }),
    handler: async ({ ticker }) => getStockQuote(String(ticker)),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_price_history",
    description: "Historical OHLCV with pagination-friendly response",
    minTier: "free",
    requiredScopes: ["market:read"],
    cacheTtlSec: 120,
    inputSchema: z.object({
      ticker: z.string().min(1),
      from: z.string(),
      to: z.string(),
      page: z.number().int().min(1).default(1),
      pageSize: z.number().int().min(1).max(100).default(30),
    }),
    handler: async ({ ticker, from, to, page, pageSize }) =>
      getPriceHistory(String(ticker), String(from), String(to), Number(page), Number(pageSize)),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_index_data",
    description: "Index values and sample composition for Nifty/Sensex/BankNifty",
    minTier: "free",
    requiredScopes: ["market:read"],
    cacheTtlSec: 60,
    inputSchema: z.object({ index: z.string().default("NIFTY50") }),
    handler: async ({ index }) => getIndexData(String(index)),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_top_gainers_losers",
    description: "Top movers snapshot",
    minTier: "free",
    requiredScopes: ["market:read"],
    cacheTtlSec: 120,
    inputSchema: z.object({ exchange: z.string().default("NSE") }),
    handler: async ({ exchange }) => getTopGainersLosers(String(exchange)),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_technical_indicators",
    description: "SMA/EMA/RSI/MACD/Bollinger for a ticker",
    minTier: "premium",
    requiredScopes: ["technicals:read"],
    cacheTtlSec: 15 * 60,
    inputSchema: z.object({ ticker: z.string().min(1) }),
    handler: async ({ ticker }) => getTechnicalIndicators(String(ticker)),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_company_news",
    description: "Latest company news with sentiment labels",
    minTier: "free",
    requiredScopes: ["news:read"],
    cacheTtlSec: 1800,
    inputSchema: z.object({
      ticker: z.string().min(1),
      page: z.number().int().min(1).default(1),
      pageSize: z.number().int().min(1).max(20).default(5),
    }),
    handler: async ({ ticker, page, pageSize }) =>
      getCompanyNews(String(ticker), Number(page), Number(pageSize)),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_market_news",
    description: "Broad Indian market news feed with pagination",
    minTier: "free",
    requiredScopes: ["news:read"],
    cacheTtlSec: 30 * 60,
    inputSchema: z.object({
      page: z.number().int().min(1).default(1),
      pageSize: z.number().int().min(1).max(20).default(10),
    }),
    handler: async ({ page, pageSize }) => getMarketNews(Number(page), Number(pageSize)),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_news_sentiment",
    description: "Aggregate company sentiment over a time window",
    minTier: "premium",
    requiredScopes: ["news:read"],
    cacheTtlSec: 30 * 60,
    inputSchema: z.object({
      ticker: z.string().min(1),
      windowDays: z.number().int().min(3).max(90).default(30),
    }),
    handler: async ({ ticker, windowDays }) => getNewsSentiment(String(ticker), Number(windowDays)),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_financial_statements",
    description: "Income statement, balance sheet, and cash flow data",
    minTier: "premium",
    requiredScopes: ["fundamentals:read"],
    cacheTtlSec: 24 * 60 * 60,
    inputSchema: z.object({ ticker: z.string().min(1) }),
    handler: async ({ ticker }) => getFinancialStatements(String(ticker)),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_key_ratios",
    description: "Key valuation and profitability ratios",
    minTier: "premium",
    requiredScopes: ["fundamentals:read"],
    cacheTtlSec: 24 * 60 * 60,
    inputSchema: z.object({ ticker: z.string().min(1) }),
    handler: async ({ ticker }) => getKeyRatios(String(ticker)),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_shareholding_pattern",
    description: "Shareholding/holder composition trend for a company",
    minTier: "premium",
    requiredScopes: ["fundamentals:read"],
    cacheTtlSec: 24 * 60 * 60,
    inputSchema: z.object({ ticker: z.string().min(1) }),
    handler: async ({ ticker }) => getShareholdingPattern(String(ticker)),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_quarterly_results",
    description: "Latest quarterly results with YoY/QoQ context where available",
    minTier: "premium",
    requiredScopes: ["fundamentals:read"],
    cacheTtlSec: 24 * 60 * 60,
    inputSchema: z.object({ ticker: z.string().min(1) }),
    handler: async ({ ticker }) => getQuarterlyResults(String(ticker)),
  });

  registerTieredTool({
    server,
    auth,
    name: "search_mutual_funds",
    description: "Search AMFI-registered schemes",
    minTier: "premium",
    requiredScopes: ["mf:read"],
    cacheTtlSec: 6 * 60 * 60,
    inputSchema: z.object({ query: z.string().min(2) }),
    handler: async ({ query }) => searchMutualFunds(String(query)),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_fund_nav",
    description: "Latest and historical NAV for an AMFI scheme code",
    minTier: "premium",
    requiredScopes: ["mf:read"],
    cacheTtlSec: 6 * 60 * 60,
    inputSchema: z.object({ schemeCode: z.string().min(3) }),
    handler: async ({ schemeCode }) => getFundNav(String(schemeCode)),
  });

  registerTieredTool({
    server,
    auth,
    name: "compare_funds",
    description: "Compare 2-5 mutual funds by NAV and return proxies",
    minTier: "premium",
    requiredScopes: ["mf:read"],
    cacheTtlSec: 6 * 60 * 60,
    inputSchema: z.object({ schemeCodes: z.array(z.string().min(3)).min(2).max(5) }),
    handler: async ({ schemeCodes }) => compareFunds(schemeCodes as string[]),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_rbi_rates",
    description: "RBI macro snapshot including repo and inflation context",
    minTier: "premium",
    requiredScopes: ["macro:read"],
    cacheTtlSec: 60 * 60,
    inputSchema: z.object({}),
    handler: async () => getRbiRates(),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_inflation_data",
    description: "CPI/WPI-style inflation time series for India",
    minTier: "analyst",
    requiredScopes: ["macro:historical"],
    cacheTtlSec: 24 * 60 * 60,
    inputSchema: z.object({}),
    handler: async () => getInflationData(),
  });

  registerTieredTool({
    server,
    auth,
    name: "get_corporate_filings",
    description: "Recent regulatory/corporate filing list for a company",
    minTier: "analyst",
    requiredScopes: ["filings:read"],
    cacheTtlSec: 24 * 60 * 60,
    inputSchema: z.object({
      ticker: z.string().min(1),
      page: z.number().int().min(1).default(1),
      pageSize: z.number().int().min(1).max(50).default(10),
    }),
    handler: async ({ ticker, page, pageSize }) =>
      getCorporateFilings(String(ticker), Number(page), Number(pageSize)),
  });

  registerTieredTool({
    server,
    auth,
    name: "cross_reference_signals",
    description: "Analyst-only cross-source confirmation/contradiction signals",
    minTier: "analyst",
    requiredScopes: ["research:generate"],
    cacheTtlSec: 30 * 60,
    inputSchema: z.object({ ticker: z.string().min(1) }),
    handler: async ({ ticker }) => crossReferenceSignals(String(ticker)),
  });

  registerTieredTool({
    server,
    auth,
    name: "generate_research_brief",
    description: "Analyst-only structured research synthesis with citations",
    minTier: "analyst",
    requiredScopes: ["research:generate"],
    cacheTtlSec: 60 * 60,
    inputSchema: z.object({ ticker: z.string().min(1) }),
    handler: async ({ ticker }) => {
      const symbol = String(ticker).toUpperCase();
      const [quote, ratios, news, macro, cross, statements, holdings, results] = await Promise.all([
        getStockQuote(symbol),
        getKeyRatios(symbol),
        getCompanyNews(symbol, 1, 5),
        getRbiRates(),
        crossReferenceSignals(symbol),
        getFinancialStatements(symbol),
        getShareholdingPattern(symbol),
        getQuarterlyResults(symbol),
      ]);

      const brief = {
        ticker: symbol,
        generatedAt: new Date().toISOString(),
        sections: {
          market: quote,
          fundamentals: ratios,
          statements,
          shareholding: holdings,
          results,
          sentiment: news,
          macro,
          crossSourceFindings: cross,
        },
        disclaimer:
          "Informational output only. This does not constitute financial or investment advice.",
      };
      saveResearch(symbol, brief);
      return brief;
    },
  });

  registerTieredTool({
    server,
    auth,
    name: "compare_companies",
    description: "Analyst-only comparison across 2-5 companies",
    minTier: "analyst",
    requiredScopes: ["research:generate"],
    cacheTtlSec: 30 * 60,
    inputSchema: z.object({ tickers: z.array(z.string()).min(2).max(5) }),
    handler: async ({ tickers }) => {
      const symbols = (tickers as string[]).map((t) => t.toUpperCase());
      const rows = await Promise.all(
        symbols.map(async (ticker) => ({
          ticker,
          quote: await getStockQuote(ticker),
          ratios: await getKeyRatios(ticker),
          sentiment: await getNewsSentiment(ticker, 30),
          holdings: await getShareholdingPattern(ticker),
        })),
      );
      return { comparedAt: new Date().toISOString(), rows };
    },
  });

  server.registerResource(
    "market_overview",
    "market://overview",
    {
      title: "Market Overview",
      description: "Nifty/Sensex snapshot plus top market movers.",
      mimeType: "application/json",
    },
    async () => {
      const [nifty, sensex, banknifty, movers] = await Promise.all([
        getIndexData("NIFTY50"),
        getIndexData("SENSEX"),
        getIndexData("BANKNIFTY"),
        getTopGainersLosers("NSE"),
      ]);
      return {
        contents: [
          {
            uri: "market://overview",
            mimeType: "application/json",
            text: JSON.stringify({ nifty, sensex, banknifty, movers }),
          },
        ],
      };
    },
  );

  server.registerResource(
    "macro_snapshot",
    "macro://snapshot",
    {
      title: "Macro Snapshot",
      description: "Latest RBI macro snapshot for quick context.",
      mimeType: "application/json",
    },
    async () => {
      const [rates, inflation] = await Promise.all([getRbiRates(), getInflationData()]);
      return {
        contents: [
          {
            uri: "macro://snapshot",
            mimeType: "application/json",
            text: JSON.stringify({
              rates,
              inflation: (inflation.cpiSeries as any[])?.slice(0, 3) ?? [],
            }),
          },
        ],
      };
    },
  );

  server.registerResource(
    "watchlist_by_user",
    new ResourceTemplate("watchlist://{user_id}/stocks", { list: undefined }),
    {
      title: "User Watchlist",
      description: "Auth-scoped watchlist stocks for the current user.",
      mimeType: "application/json",
    },
    async (_uri, variables) => {
      const requestedUser = String(variables.user_id);
      if (requestedUser !== auth.userId) {
        return {
          contents: [
            {
              uri: `watchlist://${requestedUser}/stocks`,
              mimeType: "application/json",
              text: JSON.stringify({
                error: "forbidden",
                message: "Cannot access another user's watchlist.",
              }),
            },
          ],
        };
      }
      return {
        contents: [
          {
            uri: `watchlist://${requestedUser}/stocks`,
            mimeType: "application/json",
            text: JSON.stringify({ userId: requestedUser, stocks: getWatchlist(requestedUser) }),
          },
        ],
      };
    },
  );

  server.registerResource(
    "research_latest",
    new ResourceTemplate("research://{ticker}/latest", { list: undefined }),
    {
      title: "Latest Research Brief",
      description: "Most recent generated research brief for the ticker.",
      mimeType: "application/json",
    },
    async (_uri, variables) => {
      const ticker = String(variables.ticker).toUpperCase();
      return {
        contents: [
          {
            uri: `research://${ticker}/latest`,
            mimeType: "application/json",
            text: JSON.stringify({
              ticker,
              brief: getResearch(ticker),
            }),
          },
        ],
      };
    },
  );

  server.registerPrompt(
    "quick_analysis",
    {
      description: "Fast overview using quote + headline news.",
      argsSchema: {
        ticker: z.string().min(1),
      },
    },
    ({ ticker }) => ({
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `Run get_stock_quote, get_key_ratios (if available), and get_company_news for ${ticker}. Summarize risk/reward with explicit source citations.`,
          },
        },
      ],
    }),
  );

  if (hasMinimumTier(auth.tier, "premium")) {
    server.registerPrompt(
      "deep_dive",
      {
        description: "Comprehensive PS1 deep-dive workflow.",
        argsSchema: {
          ticker: z.string().min(1),
        },
      },
      ({ ticker }) => ({
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `Run get_stock_quote, get_technical_indicators, get_financial_statements, get_key_ratios, get_shareholding_pattern, get_quarterly_results, get_company_news, get_news_sentiment, get_rbi_rates, and get_corporate_filings for ${ticker}. Return a fully structured research brief with citations.`,
            },
          },
        ],
      }),
    );

    server.registerPrompt(
      "sector_scan",
      {
        description: "Compare top companies in a sector on fundamentals and sentiment.",
        argsSchema: {
          tickers: z.array(z.string()).min(2).max(5),
        },
      },
      ({ tickers }) => ({
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `Run compare_companies for ${tickers.join(", ")}. Add per-company strengths/risks and source-backed conclusions.`,
            },
          },
        ],
      }),
    );
  }

  if (hasMinimumTier(auth.tier, "analyst")) {
    server.registerPrompt(
      "morning_brief",
      {
        description: "Analyst morning brief over market and watchlist.",
        argsSchema: {},
      },
      () => ({
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `Read market://overview, macro://snapshot, and watchlist://${auth.userId}/stocks. Then run cross_reference_signals for top 2 watchlist names and include get_market_news highlights.`,
            },
          },
        ],
      }),
    );
  }

  return server;
}
