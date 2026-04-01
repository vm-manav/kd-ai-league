import { SourceCitation } from "../types.js";

function nowIso(): string {
  return new Date().toISOString();
}

export async function getMockQuote(ticker: string): Promise<Record<string, unknown>> {
  return {
    ticker: ticker.toUpperCase(),
    ltp: 2456.25,
    changePercent: -1.3,
    volume: 4823001,
    marketCapCr: 1650210,
    pe: 24.9,
    range52w: { low: 2220.1, high: 3010.75 },
    citations: [
      { source: "NSE India", detail: `${ticker.toUpperCase()} latest quote`, timestamp: nowIso() },
    ] satisfies SourceCitation[],
  };
}

export async function getMockPriceHistory(
  ticker: string,
  from: string,
  to: string,
): Promise<Record<string, unknown>> {
  return {
    ticker: ticker.toUpperCase(),
    from,
    to,
    candles: [
      { date: from, open: 2410, high: 2440, low: 2395, close: 2421, volume: 3200000 },
      { date: to, open: 2422, high: 2461, low: 2408, close: 2456.25, volume: 4823001 },
    ],
    citations: [
      {
        source: "NSE India",
        detail: `${ticker.toUpperCase()} historical OHLCV`,
        timestamp: nowIso(),
      },
    ] satisfies SourceCitation[],
  };
}

export async function getMockRatios(ticker: string): Promise<Record<string, unknown>> {
  return {
    ticker: ticker.toUpperCase(),
    pe: 24.9,
    pb: 3.1,
    roe: 18.2,
    roce: 16.4,
    debtToEquity: 0.34,
    dividendYield: 0.48,
    citations: [
      { source: "yfinance", detail: `${ticker.toUpperCase()} ratios`, timestamp: nowIso() },
    ] satisfies SourceCitation[],
  };
}

export async function getMockNews(ticker: string, page = 1, pageSize = 5): Promise<Record<string, unknown>> {
  return {
    ticker: ticker.toUpperCase(),
    page,
    pageSize,
    total: 17,
    items: [
      {
        headline: `${ticker.toUpperCase()} expands digital services portfolio`,
        source: "NewsAPI",
        publishedAt: nowIso(),
        sentiment: "positive",
      },
      {
        headline: `Brokerages trim short-term target for ${ticker.toUpperCase()}`,
        source: "Finnhub",
        publishedAt: nowIso(),
        sentiment: "negative",
      },
    ],
    citations: [
      { source: "NewsAPI", detail: `Company news for ${ticker.toUpperCase()}` },
      { source: "Finnhub", detail: `Company coverage for ${ticker.toUpperCase()}` },
    ] satisfies SourceCitation[],
  };
}

export async function getMockMacroSnapshot(): Promise<Record<string, unknown>> {
  return {
    repoRate: 6.5,
    reverseRepo: 3.35,
    crr: 4.5,
    slr: 18.0,
    cpiInflation: 5.1,
    usdInr: 83.24,
    citations: [
      { source: "RBI DBIE", detail: "Policy rates and inflation snapshot", timestamp: nowIso() },
    ] satisfies SourceCitation[],
  };
}

export async function getMockMutualFundSearch(query: string): Promise<Record<string, unknown>> {
  return {
    query,
    schemes: [
      { schemeCode: "120503", name: "SBI Bluechip Fund - Direct Growth", amc: "SBI Mutual Fund" },
      { schemeCode: "120716", name: "ICICI Prudential Bluechip Fund - Direct Growth", amc: "ICICI Prudential MF" },
    ],
    citations: [{ source: "MFapi.in", detail: `Scheme search for ${query}` }] satisfies SourceCitation[],
  };
}

export async function getMockCrossReferenceSignals(ticker: string): Promise<Record<string, unknown>> {
  return {
    ticker: ticker.toUpperCase(),
    findings: [
      {
        claim: "Recent weakness appears sentiment-led, not fundamentals-led",
        status: "confirmed",
        confirmations: [
          {
            source: "NSE India",
            detail: `${ticker.toUpperCase()} price moved -3.9% in 5 sessions`,
          },
          {
            source: "yfinance",
            detail: "Latest quarterly revenue growth remains positive at 8.1% YoY",
          },
        ],
        contradictions: [
          {
            source: "NewsAPI",
            detail: "Headline sentiment turned negative on global IT spending caution",
          },
        ],
      },
    ],
    citations: [
      { source: "NSE India", detail: "Price action data" },
      { source: "yfinance", detail: "Fundamental trend data" },
      { source: "NewsAPI", detail: "Sentiment and narrative data" },
    ] satisfies SourceCitation[],
    disclaimer:
      "Data provided for informational and educational purposes only. Not investment advice.",
  };
}

export async function getMockIndexData(index = "NIFTY50"): Promise<Record<string, unknown>> {
  const upper = index.toUpperCase();
  return {
    index: upper,
    value: upper.includes("SENSEX") ? 73902.6 : 22430.2,
    changePercent: 0.42,
    components: ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK"],
    citations: [{ source: "NSE India", detail: `${upper} snapshot`, timestamp: nowIso() }],
  };
}

export async function getMockTopMovers(): Promise<Record<string, unknown>> {
  return {
    exchange: "NSE",
    gainers: [
      { ticker: "ULTRACEMCO", changePercent: 4.3 },
      { ticker: "LT", changePercent: 3.8 },
    ],
    losers: [
      { ticker: "WIPRO", changePercent: -3.2 },
      { ticker: "TECHM", changePercent: -2.9 },
    ],
    citations: [{ source: "NSE India", detail: "Top gainers/losers", timestamp: nowIso() }],
  };
}

export async function getMockTechnicals(ticker: string): Promise<Record<string, unknown>> {
  return {
    ticker: ticker.toUpperCase(),
    sma20: 2450.21,
    ema20: 2448.12,
    rsi14: 46.7,
    macd: { value: -3.2, signal: -1.9, histogram: -1.3 },
    bollinger: { upper: 2522.6, middle: 2450.2, lower: 2377.8 },
    citations: [{ source: "Alpha Vantage", detail: `${ticker.toUpperCase()} technical snapshot`, timestamp: nowIso() }],
  };
}

export async function getMockFinancialStatements(ticker: string): Promise<Record<string, unknown>> {
  return {
    ticker: ticker.toUpperCase(),
    incomeStatement: { revenueCr: 104532, netIncomeCr: 15420, ebitdaCr: 22301 },
    balanceSheet: { totalAssetsCr: 342110, totalDebtCr: 61340, totalEquityCr: 145890 },
    cashFlow: { operatingCashFlowCr: 21230, investingCashFlowCr: -9344, financingCashFlowCr: -4720 },
    citations: [{ source: "yfinance", detail: `${ticker.toUpperCase()} financial statements`, timestamp: nowIso() }],
  };
}

export async function getMockShareholdingPattern(ticker: string): Promise<Record<string, unknown>> {
  return {
    ticker: ticker.toUpperCase(),
    latest: { promoter: 49.1, fii: 24.4, dii: 18.8, retail: 7.7 },
    trend: [
      { quarter: "Q1", promoter: 49.0, fii: 23.9, dii: 19.2, retail: 7.9 },
      { quarter: "Q2", promoter: 49.1, fii: 24.4, dii: 18.8, retail: 7.7 },
    ],
    citations: [{ source: "BSE/NSE filings", detail: `${ticker.toUpperCase()} shareholding pattern`, timestamp: nowIso() }],
  };
}

export async function getMockQuarterlyResults(ticker: string): Promise<Record<string, unknown>> {
  return {
    ticker: ticker.toUpperCase(),
    latestQuarter: "Q3 FY25",
    revenueCr: 41764,
    netProfitCr: 8621,
    yoyRevenueGrowthPercent: 8.2,
    qoqRevenueGrowthPercent: 2.1,
    citations: [{ source: "yfinance", detail: `${ticker.toUpperCase()} quarterly results`, timestamp: nowIso() }],
  };
}

export async function getMockFundNav(schemeCode: string): Promise<Record<string, unknown>> {
  return {
    schemeCode,
    fundName: "Sample Bluechip Fund - Direct Growth",
    latestNav: 78.22,
    navDate: nowIso().slice(0, 10),
    history: [
      { date: "2025-01-01", nav: 70.1 },
      { date: "2026-01-01", nav: 78.22 },
    ],
    citations: [{ source: "MFapi.in", detail: `Scheme NAV for ${schemeCode}`, timestamp: nowIso() }],
  };
}

export async function getMockCompareFunds(schemeCodes: string[]): Promise<Record<string, unknown>> {
  return {
    schemeCodes,
    comparison: schemeCodes.map((schemeCode, idx) => ({
      schemeCode,
      latestNav: 70 + idx * 3.4,
      oneYearReturnPercent: 8 + idx * 2.1,
      volatilityScore: 12 + idx,
    })),
    citations: [{ source: "MFapi.in", detail: `Fund comparison for ${schemeCodes.join(", ")}`, timestamp: nowIso() }],
  };
}

export async function getMockNewsSentiment(ticker: string): Promise<Record<string, unknown>> {
  return {
    ticker: ticker.toUpperCase(),
    windowDays: 30,
    sentimentScore: -0.18,
    label: "slightly_negative",
    articleCount: 42,
    citations: [
      { source: "NewsAPI", detail: `${ticker.toUpperCase()} sentiment aggregation`, timestamp: nowIso() },
      { source: "Finnhub", detail: `${ticker.toUpperCase()} company news sentiment`, timestamp: nowIso() },
    ],
  };
}

export async function getMockMarketNews(page = 1, pageSize = 10): Promise<Record<string, unknown>> {
  return {
    page,
    pageSize,
    total: 50,
    items: [
      { headline: "Rally in banking stocks lifts Nifty", source: "NewsAPI", publishedAt: nowIso() },
      { headline: "Rupee weakens as crude rises", source: "GNews", publishedAt: nowIso() },
    ],
    citations: [{ source: "NewsAPI/GNews", detail: "Indian market news feed", timestamp: nowIso() }],
  };
}

export async function getMockInflationData(): Promise<Record<string, unknown>> {
  return {
    country: "India",
    cpiSeries: [
      { period: "2023", value: 5.65 },
      { period: "2024", value: 5.1 },
      { period: "2025", value: 4.8 },
    ],
    citations: [{ source: "World Bank", detail: "India CPI inflation series", timestamp: nowIso() }],
  };
}

export async function getMockCorporateFilings(ticker: string): Promise<Record<string, unknown>> {
  return {
    ticker: ticker.toUpperCase(),
    filings: [
      {
        filingId: `BSE-${ticker.toUpperCase()}-001`,
        title: "Outcome of Board Meeting - Quarterly Results",
        filingDate: nowIso().slice(0, 10),
        sourceUrl: "https://www.bseindia.com/",
      },
    ],
    citations: [{ source: "BSE India", detail: `${ticker.toUpperCase()} corporate filings`, timestamp: nowIso() }],
  };
}
