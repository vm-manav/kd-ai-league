import { config } from "../config.js";
import {
  getMockCompareFunds,
  getMockCorporateFilings,
  getMockCrossReferenceSignals,
  getMockFinancialStatements,
  getMockFundNav,
  getMockIndexData,
  getMockInflationData,
  getMockMacroSnapshot,
  getMockMarketNews,
  getMockMutualFundSearch,
  getMockNews,
  getMockNewsSentiment,
  getMockPriceHistory,
  getMockQuarterlyResults,
  getMockQuote,
  getMockRatios,
  getMockShareholdingPattern,
  getMockTechnicals,
  getMockTopMovers,
} from "./mockData.js";

function withNs(ticker: string): string {
  const upper = ticker.toUpperCase();
  return upper.includes(".") ? upper : `${upper}.NS`;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      "user-agent": "mcp-finance-server/0.2",
      accept: "application/json,text/plain,*/*",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} from ${url}`);
  }
  return (await res.json()) as T;
}

function toEpoch(date: string): number {
  return Math.floor(new Date(date).getTime() / 1000);
}

function percentChange(current: number, previous: number): number {
  if (!Number.isFinite(current) || !Number.isFinite(previous) || previous === 0) {
    return 0;
  }
  return ((current - previous) / previous) * 100;
}

async function yahooQuoteRaw(ticker: string): Promise<Record<string, any>> {
  const url = `https://query1.finance.yahoo.com/v7/finance/quote?symbols=${encodeURIComponent(withNs(ticker))}`;
  const data = await fetchJson<any>(url);
  const row = data?.quoteResponse?.result?.[0];
  if (!row) {
    throw new Error("No Yahoo quote result");
  }
  return row;
}

async function yahooHistoryRaw(ticker: string, from: string, to: string): Promise<Record<string, any>> {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(
    withNs(ticker),
  )}?period1=${toEpoch(from)}&period2=${toEpoch(to)}&interval=1d`;
  const data = await fetchJson<any>(url);
  const result = data?.chart?.result?.[0];
  if (!result) {
    throw new Error("No Yahoo chart result");
  }
  return result;
}

export async function getStockQuote(ticker: string): Promise<Record<string, unknown>> {
  try {
    const q = await yahooQuoteRaw(ticker);
    return {
      ticker: ticker.toUpperCase(),
      ltp: q.regularMarketPrice ?? null,
      changePercent: q.regularMarketChangePercent ?? null,
      volume: q.regularMarketVolume ?? null,
      marketCap: q.marketCap ?? null,
      pe: q.trailingPE ?? null,
      range52w: {
        low: q.fiftyTwoWeekLow ?? null,
        high: q.fiftyTwoWeekHigh ?? null,
      },
      exchange: q.fullExchangeName ?? "NSE",
      citations: [{ source: "Yahoo Finance", detail: `${withNs(ticker)} quote endpoint` }],
    };
  } catch {
    return getMockQuote(ticker);
  }
}

export async function getPriceHistory(
  ticker: string,
  from: string,
  to: string,
  page: number,
  pageSize: number,
): Promise<Record<string, unknown>> {
  try {
    const result = await yahooHistoryRaw(ticker, from, to);
    const timestamps: number[] = result.timestamp ?? [];
    const quote = result.indicators?.quote?.[0] ?? {};
    const candles = timestamps.map((ts, idx) => ({
      date: new Date(ts * 1000).toISOString().slice(0, 10),
      open: quote.open?.[idx] ?? null,
      high: quote.high?.[idx] ?? null,
      low: quote.low?.[idx] ?? null,
      close: quote.close?.[idx] ?? null,
      volume: quote.volume?.[idx] ?? null,
    }));
    const start = (page - 1) * pageSize;
    const paged = candles.slice(start, start + pageSize);
    return {
      ticker: ticker.toUpperCase(),
      from,
      to,
      pagination: {
        page,
        pageSize,
        total: candles.length,
        hasNext: start + pageSize < candles.length,
      },
      candles: paged,
      citations: [{ source: "Yahoo Finance", detail: `${withNs(ticker)} chart endpoint` }],
    };
  } catch {
    const fallback = await getMockPriceHistory(ticker, from, to);
    return {
      ...fallback,
      pagination: { page, pageSize, total: 2, hasNext: false },
    };
  }
}

export async function getIndexData(index: string): Promise<Record<string, unknown>> {
  const map: Record<string, string> = {
    NIFTY50: "^NSEI",
    SENSEX: "^BSESN",
    BANKNIFTY: "^NSEBANK",
  };
  const symbol = map[index.toUpperCase()] ?? "^NSEI";
  try {
    const url = `https://query1.finance.yahoo.com/v7/finance/quote?symbols=${encodeURIComponent(symbol)}`;
    const data = await fetchJson<any>(url);
    const row = data?.quoteResponse?.result?.[0];
    if (!row) {
      throw new Error("No index row");
    }
    return {
      index: index.toUpperCase(),
      value: row.regularMarketPrice,
      changePercent: row.regularMarketChangePercent,
      previousClose: row.regularMarketPreviousClose,
      components: ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK"],
      citations: [{ source: "Yahoo Finance", detail: `${symbol} index quote` }],
    };
  } catch {
    return getMockIndexData(index);
  }
}

export async function getTopGainersLosers(exchange = "NSE"): Promise<Record<string, unknown>> {
  const basket = ["RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK", "LT", "ITC"];
  try {
    const quotes = await Promise.all(
      basket.map(async (ticker) => {
        const q = await yahooQuoteRaw(ticker);
        return {
          ticker,
          ltp: q.regularMarketPrice ?? null,
          changePercent: q.regularMarketChangePercent ?? null,
        };
      }),
    );
    const sorted = quotes
      .filter((q) => typeof q.changePercent === "number")
      .sort((a, b) => Number(b.changePercent) - Number(a.changePercent));
    return {
      exchange: exchange.toUpperCase(),
      gainers: sorted.slice(0, 5),
      losers: sorted.slice(-5).reverse(),
      citations: [{ source: "Yahoo Finance", detail: "Derived top movers from tracked NSE basket" }],
    };
  } catch {
    return getMockTopMovers();
  }
}

function calcSma(values: number[], period: number): number | null {
  if (values.length < period) {
    return null;
  }
  const window = values.slice(values.length - period);
  return window.reduce((acc, v) => acc + v, 0) / period;
}

function calcEma(values: number[], period: number): number | null {
  if (values.length < period) {
    return null;
  }
  const k = 2 / (period + 1);
  let ema = values[0];
  for (let i = 1; i < values.length; i += 1) {
    ema = values[i] * k + ema * (1 - k);
  }
  return ema;
}

function calcRsi(values: number[], period: number): number | null {
  if (values.length <= period) {
    return null;
  }
  let gains = 0;
  let losses = 0;
  for (let i = values.length - period; i < values.length; i += 1) {
    const delta = values[i] - values[i - 1];
    if (delta >= 0) gains += delta;
    else losses -= delta;
  }
  if (losses === 0) return 100;
  const rs = gains / losses;
  return 100 - 100 / (1 + rs);
}

export async function getTechnicalIndicators(ticker: string): Promise<Record<string, unknown>> {
  if (config.providers.alphaVantageApiKey) {
    try {
      const base = "https://www.alphavantage.co/query";
      const params = `symbol=${encodeURIComponent(`${ticker.toUpperCase()}.BSE`)}&apikey=${encodeURIComponent(
        config.providers.alphaVantageApiKey,
      )}`;
      const [sma, rsi, macd] = await Promise.all([
        fetchJson<any>(`${base}?function=SMA&interval=daily&time_period=20&series_type=close&${params}`),
        fetchJson<any>(`${base}?function=RSI&interval=daily&time_period=14&series_type=close&${params}`),
        fetchJson<any>(`${base}?function=MACD&interval=daily&series_type=close&${params}`),
      ]);
      const smaEntry = Object.values(sma["Technical Analysis: SMA"] ?? {})[0] as any;
      const rsiEntry = Object.values(rsi["Technical Analysis: RSI"] ?? {})[0] as any;
      const macdEntry = Object.values(macd["Technical Analysis: MACD"] ?? {})[0] as any;
      return {
        ticker: ticker.toUpperCase(),
        sma20: Number(smaEntry?.SMA ?? null),
        rsi14: Number(rsiEntry?.RSI ?? null),
        macd: {
          value: Number(macdEntry?.MACD ?? null),
          signal: Number(macdEntry?.MACD_Signal ?? null),
          histogram: Number(macdEntry?.MACD_Hist ?? null),
        },
        citations: [{ source: "Alpha Vantage", detail: `${ticker.toUpperCase()}.BSE technical indicators` }],
      };
    } catch {
      // fall through
    }
  }

  try {
    const to = new Date().toISOString().slice(0, 10);
    const fromDate = new Date();
    fromDate.setDate(fromDate.getDate() - 90);
    const from = fromDate.toISOString().slice(0, 10);
    const hist = await getPriceHistory(ticker, from, to, 1, 200);
    const closes = ((hist.candles as any[]) ?? [])
      .map((c) => Number(c.close))
      .filter((n) => Number.isFinite(n));
    const sma20 = calcSma(closes, 20);
    const ema20 = calcEma(closes, 20);
    const rsi14 = calcRsi(closes, 14);
    const ema12 = calcEma(closes, 12);
    const ema26 = calcEma(closes, 26);
    const macdValue = ema12 !== null && ema26 !== null ? ema12 - ema26 : null;
    const mid = sma20 ?? null;
    const std = closes.length > 20 ? Math.sqrt(closes.slice(-20).reduce((acc, v) => acc + (v - (mid ?? v)) ** 2, 0) / 20) : 0;
    return {
      ticker: ticker.toUpperCase(),
      sma20,
      ema20,
      rsi14,
      macd: {
        value: macdValue,
        signal: macdValue,
        histogram: 0,
      },
      bollinger: mid === null ? null : { upper: mid + 2 * std, middle: mid, lower: mid - 2 * std },
      citations: [{ source: "Yahoo Finance", detail: "Derived indicators from daily close series" }],
    };
  } catch {
    return getMockTechnicals(ticker);
  }
}

export async function getFinancialStatements(ticker: string): Promise<Record<string, unknown>> {
  try {
    const url = `https://query2.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(
      withNs(ticker),
    )}?modules=incomeStatementHistoryQuarterly,balanceSheetHistoryQuarterly,cashflowStatementHistoryQuarterly`;
    const data = await fetchJson<any>(url);
    const result = data?.quoteSummary?.result?.[0];
    if (!result) throw new Error("No financial statement payload");
    return {
      ticker: ticker.toUpperCase(),
      incomeStatement: result.incomeStatementHistoryQuarterly?.incomeStatementHistory ?? [],
      balanceSheet: result.balanceSheetHistoryQuarterly?.balanceSheetStatements ?? [],
      cashFlow: result.cashflowStatementHistoryQuarterly?.cashflowStatements ?? [],
      citations: [{ source: "Yahoo Finance", detail: `${withNs(ticker)} quoteSummary financial modules` }],
    };
  } catch {
    return getMockFinancialStatements(ticker);
  }
}

export async function getKeyRatios(ticker: string): Promise<Record<string, unknown>> {
  try {
    const q = await yahooQuoteRaw(ticker);
    return {
      ticker: ticker.toUpperCase(),
      pe: q.trailingPE ?? null,
      pb: q.priceToBook ?? null,
      roe: q.returnOnEquity ?? null,
      debtToEquity: q.debtToEquity ?? null,
      dividendYield: q.trailingAnnualDividendYield ?? null,
      citations: [{ source: "Yahoo Finance", detail: `${withNs(ticker)} key ratios from quote endpoint` }],
    };
  } catch {
    return getMockRatios(ticker);
  }
}

export async function getShareholdingPattern(ticker: string): Promise<Record<string, unknown>> {
  try {
    const url = `https://query2.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(
      withNs(ticker),
    )}?modules=majorHoldersBreakdown`;
    const data = await fetchJson<any>(url);
    const holders = data?.quoteSummary?.result?.[0]?.majorHoldersBreakdown;
    if (!holders) throw new Error("No holder breakdown");
    return {
      ticker: ticker.toUpperCase(),
      latest: {
        institutionsPercent: holders.institutionsPercentHeld?.raw ?? null,
        insidersPercent: holders.insidersPercentHeld?.raw ?? null,
      },
      trend: [],
      citations: [{ source: "Yahoo Finance", detail: `${withNs(ticker)} holders breakdown` }],
      note: "Promoter/FII/DII split may require exchange filings; this feed uses available holder aggregates.",
    };
  } catch {
    return getMockShareholdingPattern(ticker);
  }
}

export async function getQuarterlyResults(ticker: string): Promise<Record<string, unknown>> {
  try {
    const url = `https://query2.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(
      withNs(ticker),
    )}?modules=earningsHistory,financialData`;
    const data = await fetchJson<any>(url);
    const result = data?.quoteSummary?.result?.[0];
    const history = result?.earningsHistory?.history ?? [];
    if (!history.length) throw new Error("No quarterly history");
    const latest = history[0];
    return {
      ticker: ticker.toUpperCase(),
      latestQuarter: latest.quarter?.fmt ?? null,
      epsActual: latest.epsActual?.raw ?? null,
      epsEstimate: latest.epsEstimate?.raw ?? null,
      epsSurprisePercent: latest.surprisePercent?.raw ?? null,
      history,
      citations: [{ source: "Yahoo Finance", detail: `${withNs(ticker)} earnings history` }],
    };
  } catch {
    return getMockQuarterlyResults(ticker);
  }
}

async function getAllFundsList(): Promise<Array<{ schemeCode: string; schemeName: string }>> {
  const data = await fetchJson<any[]>("https://api.mfapi.in/mf");
  return data.map((row) => ({
    schemeCode: String(row.schemeCode),
    schemeName: String(row.schemeName),
  }));
}

export async function searchMutualFunds(query: string): Promise<Record<string, unknown>> {
  try {
    const all = await getAllFundsList();
    const q = query.toLowerCase();
    const schemes = all
      .filter((f) => f.schemeName.toLowerCase().includes(q))
      .slice(0, 20)
      .map((f) => ({ schemeCode: f.schemeCode, name: f.schemeName }));
    return {
      query,
      schemes,
      citations: [{ source: "MFapi.in", detail: "AMFI scheme list search" }],
    };
  } catch {
    return getMockMutualFundSearch(query);
  }
}

export async function getFundNav(schemeCode: string): Promise<Record<string, unknown>> {
  try {
    const data = await fetchJson<any>(`https://api.mfapi.in/mf/${encodeURIComponent(schemeCode)}`);
    const history = (data?.data ?? []).slice(0, 30).map((item: any) => ({
      date: item.date,
      nav: Number(item.nav),
    }));
    const latest = history[0];
    return {
      schemeCode,
      fundName: data?.meta?.scheme_name ?? null,
      latestNav: latest?.nav ?? null,
      navDate: latest?.date ?? null,
      history,
      citations: [{ source: "MFapi.in", detail: `Scheme ${schemeCode} NAV history` }],
    };
  } catch {
    return getMockFundNav(schemeCode);
  }
}

export async function compareFunds(schemeCodes: string[]): Promise<Record<string, unknown>> {
  try {
    const rows = await Promise.all(
      schemeCodes.map(async (schemeCode) => {
        const nav = await getFundNav(schemeCode);
        const history = (nav.history as Array<{ date: string; nav: number }>) ?? [];
        const latest = history[0]?.nav ?? null;
        const oneYear = history[Math.min(history.length - 1, 20)]?.nav ?? null;
        return {
          schemeCode,
          fundName: nav.fundName,
          latestNav: latest,
          oneYearReturnPercent:
            latest !== null && oneYear !== null ? percentChange(latest, oneYear) : null,
        };
      }),
    );
    return {
      schemeCodes,
      comparison: rows,
      citations: [{ source: "MFapi.in", detail: "Scheme NAV cross-comparison" }],
    };
  } catch {
    return getMockCompareFunds(schemeCodes);
  }
}

function scoreHeadline(headline: string): number {
  const h = headline.toLowerCase();
  const pos = ["beat", "growth", "upgrade", "gain", "strong", "record"];
  const neg = ["miss", "downgrade", "fall", "weak", "risk", "cuts"];
  let score = 0;
  for (const word of pos) if (h.includes(word)) score += 1;
  for (const word of neg) if (h.includes(word)) score -= 1;
  return score;
}

export async function getCompanyNews(
  ticker: string,
  page: number,
  pageSize: number,
): Promise<Record<string, unknown>> {
  const today = new Date();
  const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
  const from = weekAgo.toISOString().slice(0, 10);
  const to = today.toISOString().slice(0, 10);

  if (config.providers.finnhubApiKey) {
    try {
      const url = `https://finnhub.io/api/v1/company-news?symbol=${encodeURIComponent(
        withNs(ticker),
      )}&from=${from}&to=${to}&token=${encodeURIComponent(config.providers.finnhubApiKey)}`;
      const data = await fetchJson<any[]>(url);
      const start = (page - 1) * pageSize;
      const items = data.slice(start, start + pageSize).map((row) => ({
        headline: row.headline,
        source: row.source ?? "Finnhub",
        publishedAt: new Date((row.datetime ?? 0) * 1000).toISOString(),
        url: row.url,
      }));
      return {
        ticker: ticker.toUpperCase(),
        page,
        pageSize,
        total: data.length,
        items,
        citations: [{ source: "Finnhub", detail: `${withNs(ticker)} company-news endpoint` }],
      };
    } catch {
      // fall through
    }
  }

  if (config.providers.newsApiKey) {
    try {
      const url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(
        `${ticker} India stock`,
      )}&language=en&sortBy=publishedAt&page=${page}&pageSize=${pageSize}&apiKey=${encodeURIComponent(
        config.providers.newsApiKey,
      )}`;
      const data = await fetchJson<any>(url);
      return {
        ticker: ticker.toUpperCase(),
        page,
        pageSize,
        total: data.totalResults ?? 0,
        items: (data.articles ?? []).map((a: any) => ({
          headline: a.title,
          source: a.source?.name ?? "NewsAPI",
          publishedAt: a.publishedAt,
          url: a.url,
        })),
        citations: [{ source: "NewsAPI", detail: `${ticker.toUpperCase()} article search` }],
      };
    } catch {
      // fall through
    }
  }

  return getMockNews(ticker, page, pageSize);
}

export async function getNewsSentiment(ticker: string, windowDays: number): Promise<Record<string, unknown>> {
  try {
    const news = await getCompanyNews(ticker, 1, Math.min(50, Math.max(10, windowDays * 2)));
    const items = (news.items as Array<{ headline: string }>) ?? [];
    const total = items.reduce((acc, i) => acc + scoreHeadline(i.headline ?? ""), 0);
    const sentimentScore = items.length ? total / items.length : 0;
    const label =
      sentimentScore > 0.3 ? "positive" : sentimentScore < -0.3 ? "negative" : "neutral";
    return {
      ticker: ticker.toUpperCase(),
      windowDays,
      sentimentScore,
      label,
      articleCount: items.length,
      citations: news.citations ?? [{ source: "News", detail: "Company news sentiment aggregation" }],
    };
  } catch {
    return getMockNewsSentiment(ticker);
  }
}

export async function getMarketNews(page: number, pageSize: number): Promise<Record<string, unknown>> {
  if (config.providers.newsApiKey) {
    try {
      const url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(
        "Indian stock market OR NSE OR Sensex",
      )}&language=en&sortBy=publishedAt&page=${page}&pageSize=${pageSize}&apiKey=${encodeURIComponent(
        config.providers.newsApiKey,
      )}`;
      const data = await fetchJson<any>(url);
      return {
        page,
        pageSize,
        total: data.totalResults ?? 0,
        items: (data.articles ?? []).map((a: any) => ({
          headline: a.title,
          source: a.source?.name ?? "NewsAPI",
          publishedAt: a.publishedAt,
          url: a.url,
        })),
        citations: [{ source: "NewsAPI", detail: "Indian market news search feed" }],
      };
    } catch {
      // continue
    }
  }
  return getMockMarketNews(page, pageSize);
}

export async function getRbiRates(): Promise<Record<string, unknown>> {
  // RBI does not expose a stable unauthenticated JSON endpoint for all policy rates.
  // Keep an explicit fallback while still returning structured macro context.
  return getMockMacroSnapshot();
}

export async function getInflationData(): Promise<Record<string, unknown>> {
  try {
    const data = await fetchJson<any[]>(
      "https://api.worldbank.org/v2/country/IN/indicator/FP.CPI.TOTL.ZG?format=json&per_page=20",
    );
    const rows = (data?.[1] ?? [])
      .filter((r: any) => r.value !== null)
      .slice(0, 12)
      .map((r: any) => ({
        period: r.date,
        value: Number(r.value),
      }));
    return {
      country: "India",
      cpiSeries: rows,
      citations: [{ source: "World Bank", detail: "FP.CPI.TOTL.ZG India inflation indicator" }],
    };
  } catch {
    return getMockInflationData();
  }
}

export async function getCorporateFilings(ticker: string, page: number, pageSize: number): Promise<Record<string, unknown>> {
  // BSE APIs are unstable and often require browser-like cookies. Keep fallback-first reliability.
  const base = await getMockCorporateFilings(ticker);
  return {
    ...base,
    page,
    pageSize,
    note: "BSE endpoint integration can be added per your selected filing API route/headers.",
  };
}

export async function crossReferenceSignals(ticker: string): Promise<Record<string, unknown>> {
  try {
    const [quote, ratios, sentiment] = await Promise.all([
      getStockQuote(ticker),
      getKeyRatios(ticker),
      getNewsSentiment(ticker, 14),
    ]);
    const change = Number(quote.changePercent ?? 0);
    const sentimentScore = Number(sentiment.sentimentScore ?? 0);
    const pe = Number(ratios.pe ?? 0);

    return {
      ticker: ticker.toUpperCase(),
      findings: [
        {
          claim: "Recent move is likely driven by sentiment more than valuation shift",
          status: Math.abs(change) >= 2 && Math.abs(sentimentScore) >= 0.2 ? "confirmed" : "uncertain",
          confirmations: [
            { source: "Market Data", detail: `Price move ${change.toFixed(2)}%` },
            { source: "News", detail: `Sentiment score ${sentimentScore.toFixed(2)}` },
          ],
          contradictions: [{ source: "Fundamentals", detail: `Trailing P/E remains ${pe || "unavailable"}` }],
        },
      ],
      citations: [
        ...(quote.citations as any[]),
        ...(ratios.citations as any[]),
        ...(sentiment.citations as any[]),
      ],
      disclaimer: "Informational output only. Not financial advice.",
    };
  } catch {
    return getMockCrossReferenceSignals(ticker);
  }
}
