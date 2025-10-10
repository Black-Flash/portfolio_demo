#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Portfolio Runner — FINAL (single-file)
- FIX: NameError: valuate_nfts (function included)
- EXCLUDE robust + debug exports
- Loans with LTV, liquidation price, drawdown to liq, monthly interest (8% & custom APR)
- Hybrid pricing (CoinGecko -> Binance USDT fallback)
- NFT valuation: manual -> floor_native*native_symbol -> OpenSea (optional)
- Summary splits Assets / Liabilities / Equity + Excluded variants

Usage
  python last_portfolio_fixed.py --file template.csv --encoding utf-8-sig ^
    --price-source hybrid --opensea true --liq-ltv 0.80 --debug-exclude true
"""
import argparse, os, re, sys, json, math, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any

import requests
import pandas as pd

# ------------------------------- Utils -------------------------------
def now_utc():
    return datetime.now(timezone.utc)

def safe_float(x, default=None):
    try:
        if isinstance(x, str):
            sx = x.strip().replace(",", "").replace(" ", "")
            if sx in ("", "-", "nan", "NaN", "None", "null"):
                return default
            return float(sx)
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def is_finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False

def nz(x, fallback=0.0) -> float:
    try:
        fx = float(x)
        if math.isnan(fx):
            return float(fallback)
        return fx
    except Exception:
        return float(fallback)

def normalize_symbol(sym: str) -> str:
    if not sym:
        return ""
    return re.sub(r"[^A-Z0-9]", "", sym.strip().upper())

def norm_text(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def section_kind(name: str) -> str:
    s = norm_text(name)
    if not s:
        return "unknown"
    exchanges = {"binance","bybit","bitget","okx","okex","huobi","htx","upbit","gate.io","gate","coinbase","kraken","kucoin","bitfinex","bitstamp","mexc","bithumb"}
    if s in exchanges:
        return "exchange"
    wallet_keywords = ("wallet","metamask","phantom","ledger","trezor","safe","rabby","xdefi","keplr","tronlink","okx wallet","coinbase wallet")
    if any(k in s for k in wallet_keywords):
        return "wallet"
    return "exchange"

# ------------------------------- Pricing -------------------------------
CG_BASE = "https://api.coingecko.com/api/v3"
STABLES = {"USD","USDT","USDC","BUSD","FDUSD","DAI","PYUSD"}

CG_SHORT = {
    "BTC":"bitcoin","ETH":"ethereum","SOL":"solana","BNB":"binancecoin","XRP":"ripple","ADA":"cardano",
    "DOGE":"dogecoin","DOT":"polkadot","AVAX":"avalanche-2","MATIC":"polygon","LINK":"chainlink",
    "TRX":"tron","ATOM":"cosmos","NEAR":"near","ALGO":"algorand","ETC":"ethereum-classic","BCH":"bitcoin-cash",
    "LTC":"litecoin","UNI":"uniswap","AAVE":"aave","INJ":"injective","RUNE":"thorchain","STX":"stacks",
    "ORDI":"ordinals","JUP":"jupiter-exchange-solana","PEPE":"pepe","WIF":"dogwifcoin","BONK":"bonk",
    "WBTC":"wrapped-bitcoin","WETH":"weth","ENA":"ethena","ETHFI":"ether-fi","PENDLE":"pendle","BLUR":"blur","ALT":"altlayer",
    "SUI":"sui","APT":"aptos","TIA":"celestia","SEI":"sei-network","TAO":"bittensor","GALA":"gala",
    "IO":"io-net","ZEN":"horizen","ZEC":"zcash"
}
BINANCE_BASE = "https://api.binance.com"

def http_get_json(url: str, params=None, headers=None, timeout=25):
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

def cg_resolve_id(symbol: str) -> Optional[str]:
    if not symbol:
        return None
    if symbol.upper() in CG_SHORT:
        return CG_SHORT[symbol.upper()]
    try:
        data = http_get_json(f"{CG_BASE}/search", params={"query": symbol.lower()})
        coins = data.get("coins", [])
        exact = [c for c in coins if (c.get("symbol","") or "").lower() == symbol.lower()]
        if exact:
            return exact[0]["id"]
        if coins:
            return coins[0]["id"]
    except Exception:
        pass
    return None

def cg_simple_price(ids: List[str], tries=3, sleep_base=0.8) -> Dict[str,float]:
    out = {}
    if not ids: return out
    uniq = sorted(set(ids))
    for attempt in range(tries):
        try:
            data = http_get_json(f"{CG_BASE}/simple/price", params={"ids": ",".join(uniq), "vs_currencies":"usd"})
            for cid in uniq:
                px = data.get(cid, {}).get("usd")
                if px is not None:
                    out[cid] = float(px)
            return out
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code in (429, 502, 503, 504):
                time.sleep(sleep_base * (2 ** attempt))
                continue
            else:
                break
        except Exception:
            time.sleep(sleep_base * (2 ** attempt))
    return out

def prices_from_coingecko(rows: List[dict]) -> Dict[str, float]:
    ids = {}
    for r in rows:
        sym = normalize_symbol(r.get("symbol",""))
        if not sym or sym in STABLES:
            continue
        cid = (r.get("coingecko_id") or "").strip().lower()
        if not cid:
            cid = cg_resolve_id(sym)
        if cid:
            ids[sym] = cid
    out = {}
    if ids:
        id2px = cg_simple_price(list(ids.values()))
        for sym, cid in ids.items():
            px = id2px.get(cid)
            if px is not None:
                out[sym] = px
    for r in rows:
        sym = normalize_symbol(r.get("symbol",""))
        if sym in STABLES:
            out[sym] = 1.0
    return out

def binance_price_usdt(symbol: str) -> Optional[float]:
    pairs = [f"{symbol}USDT", f"{symbol}BUSD", f"{symbol}FDUSD"]
    for sym in pairs:
        try:
            data = http_get_json(f"{BINANCE_BASE}/api/v3/ticker/price", params={"symbol": sym})
            px = safe_float(data.get("price"))
            if is_finite(px):
                return float(px)
        except Exception:
            continue
    return None

def prices_hybrid(rows: List[dict]) -> Dict[str,float]:
    out = prices_from_coingecko(rows)
    miss_syms = []
    for r in rows:
        sym = normalize_symbol(r.get("symbol",""))
        if not sym: continue
        if sym in STABLES:
            out[sym] = 1.0
            continue
        if sym not in out:
            miss_syms.append(sym)
    if miss_syms:
        filled = []
        for sym in miss_syms:
            px = binance_price_usdt(sym)
            if is_finite(px):
                out[sym] = float(px)
                filled.append(sym)
        if filled:
            print(f"[info] Filled prices via Binance for: {', '.join(sorted(set(filled)))} (USDT≈USD)")
        still = [s for s in miss_syms if s not in out]
        if still:
            print(f"[warn] Missing price after hybrid fallback: {', '.join(sorted(set(still)))}. Consider adding 'coingecko_id' or 'manual_price_usd'.")
    return out

def prices_from_rows(rows: List[dict], mode="hybrid") -> Dict[str,float]:
    if mode == "gecko":
        return prices_from_coingecko(rows)
    elif mode == "binance":
        out = {}
        for r in rows:
            sym = normalize_symbol(r.get("symbol",""))
            if not sym: continue
            if sym in STABLES:
                out[sym] = 1.0
            else:
                px = binance_price_usdt(sym)
                if is_finite(px):
                    out[sym] = float(px)
        return out
    else:
        return prices_hybrid(rows)

# ------------------------------- OpenSea (NFT) -------------------------------
CHAIN_ALIAS = {
    "eth":"ethereum","ethereum":"ethereum","mainnet":"ethereum","eth mainnet":"ethereum","ethereum mainnet":"ethereum",
    "polygon":"polygon","matic":"polygon","polygon pos":"polygon",
    "arbitrum":"arbitrum","arb":"arbitrum","arbitrum one":"arbitrum",
    "optimism":"optimism","op":"optimism","optimism mainnet":"optimism","op mainnet":"optimism",
    "base":"base","zora":"zora","linea":"linea","scroll":"scroll",
    "bsc":"bsc","bnb":"bsc","bnb chain":"bsc","binance smart chain":"bsc",
    "avalanche":"avalanche","avax":"avalanche","avalanche c-chain":"avalanche",
    "solana":"solana"
}
CHAIN_NATIVE = {
    "ethereum":"ETH","polygon":"MATIC","arbitrum":"ETH","optimism":"ETH","base":"ETH","zora":"ETH","linea":"ETH","scroll":"ETH",
    "bsc":"BNB","avalanche":"AVAX","solana":"SOL"
}
CHAIN_TRY = ["ethereum","base","arbitrum","optimism","polygon","zora","linea","scroll","bsc","avalanche","solana"]

def norm_chain(s: Optional[str]) -> Optional[str]:
    if not s: return None
    key = norm_text(s)
    return CHAIN_ALIAS.get(key, key)

def _normalize_contract(addr: Optional[str]) -> Optional[str]:
    if not addr: return None
    a = addr.strip()
    if a.startswith("0x"):
        return a.lower()
    return a

def opensea_slug_from_contract(chain: str, address: str, api_key: Optional[str]) -> Optional[str]:
    url = "https://api.opensea.io/api/v2/collections"
    headers = {"accept":"application/json"}
    if api_key: headers["X-API-KEY"] = api_key
    try:
        data = http_get_json(url, params={"chain": chain, "address": address}, headers=headers)
        if isinstance(data, dict) and "collections" in data:
            cols = data.get("collections") or []
        else:
            cols = data if isinstance(data, list) else []
        if cols:
            slug = cols[0].get("slug") or cols[0].get("collection","")
            return slug
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            print(f"[info] OpenSea rate-limited on chain={chain} for contract; will try other chains")
        return None
    except Exception:
        return None
    return None

def opensea_slug_from_contract_multi(address: str, api_key: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    addr = _normalize_contract(address)
    for ch in CHAIN_TRY:
        slug = opensea_slug_from_contract(ch, addr, api_key)
        if slug:
            print(f"[info] OpenSea slug resolved via contract on chain={ch}: {slug}")
            return slug, ch
    return None, None

def opensea_floor_native(slug: str, api_key: Optional[str]) -> Optional[float]:
    headers = {"accept":"application/json"}
    if api_key: headers["X-API-KEY"] = api_key
    url = f"https://api.opensea.io/api/v2/collections/{slug}/stats"
    try:
        data = http_get_json(url, headers=headers)
        total = data.get("total") if isinstance(data, dict) else None
        val = None
        if isinstance(total, dict):
            val = total.get("floor_price")
        if val is None and isinstance(data, dict):
            val = data.get("floor_price")
        return safe_float(val, None)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            print("[info] OpenSea floor stats rate-limited (429).")
        return None
    except Exception:
        return None

def opensea_collection_info(slug: str, api_key: Optional[str]) -> dict:
    headers = {"accept":"application/json"}
    if api_key: headers["X-API-KEY"] = api_key
    url = f"https://api.opensea.io/api/v2/collections/{slug}"
    try:
        data = http_get_json(url, headers=headers)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def infer_chain_from_collection(slug: str, api_key: Optional[str]) -> Optional[str]:
    info = opensea_collection_info(slug, api_key)
    chains = info.get("chains")
    if isinstance(chains, list) and chains:
        ch = chains[0].get("chain")
        chn = norm_chain(ch)
        if chn:
            return chn
    for k in ("primary_chain","chain"):
        ch = info.get(k)
        if ch:
            chn = norm_chain(ch)
            if chn:
                return chn
    return None

# ------------------------------- CSV & Parsing -------------------------------
def read_csv_safely(path: Path, enc_hint: Optional[str]=None) -> pd.DataFrame:
    order = []
    if enc_hint: order.append(enc_hint)
    order += ["utf-8-sig","utf-8","cp949","euc-kr","latin1"]
    tried = []
    for enc in dict.fromkeys(order):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[info] CSV loaded with encoding: {enc}")
            return df
        except Exception as e:
            tried.append(f"{enc}: {e.__class__.__name__}: {e}")
    print("[error] Could not read CSV with tried encodings:")
    for t in tried: print("  -", t)
    raise RuntimeError("CSV read failed")

def parse_template(df: pd.DataFrame):
    holdings_rows, perp_rows, nft_rows, loan_rows, exclude_rows = [], [], [], [], []
    nrows = df.shape[0]
    r = 0
    while r < nrows:
        row = df.iloc[r]
        first_idx = row.first_valid_index()
        if first_idx is None:
            r += 1; continue
        first_val = str(row[first_idx]).strip()

        # PERP block
        if first_val.lower() == "perp" and r+1 < nrows:
            header = [str(c).strip() for c in list(df.iloc[r+1].fillna(""))]
            cmap = {k: i for i, k in enumerate(header)}
            rr = r + 2
            while rr < nrows and not all(pd.isna(df.iloc[rr])):
                row2 = list(df.iloc[rr])
                sym = str(row2[cmap.get("Symbol", 0)]).strip() if "Symbol" in cmap else ""
                side = str(row2[cmap.get("Long/Short", 1)]).strip().lower() if "Long/Short" in cmap else ""
                lev  = safe_float(row2[cmap.get("Leverage", 2)], None) if "Leverage" in cmap else None
                entry = safe_float(row2[cmap.get("Entry", 3)], None) if "Entry" in cmap else None
                margin = safe_float(row2[cmap.get("Margin", 4)], None) if "Margin" in cmap else None
                if sym:
                    perp_rows.append({"symbol": normalize_symbol(sym), "side": ("long" if "short" not in side else "short"),
                                      "leverage": lev, "entry": entry, "margin": margin})
                rr += 1
            r = rr + 1; continue

        # NFT block
        if first_val.lower() == "nft" and r+1 < nrows:
            header = [str(c).strip().lower() for c in list(df.iloc[r+1].fillna(""))]
            cmap = {k: i for i, k in enumerate(header)}
            rr = r + 2
            while rr < nrows and not all(pd.isna(df.iloc[rr])):
                row2 = list(df.iloc[rr])
                name = str(row2[cmap.get("name", 0)]).strip() if "name" in cmap else ""
                slug = str(row2[cmap.get("opensea_slug", cmap.get("symbol", 1))]).strip().lower() if ("opensea_slug" in cmap or "symbol" in cmap) else ""
                contract = str(row2[cmap.get("contract_address", -1)]).strip() if "contract_address" in cmap else ""
                chain = str(row2[cmap.get("chain", -1)]).strip().lower() if "chain" in cmap else ""
                qty  = safe_float(row2[cmap.get("quantity", 3)], None) if "quantity" in cmap else None
                mp   = safe_float(row2[cmap.get("manual_price_usd", 4)], None) if "manual_price_usd" in cmap else None
                floor_nat = safe_float(row2[cmap.get("floor_native", 5)], None) if "floor_native" in cmap else None
                native_sym = str(row2[cmap.get("native_symbol", 6)]).strip().upper() if "native_symbol" in cmap else ""
                if name:
                    nft_rows.append({"collection": name, "opensea_slug": (slug or None),
                                     "contract_address": (contract or None), "chain": (chain or None),
                                     "quantity": qty, "manual_price_usd": mp,
                                     "floor_native": floor_nat, "native_symbol": native_sym})
                rr += 1
            r = rr + 1; continue

        # LOANS block
        if first_val.lower() == "loans" and r+1 < nrows:
            header = [str(c).strip().lower() for c in list(df.iloc[r+1].fillna(""))]
            cmap = {k: i for i, k in enumerate(header)}
            rr = r + 2
            while rr < nrows and not all(pd.isna(df.iloc[rr])):
                row2 = list(df.iloc[rr])
                source = str(row2[cmap.get("source_name", 0)]).strip() if "source_name" in cmap else "manual"
                loan_asset = str(row2[cmap.get("loan_asset", 1)]).strip() if "loan_asset" in cmap else ""
                principal  = safe_float(row2[cmap.get("principal", 2)], None) if "principal" in cmap else None
                coll_asset = str(row2[cmap.get("collateral_asset", 3)]).strip() if "collateral_asset" in cmap else ""
                coll_qty   = safe_float(row2[cmap.get("collateral_qty", 4)], None) if "collateral_qty" in cmap else None
                current_ltv = safe_float(row2[cmap.get("current_ltv", 5)], None) if "current_ltv" in cmap else None
                apr_over   = safe_float(row2[cmap.get("apr_override", 6)], None) if "apr_override" in cmap else None
                if loan_asset and principal is not None and principal > 0:
                    loan_rows.append({
                        "source_name": source, "loan_asset": normalize_symbol(loan_asset), "principal": principal,
                        "collateral_asset": normalize_symbol(coll_asset) if coll_asset else None,
                        "collateral_qty": coll_qty, "current_ltv": current_ltv, "apr_override": apr_over
                    })
                rr += 1
            r = rr + 1; continue

        # EXCLUDE block
        if first_val.lower() == "exclude" and r+1 < nrows:
            header = [str(c).strip().lower() for c in list(df.iloc[r+1].fillna(""))]
            cmap = {k: i for i, k in enumerate(header)}
            rr = r + 2
            def get(row2, *names):
                for nm in names:
                    if nm in cmap:
                        return row2[cmap[nm]]
                return ""
            while rr < nrows and not all(pd.isna(df.iloc[rr])):
                row2 = list(df.iloc[rr])
                t   = str(get(row2, "type","Type")).strip().lower()
                v   = str(get(row2, "value","Value")).strip()
                src = str(get(row2, "source_name","Source_Name","source","Source")).strip()
                asset = str(get(row2, "asset","Asset","symbol","Symbol")).strip()
                if t:
                    exclude_rows.append({"type": t, "value": v, "source_name": src, "asset": asset})
                rr += 1
            r = rr + 1; continue

        # Generic holdings block
        if r + 1 < nrows:
            header = [str(c).strip().lower() for c in list(df.iloc[r+1].fillna(""))]
            if "symbol" in header and "quantity" in header:
                cmap = {k: i for i, k in enumerate(header)}
                source_name = first_val
                s_type = section_kind(source_name)
                rr = r + 2
                while rr < nrows and not all(pd.isna(df.iloc[rr])):
                    row2 = list(df.iloc[rr])
                    name = str(row2[cmap.get("name", 0)]).strip() if "name" in cmap else ""
                    sym  = str(row2[cmap.get("symbol", 1)]).strip() if "symbol" in cmap else ""
                    cid  = str(row2[cmap.get("coingecko_id", 2)]).strip() if "coingecko_id" in cmap else ""
                    qty  = safe_float(row2[cmap.get("quantity", 3)], None) if "quantity" in cmap else None
                    mp   = safe_float(row2[cmap.get("manual_price_usd", 4)], None) if "manual_price_usd" in cmap else None
                    if sym and qty is not None and qty > 0:
                        holdings_rows.append({
                            "source_type": s_type, "source_name": source_name,
                            "name": name, "symbol": normalize_symbol(sym), "coingecko_id": (cid or None),
                            "quantity": qty, "manual_price_usd": mp
                        })
                    rr += 1
                r = rr + 1; continue

        r += 1

    return holdings_rows, perp_rows, nft_rows, loan_rows, exclude_rows

# ------------------------------- Valuation -------------------------------
def valuate_holdings(holdings_rows: List[dict], price_mode="hybrid"):
    manual_price = {}
    for r in holdings_rows:
        sym = r["symbol"]
        mp = r.get("manual_price_usd")
        if mp is not None and mp > 0:
            manual_price[sym] = mp
    price_rows = [{"symbol": r["symbol"], "coingecko_id": r.get("coingecko_id")} for r in holdings_rows]
    prices = prices_from_rows(price_rows, mode=price_mode)
    for s, px in manual_price.items():
        prices[s] = px
    detailed = []
    missing = []
    for r in holdings_rows:
        sym = r["symbol"]
        qty = r["quantity"]
        if not is_finite(qty) or qty <= 0:
            continue
        px  = prices.get(sym)
        if px is None:
            missing.append(sym)
        val = qty * px if (px is not None) else None
        detailed.append({
            "source_type": r["source_type"], "source_name": r["source_name"],
            "asset": sym, "qty": qty, "price_usd": px, "value_usd": val,
            "price_source": ("manual" if sym in manual_price else ("market" if px is not None else None)),
            "name": r.get("name"), "coingecko_id": r.get("coingecko_id")
        })
    if missing:
        uniq = ", ".join(sorted(set(missing)))
        print(f"[warn] Missing USD price for: {uniq}. Fill 'coingecko_id' or set Manual_Price_USD.")
    df = pd.DataFrame(detailed, columns=["source_type","source_name","asset","qty","price_usd","value_usd","price_source","name","coingecko_id"])
    if not df.empty:
        df["value_usd"] = pd.to_numeric(df["value_usd"], errors="coerce").fillna(0.0)
        g = df.groupby(["source_type","source_name"], dropna=False)["value_usd"].sum().reset_index().rename(columns={"value_usd":"total_value_usd"})
        total = float(df["value_usd"].sum())
    else:
        g = pd.DataFrame(columns=["source_type","source_name","total_value_usd"])
        total = 0.0
    return df, g, total, prices

def valuate_perp(perp_rows: List[dict], prices: Dict[str,float]):
    out = []
    upnl_total = 0.0
    for p in perp_rows:
        sym = p.get("symbol","")
        side = (p.get("side","long") or "long").lower()
        lev = safe_float(p.get("leverage", None), None)
        entry = safe_float(p.get("entry", None), None)
        margin = safe_float(p.get("margin", None), None)
        if not (sym and entry and margin and lev and is_finite(entry) and is_finite(margin) and is_finite(lev) and lev>0 and entry>0 and margin>0):
            out.append({
                "symbol": sym, "side": side, "leverage": lev, "entry": entry, "margin": margin,
                "qty": None, "mark_price": None, "notional_usd": None, "unrealized_pnl_usd": None
            })
            continue
        qty = (margin * lev) / entry
        mark = prices.get(sym)
        notional = qty * mark if (mark is not None) else None
        upnl = None
        if mark is not None:
            upnl = (entry - mark) * qty if side == "short" else (mark - entry) * qty
            upnl_total += nz(upnl, 0.0)
        out.append({
            "symbol": sym, "side": side, "leverage": lev, "entry": entry, "margin": margin,
            "qty": qty, "mark_price": mark, "notional_usd": notional, "unrealized_pnl_usd": upnl
        })
    df = pd.DataFrame(out, columns=["symbol","side","leverage","entry","margin","qty","mark_price","notional_usd","unrealized_pnl_usd"])
    if not df.empty:
        df["unrealized_pnl_usd"] = pd.to_numeric(df["unrealized_pnl_usd"], errors="coerce").fillna(0.0)
        upnl_total = float(df["unrealized_pnl_usd"].sum())
    else:
        upnl_total = 0.0
    return df, upnl_total

def valuate_loans(loan_rows: List[dict], prices: Dict[str,float], liq_ltv: float):
    out = []
    liabilities_total = 0.0
    for it in loan_rows:
        src = it["source_name"]
        loan_asset = it["loan_asset"]
        principal = safe_float(it.get("principal",0.0), 0.0) or 0.0
        coll_asset = it.get("collateral_asset")
        coll_qty   = safe_float(it.get("collateral_qty",0.0), None) if it.get("collateral_qty") is not None else None
        current_ltv = safe_float(it.get("current_ltv", None), None)
        apr_over = safe_float(it.get("apr_override", None), None)

        loan_px = prices.get(loan_asset, 1.0 if loan_asset in STABLES else None)
        coll_px = prices.get(coll_asset, None) if coll_asset else None

        principal_usd = (principal * loan_px) if (is_finite(principal) and loan_px is not None) else None
        collateral_usd = ((coll_qty or 0.0) * coll_px) if (collateral_asset := coll_asset) and is_finite(coll_qty) and coll_px is not None else None

        if current_ltv is None and (principal_usd is not None) and (collateral_usd not in (None, 0)):
            current_ltv = principal_usd / collateral_usd

        monthly_interest_8pct = (principal_usd or 0.0) * 0.08 * (30.0/365.0) if principal_usd is not None else None
        monthly_interest_custom = (principal_usd or 0.0) * (apr_over or 0.0) * (30.0/365.0) if (principal_usd is not None and apr_over) else None

        liq_price_usd = None
        drawdown_to_liq_pct = None
        drawdown_to_liq_abs = None
        if (principal_usd is not None) and (is_finite(coll_qty) and coll_qty > 0) and (liq_ltv and liq_ltv > 0):
            liq_price_usd = principal_usd / (coll_qty * liq_ltv)
            if is_finite(coll_px):
                drawdown_to_liq_pct = (liq_price_usd / coll_px) - 1.0
                drawdown_to_liq_abs = coll_px - liq_price_usd

        if principal_usd is not None and is_finite(principal_usd):
            liabilities_total += principal_usd

        out.append({
            "source_name": src, "loan_asset": loan_asset, "principal": principal, "principal_usd": principal_usd,
            "collateral_asset": coll_asset, "collateral_qty": coll_qty, "collateral_usd": collateral_usd,
            "current_ltv": current_ltv, "monthly_interest_8pct": monthly_interest_8pct,
            "monthly_interest_custom": monthly_interest_custom, "apr_used_custom": apr_over,
            "liq_ltv_used": liq_ltv, "liq_price_usd": liq_price_usd,
            "drawdown_to_liq_pct": drawdown_to_liq_pct, "drawdown_to_liq_abs_usd": drawdown_to_liq_abs
        })
    df = pd.DataFrame(out, columns=["source_name","loan_asset","principal","principal_usd","collateral_asset","collateral_qty","collateral_usd","current_ltv","monthly_interest_8pct","monthly_interest_custom","apr_used_custom","liq_ltv_used","liq_price_usd","drawdown_to_liq_pct","drawdown_to_liq_abs_usd"])
    if not df.empty:
        df["principal_usd"] = pd.to_numeric(df["principal_usd"], errors="coerce").fillna(0.0)
        liabilities_total = float(df["principal_usd"].sum())
    else:
        liabilities_total = 0.0
    return df, liabilities_total

# --- NFT valuation (NameError fix, complete) ---
def valuate_nfts(nft_rows, use_os=False, os_api_key=None, prices=None):
    """
    Priority:
      1) manual_price_usd
      2) floor_native * native_symbol(USD)
      3) OpenSea floor (optional): slug or contract(+chain) -> floor_native * native chain price
    Return: (df_nft, total_nft_usd)
    """
    out, total = [], 0.0
    if use_os and not os_api_key:
        print("[info] OPENSEA_API_KEY not set. Prefer 'manual_price_usd' OR 'floor_native+native_symbol'. OpenSea may rate-limit.")

    need_manual = []
    for n in nft_rows:
        raw_qty = n.get("quantity", 1.0)
        qty = safe_float(raw_qty, 1.0)
        if not is_finite(qty) or qty <= 0:
            print(f"[warn] NFT row qty invalid -> defaulting to 1.0 | collection={n.get('collection')} qty={raw_qty}")
            qty = 1.0

        price = safe_float(n.get("manual_price_usd"), None)
        src = "manual" if is_finite(price) else None

        floor_native = safe_float(n.get("floor_native"), None)
        native_sym = (n.get("native_symbol") or "").upper()

        slug = (n.get("opensea_slug") or "").strip().lower()
        chain = norm_chain(n.get("chain")) if 'norm_chain' in globals() else n.get("chain")
        contract = n.get("contract_address")
        if isinstance(contract, str) and contract.startswith("0x"):
            contract = contract.lower()

        # (B) floor_native * native_symbol → USD
        if src is None and is_finite(floor_native) and native_sym:
            native_sym_norm = normalize_symbol(native_sym)
            px_native = prices.get(native_sym_norm) if prices else None
            if px_native is None:
                px_native = prices_from_rows([{"symbol": native_sym_norm}], mode="hybrid").get(native_sym_norm)
            if is_finite(px_native):
                price = float(floor_native) * float(px_native)
                src = "floor_native_override"

        # (C) OpenSea (optional)
        if src is None and use_os and (slug or contract):
            try:
                if slug and not chain and 'infer_chain_from_collection' in globals():
                    chain = infer_chain_from_collection(slug, os_api_key) or chain
                if not slug and contract:
                    if chain and 'opensea_slug_from_contract' in globals():
                        slug = opensea_slug_from_contract(chain, contract, os_api_key)
                    elif 'opensea_slug_from_contract_multi' in globals():
                        slug, chain = opensea_slug_from_contract_multi(contract, os_api_key)

                if slug and 'opensea_floor_native' in globals():
                    floor_native_os = opensea_floor_native(slug, os_api_key)
                    if is_finite(floor_native_os):
                        chain_for_symbol = chain or (infer_chain_from_collection(slug, os_api_key) if 'infer_chain_from_collection' in globals() else None) or "ethereum"
                        native2 = CHAIN_NATIVE.get(chain_for_symbol, "ETH") if 'CHAIN_NATIVE' in globals() else "ETH"
                        px_native2 = prices.get(native2) if prices else None
                        if px_native2 is None:
                            px_native2 = prices_from_rows([{"symbol": native2}], mode="hybrid").get(native2)
                        if is_finite(px_native2):
                            price = float(floor_native_os) * float(px_native2)
                            floor_native = float(floor_native_os)
                            native_sym = native2
                            src = "opensea_floor_native"
                            print(f"[info] NFT floor via OpenSea | collection={n.get('collection')} slug={slug} chain={chain_for_symbol} floor_native={floor_native_os} native={native2}")
            except Exception:
                pass

        if not is_finite(price):
            need_manual.append(n.get("collection","(unknown)"))
            price = 0.0

        val = float(price) * float(qty)
        total += val
        out.append({
            "collection": n.get("collection",""),
            "opensea_slug": (slug or None) if slug else None,
            "contract_address": contract,
            "chain": chain,
            "qty": qty,
            "est_price_usd_per_item": (float(price) if price else None) if price else None,
            "value_usd": val,
            "price_source": src,
            "floor_native": (float(floor_native) if is_finite(floor_native) else None),
            "native_symbol": (native_sym or None)
        })

    if need_manual:
        print("[warn] NFT price unresolved for:", ", ".join(sorted(set(need_manual))),
              "| Provide 'manual_price_usd' OR 'floor_native'+'native_symbol' OR set OPENSEA_API_KEY & slug/contract.")

    df = pd.DataFrame(
        out,
        columns=["collection","opensea_slug","contract_address","chain","qty",
                 "est_price_usd_per_item","value_usd","price_source","floor_native","native_symbol"]
    )
    if not df.empty:
        df["value_usd"] = pd.to_numeric(df["value_usd"], errors="coerce").fillna(0.0)
        total = float(df["value_usd"].sum())
    else:
        total = 0.0
    return df, total

# ------------------------------- Exclusions -------------------------------

def build_exclude_sets(exclude_rows: List[dict]):
    """
    Robust EXCLUDE reader.

    Accepts rows like:
      type=asset, source_name=binance, asset=ZEC            -> exclude this (source, asset) pair only
      type=asset, value=ZEC                                  -> exclude this asset globally
      type=holding|holding_by_source|pair, source_name=..., asset=... -> exclude pair
      type=source, value=binance                              -> exclude all rows from this source
      type=source_type, value=exchange|wallet                 -> exclude by source_type
      type=nft_collection, value=...                          -> exclude NFT by collection name
      type=nft_slug, value=...                                -> exclude NFT by OpenSea slug
    Blank/NaN/None/"-" values are treated as empty.
    """
    def _blank(v):
        if v is None:
            return True
        s = str(v).strip().lower()
        return s in ("", "nan", "none", "null", "-")

    ex_assets, ex_sources, ex_source_types, ex_nft_collections, ex_nft_slugs, ex_pairs = set(), set(), set(), set(), set(), set()

    for e in exclude_rows:
        t = norm_text(e.get("type"))
        raw_v = e.get("value")
        v = None if _blank(raw_v) else str(raw_v).strip()

        src = norm_text(e.get("source_name"))
        asset_field = e.get("asset")
        asset_norm = normalize_symbol(asset_field) if asset_field else None

        # Asset/Symbol rows:
        # - If a source_name is provided together with asset, interpret as a pair-only exclusion.
        # - Otherwise, exclude the asset globally (using 'value' first, then 'asset' column as fallback).
        if t in ("asset", "symbol"):
            if src and asset_norm:
                ex_pairs.add((src, asset_norm))
            else:
                asset_val = None
                if v and not _blank(v):
                    asset_val = normalize_symbol(v)
                elif asset_norm:
                    asset_val = asset_norm
                if asset_val:
                    ex_assets.add(asset_val)

        # Source-only exclusion
        elif t in ("source", "source_name"):
            if v and not _blank(v):
                ex_sources.add(norm_text(v))
            elif src:
                ex_sources.add(src)

        # Source type: "exchange", "wallet", etc.
        elif t in ("source_type",):
            if v and not _blank(v):
                ex_source_types.add(norm_text(v))

        # NFTs by collection
        elif t in ("nft_collection", "collection"):
            if v and not _blank(v):
                ex_nft_collections.add(norm_text(v))

        # NFTs by slug
        elif t in ("nft_slug", "slug"):
            if v and not _blank(v):
                ex_nft_slugs.add(norm_text(v))

        # Explicit pair syntax
        elif t in ("holding", "holding_by_source", "pair", "asset_by_source"):
            if src and asset_norm:
                ex_pairs.add((src, asset_norm))

    return ex_assets, ex_sources, ex_source_types, ex_nft_collections, ex_nft_slugs, ex_pairs


def apply_exclusions(df_hold: pd.DataFrame, df_nft: pd.DataFrame, ex_sets, debug=False):
    ex_assets, ex_sources, ex_source_types, ex_nft_cols, ex_nft_slugs, ex_pairs = ex_sets
    excluded_hold = pd.DataFrame(columns=df_hold.columns) if df_hold is not None and not df_hold.empty else pd.DataFrame()
    excluded_nft = pd.DataFrame(columns=df_nft.columns) if df_nft is not None and not df_nft.empty else pd.DataFrame()

    if df_hold is not None and not df_hold.empty and "value_usd" in df_hold:
        df = df_hold.copy()
        df["__src"]   = df["source_name"].astype(str).map(norm_text)
        df["__stype"] = df["source_type"].astype(str).map(norm_text)
        df["__asset"] = df["asset"].astype(str).map(normalize_symbol)

        mask = pd.Series(True, index=df.index)
        if ex_assets:
            mask &= ~df["__asset"].isin(ex_assets)
        if ex_sources:
            mask &= ~df["__src"].isin(ex_sources)
        if ex_source_types:
            mask &= ~df["__stype"].isin(ex_source_types)
        if ex_pairs:
            pair_mask = df.apply(lambda r: (r["__src"], r["__asset"]) in ex_pairs, axis=1)
            mask &= ~pair_mask

        excluded_hold = df.loc[~mask].drop(columns=["__src","__stype","__asset"])
        kept_hold     = df.loc[mask].drop(columns=["__src","__stype","__asset"])

        total_kept  = float(pd.to_numeric(kept_hold["value_usd"], errors="coerce").fillna(0.0).sum())
        total_excld = float(pd.to_numeric(excluded_hold["value_usd"], errors="coerce").fillna(0.0).sum())
    else:
        total_kept = 0.0
        total_excld = 0.0
        kept_hold = df_hold.copy() if df_hold is not None else pd.DataFrame()

    if df_nft is not None and not df_nft.empty and "value_usd" in df_nft:
        df = df_nft.copy()
        df["__col"] = df["collection"].astype(str).map(norm_text)
        df["__slug"] = df["opensea_slug"].astype(str).map(norm_text)

        maskn = pd.Series(True, index=df.index)
        if ex_nft_cols:
            maskn &= ~df["__col"].isin(ex_nft_cols)
        if ex_nft_slugs:
            maskn &= ~df["__slug"].isin(ex_nft_slugs)

        excluded_nft = df.loc[~maskn].drop(columns=["__col","__slug"])
        kept_nft     = df.loc[maskn].drop(columns=["__col","__slug"])
        total_kept_n  = float(pd.to_numeric(kept_nft["value_usd"], errors="coerce").fillna(0.0).sum())
        total_excld_n = float(pd.to_numeric(excluded_nft["value_usd"], errors="coerce").fillna(0.0).sum())
    else:
        kept_nft = df_nft.copy() if df_nft is not None else pd.DataFrame()
        total_kept_n = 0.0
        total_excld_n = 0.0

    if debug:
        if not excluded_hold.empty:
            print("[debug] Excluded HOLDINGS rows:")
            print(excluded_hold[["source_name","asset","qty","value_usd"]].to_string(index=False))
        else:
            print("[debug] Excluded HOLDINGS rows: (none)")
        if not excluded_nft.empty:
            print("[debug] Excluded NFT rows:")
            print(excluded_nft[["collection","qty","value_usd"]].to_string(index=False))
        else:
            print("[debug] Excluded NFT rows: (none)")

    return kept_hold, kept_nft, total_excld, total_excld_n, excluded_hold, excluded_nft

# ------------------------------- Main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to your template CSV")
    ap.add_argument("--encoding", default=None, help="CSV encoding (optional)")
    ap.add_argument("--opensea", default="false", choices=["true","false"], help="Use OpenSea for NFT floor")
    ap.add_argument("--liq-ltv", type=float, default=0.80, help="Liquidation LTV threshold (default: 0.80)")
    ap.add_argument("--output-prefix", default="portfolio_values", help="Output prefix for files")
    ap.add_argument("--price-source", default="hybrid", choices=["hybrid","gecko","binance"], help="Market price source logic")
    ap.add_argument("--debug-exclude", default="false", choices=["true","false"], help="Print which rows were excluded")
    args = ap.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"[fatal] File not found: {csv_path}")
        sys.exit(1)

    # Read & Parse
    df = read_csv_safely(csv_path, args.encoding)
    holdings_rows, perp_rows, nft_rows, loan_rows, exclude_rows = parse_template(df)

    # Prices universe
    syms = set(r["symbol"] for r in holdings_rows)
    syms |= set(p["symbol"] for p in perp_rows if p.get("symbol"))
    syms |= set(normalize_symbol(x["loan_asset"]) for x in loan_rows if x.get("loan_asset"))
    syms |= set(normalize_symbol(x["collateral_asset"]) for x in loan_rows if x.get("collateral_asset"))
    for n in nft_rows:
        ns = n.get("native_symbol")
        if ns:
            syms.add(normalize_symbol(ns))
    price_rows = [{"symbol": s} for s in syms if s]
    prices = prices_from_rows(price_rows, mode=args.price_source)

    # Valuations
    df_hold, df_source, total_hold, prices = valuate_holdings(holdings_rows, price_mode=args.price_source)
    df_perp, upnl_total = valuate_perp(perp_rows, prices)
    df_loans, total_liab = valuate_loans(loan_rows, prices, args.liq_ltv)

    use_os = (args.opensea.lower() == "true")
    df_nft, total_nft = valuate_nfts(nft_rows, use_os, os.environ.get("OPENSEA_API_KEY"), prices)

    # Exclusions
    ex_sets = build_exclude_sets(exclude_rows)
    debug = (args.debug_exclude.lower() == "true")
    kept_hold, kept_nft, excl_hold_usd, excl_nft_usd, excluded_hold, excluded_nft = apply_exclusions(df_hold, df_nft, ex_sets, debug=debug)

    # Totals
    total_hold = nz(total_hold, 0.0)
    total_nft = nz(total_nft, 0.0)
    upnl_total = nz(upnl_total, 0.0)
    total_liab = nz(total_liab, 0.0)

    assets_total = total_hold + total_nft + upnl_total
    equity_total = assets_total - total_liab

    # Excluded totals (holdings & NFTs only; perp upnl not excluded)
    assets_total_excl = float(pd.to_numeric(kept_hold["value_usd"], errors="coerce").fillna(0.0).sum()) \
                        + float(pd.to_numeric(kept_nft["value_usd"], errors="coerce").fillna(0.0).sum()) \
                        + upnl_total
    equity_total_excl = assets_total_excl - total_liab

    # Outputs
    ts = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    pref = args.output_prefix

    df_hold.to_csv(f"{pref}_detailed.csv", index=False)
    df_source.to_csv(f"{pref}_by_source.csv", index=False)
    df_perp.to_csv(f"{pref}_perp.csv", index=False)
    df_loans.to_csv(f"{pref}_loans.csv", index=False)
    df_nft.to_csv(f"{pref}_nfts.csv", index=False)
    excluded_hold.to_csv(f"{pref}_excluded_holdings.csv", index=False)
    excluded_nft.to_csv(f"{pref}_excluded_nfts.csv", index=False)

    summary = {
        "timestamp": ts,
        "base_currency": "USD",
        "parameters": {"liq_ltv": args.liq_ltv, "opensea": use_os, "price_source": args.price_source},
        "totals": {
            "assets_holdings_usd": round(assets_total - total_nft - upnl_total, 2),  # equals total_hold
            "assets_nfts_usd": round(total_nft, 2),
            "perp_unrealized_pnl_usd": round(upnl_total, 2),
            "assets_total_including_perp_upnl_usd": round(assets_total, 2),
            "liabilities_usd": round(total_liab, 2),
            "equity_usd": round(equity_total, 2),
            "excluded_holdings_usd": round(excl_hold_usd, 2),
            "excluded_nfts_usd": round(excl_nft_usd, 2),
            "assets_total_excluded_usd": round(assets_total_excl, 2),
            "equity_excluded_usd": round(equity_total_excl, 2)
        },
        "counts": {
            "holdings_rows": int(len(df_hold)),
            "sources": int(len(df_source)),
            "nft_items": int(len(df_nft)),
            "perp_rows": int(len(df_perp)),
            "loan_rows": int(len(df_loans)),
            "excluded_holdings_rows": int(len(excluded_hold)),
            "excluded_nft_rows": int(len(excluded_nft))
        }
    }
    with open(f"{pref}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[{ts}] ASSETS ${assets_total:,.2f} (holdings {assets_total - total_nft - upnl_total:,.2f} + nfts {total_nft:,.2f} + perpUPNL {upnl_total:,.2f}) | LIAB ${total_liab:,.2f} | EQUITY ${equity_total:,.2f}")
    print(f"[exclusion] -excluded: holdings ${excl_hold_usd:,.2f}, nfts ${excl_nft_usd:,.2f}")
    print(f"[exclusion] assets_excl ${assets_total_excl:,.2f} | equity_excl ${equity_total_excl:,.2f}")
    print("Saved ->", f"{pref}_detailed.csv, {pref}_by_source.csv, {pref}_perp.csv, {pref}_loans.csv, {pref}_nfts.csv, "
                    f"{pref}_excluded_holdings.csv, {pref}_excluded_nfts.csv, {pref}_summary.json")

if __name__ == "__main__":
    main()
