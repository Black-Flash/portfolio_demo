# portfolio_demo
With CSV (template) file, track your crypto assets in CEX &amp; Wallet

After you execute it, you can get
1) Your total asset value : loans + equity + PNL + NFT
2) Your loans value and its info : current ltv, liq price, how much it drops for liquidiation
3) 

1. 100% Prompt-driven development (GPT 5.0)
2. Only Used Coingecko API (No CEX API)
3. Tried Using OPENSEA API for NFT but failed
4. This file provides real-time price of your assets, your PNL on futures, your NFT, your current loans and Total value.
5. Dollar($) based

# How to use Template

I recommend you to fill the column which is 'bold'-written.

A. You can track the price of assets in CEX, Wallet.

For CEX, you can change the name(binance, byit, bitget) to any CEX you want.
For Wallet, you can set the name of it freely like Wallet1, Wallet2 and etc.

There are 6 columns for each block; name, symbol, coingecko_id, quantity, manual_price_usd, and notes.
- **name **: type its name
- **symbol** : Ticker
- **coingecko_id** : This is really important. You should not just type 'its ticker'. You can check 'coingecko_id' on the info of each coin found in coingecko. Like that, you should check its 'api id' of each coin and type it on 'coingecko_id'
<img width="583" height="688" alt="image" src="https://github.com/user-attachments/assets/22007ba4-65e0-4db1-9973-84a6a4b1a4bb" />
- **quantity** : the amount you have
- manual_price_usd : You can set its price 'manually'. If it is blank, py file would check its price using coingecko_id
- notes : memo

B. Perp

In Perp block, there are 7 blocks; Symbol, Long/Short, Leverage, Entry, Margin, Liq, and PNL

- **Symbol** : Type your asset's name whose position is open
- **Long/Short** : Long or Short
- **Leverage** : Your leverage
- **Entry** : Entry price
- **Margin** : Money you used when you open your position
- Liq : Liquidation price
- PNL : Profit and Loss

C. NFT

5 blocks : name, opensea_slug, contract_address, chain, manual_price_usd
- opnesea_slug : you can check it on url
<img width="737" height="51" alt="image" src="https://github.com/user-attachments/assets/f980489d-b3cb-46e6-aab5-ec4d3794aed3" />
- contract_address : NFT's
- chain : eth, ...
- manual_price_usd
- **floor_native & native_symbol** : If its current price is 20 ETH, you should type 20 in floor_native & ETH in native_symbol

[warning]
I tried to fetch the FP of NFT's asset from opensea with slugs & contract address but failed. In this version, you can't help typing its 'manual_price_usd' or floor_native & native_symbol

D. Loans

This is just optional but I think you might leveraging your position with your crpto spot assets. 
There are 8 blocks; source_name, loan_assets, principal, collateral_asset, collateral_qty, current_ltv, apr_override, and notes
- **source_name** : CEX or Defi protocol you are using
- **loan_asset** : What you borrowed
- **principal** : How much you borrow; the number of your loan_asset
- **collateral_asset** : What you collaterize for your loans
- **collateral_qty** : How much you collaterize
- current_ltv
- apr_override : it would be calculated based on APR 8%. In the result, you can check 30-day interest based on APR 8%
- notes : memo

E. Exclude

You might want the result which the value you set would be excluded from your total assets.
There are 4 blocks; type, value, source_name, asset
- type : asset, NFT, perp and etc. just for checking yourself
- value : how many
- source_name : this is same with '**coingecko_id**'.
- asset : BTC, ETH, and etc; anything you want to exclude

  
# Flow tracking the price
Coingecko API -> somewhat error? -> use binance info; You don't have to take care about it

# Result after you execute this code with template.csv
<img width="1876" height="147" alt="image" src="https://github.com/user-attachments/assets/357f9c61-bb6e-4f36-ac54-87b31e5d8698" />






Any feedback is welcome :D
