# Lease-Up Dashboard

A Streamlit dashboard for analyzing apartment lease-up performance across Austin and Akron (2008-2020), with interactive visuals, clustering, and GenAI-powered narrative insights.

## What This App Does

- Loads market-level Excel data and engineers lease-up features.
- Tracks delivery, stabilization, lease-up speed, and rent growth.
- Builds property clusters using standardized features + UMAP/PCA + KMeans.
- Uses Google Gemini to explain results in plain English.

## Core Metrics Calculated

- `delivery_month`, `delivery_year`
- `stabilization_month`, `lease_up_months`, `did_not_stabilize`
- `eff_rent_delivery`, `eff_rent_endpoint`, `rent_growth_pct`, `neg_rent_growth`
- Engineered features:
  - `age_at_delivery`
  - `delivery_month_of_year`
  - `occ_ramp_rate`
  - `ask_eff_spread_at_delivery`
  - `ac_at_delivery`
  - `conc_at_delivery`
  - `size_tier`

## AI Components: How We Built Them

The app has two AI touchpoints:

1. **Ask AI tab (Q&A on filtered data)**
   - A compact data context is generated from the current filtered dataframe.
   - Context includes totals, stabilization stats, lease-up averages by market, negative rent growth, year range, and top slowest lease-ups.
   - User question is appended to this context and sent to Gemini (`gemini-2.5-flash`).

2. **Cluster interpretation assistant**
   - Cluster profile table (mean feature values by cluster) is converted to text.
   - Prompt asks Gemini to explain each cluster in business terms:
     - what property type it resembles,
     - why lease-up pace looks that way,
     - practical investor implication.

Both AI calls use one helper (`ask_gemini`) and are wrapped with simple error handling to avoid UI crashes.

## Prompt Engineering Techniques Used

We intentionally kept prompts simple and structured:

- **Role prompting**: "You are a real estate analyst" to anchor tone and domain.
- **Grounding with numeric context**: model receives computed aggregates and profile tables, not open-ended raw text.
- **Scoped instruction**: asks for concise and professional output, with a clear format (e.g., 3-4 sentences per cluster).
- **Data-bound behavior**: prompt explicitly asks the model to answer from provided data context.
- **Token efficiency**: sends summary statistics and curated records (top 10 slowest) rather than full dataset.

This balance keeps responses useful, readable, and cheaper to run.

## Local Run

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Add Streamlit secrets:
   - Create `.streamlit/secrets.toml`
   - Add:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

3. Start app:
   - `streamlit run app.py`

## Notes

- If you host data externally, configure secure links in Streamlit secrets under `[DATA_SOURCES]`.
- If local Excel files are present in repo root, app uses those by default.
