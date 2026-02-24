import scrapy
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import logging
import re
import seaborn as sns

from datetime import datetime, date
from urllib.parse import urlparse

mpl.rcParams["axes.facecolor"] = "0.95"


class DrillholeDataSpider(scrapy.Spider):
    name = "drillhole_data"
    allowed_domains = ["www.juniorminingnetwork.com"]
    start_urls = [
        "https://www.juniorminingnetwork.com/mining-topics/topic/drill-results.html?page=1&format=json"
    ]

    def __init__(self, price_dict, n_previous_days, *args, **kwargs):
        self.price_dict = price_dict
        self.n_previous_days = n_previous_days
        self.today = date.today()
        self.page = 1
        self.article_audit_log = []
        self.audit_log_path = "article_parse_audit.json"
        logger = logging.getLogger("matplotlib")  # supress mpl INFO output to terminal
        logger.setLevel(logging.ERROR)

    def parse(self, response):
        """parse JSON article objects"""
        articles = json.loads(response.text)["articles"]
        last_date = articles[-1]["formatted_date"]
        last_date = datetime.strptime(last_date.strip(), "%B %d, %Y").date()
        last_delta = (self.today - last_date).days

        # check if we will need the next page or not
        if last_delta <= self.n_previous_days:
            next_page = True
        else:
            next_page = False

        # parse articles if they are within the date range
        for article in articles:
            raw_date = article.get("formatted_date")
            raw_link = article.get("link")

            try:
                date = datetime.strptime(raw_date.strip(), "%B %d, %Y").date()
            except (AttributeError, ValueError):
                self._record_article_status(
                    status="failed",
                    reason="invalid_article_date",
                    article_link=raw_link,
                    article_date=raw_date,
                )
                continue

            delta = (self.today - date).days
            if delta <= self.n_previous_days:
                url = response.urljoin(raw_link) if raw_link else None
                if not self._is_valid_url(url):
                    self._record_article_status(
                        status="failed",
                        reason="invalid_article_url",
                        article_link=raw_link,
                        article_date=raw_date,
                    )
                    continue

                yield scrapy.Request(
                    url=url,
                    callback=self.parse_article,
                    errback=self.handle_article_error,
                    cb_kwargs={
                        "date": raw_date,
                        "link": url,
                    },
                )
        # do we need the next page?
        if next_page:
            self.page += 1
            next_link = (
                f"/mining-topics/topic/drill-results.html?page={self.page}&format=json"
            )
            next_url = response.urljoin(next_link)
            yield scrapy.Request(next_url, callback=self.parse)

    def parse_meta(self, response):
        """parse article meta data"""
        item_page = response.css("div.item-page")
        info = item_page.css("div.js-tag-info a::text").getall()
        info = [x.strip() for x in info]
        ticker = info[-2] if len(info) >= 2 else None
        links = item_page.css("div.js-tag-info a::attr(href)").getall()
        web = next((x for x in links if "html" not in x), None)
        return ticker, web

    def parse_stock_quote(self, response):
        """parse article financial information"""
        quotes = response.css("table.stock-quote-module").get()
        if quotes is None:
            return None, None

        quote_tables = self._safe_read_html(quotes)
        if len(quote_tables) == 0:
            return None, None

        quotes = pd.concat(quote_tables, ignore_index=True)
        quotes = quotes.set_index(0)
        last_trade = (
            quotes.loc["Last Trade:", 1] if "Last Trade:" in quotes.index else None
        )
        market_cap = (
            quotes.loc["Market Cap:", 1] if "Market Cap:" in quotes.index else None
        )
        return last_trade, market_cap

    def parse_tabular_intervals(self, response):
        """parse tabular interval data from the article"""

        # get the part of the page with the tabular data
        tables = response.css("table")
        if len(tables) == 0:
            return self.parse_unstructured_intervals(response)

        # check the tables for ddh data and process them
        dfs = []
        for tab in tables:
            table_html = tab.get()
            if table_html is None:
                continue

            candidates = self._safe_read_html(table_html.replace(",", "."), header=0)

            if len(candidates) == 0:
                fallback_table = self._manual_parse_html_table(tab)
                if fallback_table is not None:
                    candidates = [fallback_table]

            for table in candidates:
                table = self._clean_interval_table(table)
                if self._is_interval_table(table):
                    table["parse_source"] = "table_html"
                    dfs.append(table)

        if len(dfs) == 0:  # no ddh related tables
            return self.parse_unstructured_intervals(response)
        else:
            df = pd.concat(dfs, ignore_index=True)
            df = df.dropna(how="all")
            return df

    def _clean_interval_table(self, df):
        """Standardize columns and strip empty records from parsed tables."""
        table = df.copy()
        table.columns = [self._normalize(str(c)) for c in table.columns]

        for col in table.columns:
            if table[col].dtype == object:
                table[col] = table[col].astype(str).str.strip()
                table[col] = table[col].replace({"": np.nan, "nan": np.nan})

        table = table.dropna(how="all")
        return table

    def _manual_parse_html_table(self, tab):
        """Fallback parser that builds a DataFrame from <tr>/<th>/<td> cells."""
        rows = tab.css("tr")
        if len(rows) == 0:
            return None

        extracted = []
        for row in rows:
            cells = row.css("th::text, td::text").getall()
            cells = [c.strip() for c in cells if c.strip()]
            if len(cells) > 0:
                extracted.append(cells)

        if len(extracted) < 2:
            return None

        widths = [len(r) for r in extracted]
        n_cols = max(widths)
        header_idx = widths.index(n_cols)

        header = extracted[header_idx]
        if len(header) < 2:
            return None

        body = []
        for r in extracted[header_idx + 1 :]:
            padded = r + [None] * (n_cols - len(r))
            body.append(padded[:n_cols])

        if len(body) == 0:
            return None

        return pd.DataFrame(body, columns=header)

    def _is_interval_table(self, df):
        """Heuristic check for interval tables (from/to + at least one grade-like column)."""
        cols = list(df.columns)
        norm_cols = [self._normalize(str(c)) for c in cols]

        from_cols = [c for c in norm_cols if c.startswith("from")]
        to_cols = [c for c in norm_cols if c.startswith("to")]
        length_cols = [
            c
            for c in norm_cols
            if any(token in c for token in ["interval", "length", "width"])
        ]

        grade_like = [
            c
            for c in norm_cols
            if re.search(r"\b(au|ag|pt|pd|cu|zn|pb|ni|co|mo|u308)\b", c, re.I)
        ]

        has_depth = (len(from_cols) > 0 and len(to_cols) > 0) or len(length_cols) > 0
        return has_depth and len(grade_like) > 0

    def parse_unstructured_intervals(self, response):
        """Regex fallback for unstructured interval text."""
        text = " ".join(response.css("p ::text, li ::text").getall())
        text = re.sub(r"\s+", " ", text)

        pattern = re.compile(
            r"from\s*(?P<from>\d+(?:\.\d+)?)\s*(?:m|meters?)\s*"
            r"(?:to|-)+\s*(?P<to>\d+(?:\.\d+)?)\s*(?:m|meters?).{0,80}?"
            r"(?P<grade>\d+(?:\.\d+)?)\s*(?P<unit>g/t|gpt|%|ppm|ppb)\s*"
            r"(?P<elem>Au|Ag|Pt|Pd|Cu|Zn|Pb|Ni|Co|Mo|U308)",
            re.IGNORECASE,
        )

        records = []
        for m in pattern.finditer(text):
            start = float(m.group("from"))
            end = float(m.group("to"))
            records.append(
                {
                    "from": start,
                    "to": end,
                    "length": end - start,
                    f"{m.group('elem')} ({m.group('unit')})": float(m.group("grade")),
                    "parse_source": "regex_text",
                }
            )

        if len(records) == 0:
            return None
        return pd.DataFrame(records)

    def parse_tabular_drillholes(self, response):
        """parse tabular drillhole data (survey and collars) from the article"""
        tables = response.css("table")
        if len(tables) == 0:
            return None

        ddh_frames = []
        for tab in tables:
            table_html = tab.get()
            if table_html is None:
                continue

            candidates = self._safe_read_html(table_html.replace(",", "."), header=0)
            if len(candidates) == 0:
                fallback_table = self._manual_parse_html_table(tab)
                if fallback_table is not None:
                    candidates = [fallback_table]

            for table in candidates:
                table = self._clean_interval_table(table)
                if self._is_drillhole_table(table):
                    ddh_frames.append(self._standardize_drillhole_table(table))

        if len(ddh_frames) == 0:
            return None

        df_ddh = pd.concat(ddh_frames, ignore_index=True)
        df_ddh = df_ddh.dropna(subset=["hole_id"], how="any")
        df_ddh = df_ddh.drop_duplicates(subset=["hole_id"], keep="first")

        numeric_cols = ["azimuth", "inclination", "easting", "northing", "elevation"]
        for col in numeric_cols:
            if col in df_ddh.columns:
                df_ddh[col] = pd.to_numeric(df_ddh[col], errors="coerce")

        return df_ddh if len(df_ddh) > 0 else None

    def _is_drillhole_table(self, df):
        """Heuristic check for collar/survey tables with spatial drillhole fields."""
        cols = [self._normalize(str(c)) for c in df.columns]

        has_hole = any("hole id" in c or c == "hole" for c in cols)
        has_orientation = any("azimuth" in c or "bearing" in c for c in cols) and any(
            any(token in c for token in ["dip", "inclination", "incl"])
            for c in cols
        )
        has_xy = any("easting" in c or c in {"x", "xcoord"} for c in cols) and any(
            "northing" in c or c in {"y", "ycoord"} for c in cols
        )

        return has_hole and (has_orientation or has_xy)

    def _standardize_drillhole_table(self, df):
        """Rename drillhole table columns to canonical names used downstream."""
        table = df.copy()

        rename_map = {}
        for col in table.columns:
            ncol = self._normalize(str(col))

            if "hole id" in ncol or ncol == "hole":
                rename_map[col] = "hole_id"
            elif "azimuth" in ncol or "bearing" in ncol:
                rename_map[col] = "azimuth"
            elif any(token in ncol for token in ["inclination", "dip", "incl"]):
                rename_map[col] = "inclination"
            elif "easting" in ncol or ncol in {"x", "xcoord"}:
                rename_map[col] = "easting"
            elif "northing" in ncol or ncol in {"y", "ycoord"}:
                rename_map[col] = "northing"
            elif "elevation" in ncol or "rl" == ncol:
                rename_map[col] = "elevation"

        table = table.rename(columns=rename_map)

        keep = [
            c
            for c in ["hole_id", "azimuth", "inclination", "easting", "northing", "elevation"]
            if c in table.columns
        ]
        table = table[keep]

        if "hole_id" in table.columns:
            table["hole_id"] = table["hole_id"].astype(str).str.strip()
            table["hole_id"] = table["hole_id"].replace({"": np.nan, "nan": np.nan})

        return table

    def merge_significant_intervals_with_drillholes(
        self, df_ddh, intervals_df, significant_df, depth_tolerance=1e-3
    ):
        """
        Join significant grade intervals to drillhole collar/orientation data and
        desurvey from/to downhole depths into XYZ coordinates.
        """
        if df_ddh is None or intervals_df is None or significant_df is None:
            return None

        if len(df_ddh) == 0 or len(intervals_df) == 0 or len(significant_df) == 0:
            return None

        intervals = intervals_df.copy()
        significant = significant_df.copy()

        if "hole_id" not in intervals.columns:
            hole_cols = [
                c
                for c in intervals.columns
                if self._normalize(str(c)) in {"hole id", "holeid", "hole"}
            ]
            if len(hole_cols) > 0:
                intervals = intervals.rename(columns={hole_cols[0]: "hole_id"})

        required = {"hole_id", "from", "to", "AuEQ"}
        if not required.issubset(set(intervals.columns)):
            return None

        if not {"from", "to", "AuEQ"}.issubset(set(significant.columns)):
            return None

        for col in ["from", "to", "AuEQ"]:
            intervals[col] = pd.to_numeric(intervals[col], errors="coerce")
            significant[col] = pd.to_numeric(significant[col], errors="coerce")

        intervals["hole_id"] = intervals["hole_id"].astype(str).str.strip()

        sig_keys = significant[["from", "to", "AuEQ"]].dropna().drop_duplicates()
        intervals["_k_from"] = np.round(intervals["from"], 3)
        intervals["_k_to"] = np.round(intervals["to"], 3)
        intervals["_k_au"] = np.round(intervals["AuEQ"], 3)
        sig_keys["_k_from"] = np.round(sig_keys["from"], 3)
        sig_keys["_k_to"] = np.round(sig_keys["to"], 3)
        sig_keys["_k_au"] = np.round(sig_keys["AuEQ"], 3)

        sig_intervals = intervals.merge(
            sig_keys[["_k_from", "_k_to", "_k_au"]],
            on=["_k_from", "_k_to", "_k_au"],
            how="inner",
        ).drop(columns=["_k_from", "_k_to", "_k_au"])

        merged = sig_intervals.merge(df_ddh, on="hole_id", how="inner")
        if len(merged) == 0:
            return None

        for col in ["from", "to", "azimuth", "inclination", "easting", "northing", "elevation"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

        merged = merged.dropna(
            subset=["from", "to", "azimuth", "inclination", "easting", "northing"]
        )
        if len(merged) == 0:
            return None

        az = np.deg2rad(merged["azimuth"])
        dip = np.deg2rad(merged["inclination"])

        cos_dip = np.cos(dip)
        sin_dip = np.sin(dip)

        merged["x_from"] = merged["easting"] + merged["from"] * cos_dip * np.sin(az)
        merged["y_from"] = merged["northing"] + merged["from"] * cos_dip * np.cos(az)
        merged["z_from"] = merged.get("elevation", 0.0) + merged["from"] * sin_dip

        merged["x_to"] = merged["easting"] + merged["to"] * cos_dip * np.sin(az)
        merged["y_to"] = merged["northing"] + merged["to"] * cos_dip * np.cos(az)
        merged["z_to"] = merged.get("elevation", 0.0) + merged["to"] * sin_dip

        out_cols = [
            c
            for c in [
                "hole_id",
                "from",
                "to",
                "AuEQ",
                "x_from",
                "y_from",
                "z_from",
                "x_to",
                "y_to",
                "z_to",
            ]
            if c in merged.columns
        ]

        return merged[out_cols].reset_index(drop=True)

    def parse_article(self, response, date, link):
        """parse article information and yield requested tables"""

        # get the part of the page with the tabular data
        full_page = response.css("div.row.row-flex.row-large")
        item_page = full_page.css("div.item-page")

        # meta is item_page, quotes are full_page, tables are item_page
        ticker, web = self.parse_meta(item_page)
        last_trade, market_cap = self.parse_stock_quote(full_page)
        intervals = self.parse_tabular_intervals(item_page)

        has_valid_tabular_data = (
            intervals is not None
            and "parse_source" in intervals.columns
            and (intervals["parse_source"] == "table_html").any()
        )

        if intervals is None:
            self._record_article_status(
                status="failed",
                reason="no_interval_data_extracted",
                article_link=link,
                article_date=date,
            )
            return

        sig_ints, commod = self.calc_significant_intercepts(intervals, self.price_dict)

        if sig_ints is None and has_valid_tabular_data:
            self._record_article_status(
                status="non_significant",
                reason="valid_tabular_data_below_significance_threshold",
                article_link=link,
                article_date=date,
            )
            return

        if sig_ints is not None:
            sig_ints = sig_ints.to_dict()  # scrapy needs a dictionary
            sig_ints["commodities"] = ", ".join(commod)
            sig_ints["ticker"] = ticker
            sig_ints["last_trade"] = last_trade
            sig_ints["market_cap"] = market_cap
            sig_ints["web"] = web
            sig_ints["article_date"] = date
            sig_ints["article_link"] = link
            yield sig_ints

    def handle_article_error(self, failure):
        """Capture request/response level failures for article pages."""
        req = failure.request
        self._record_article_status(
            status="failed",
            reason="request_error",
            article_link=req.url,
            article_date=req.cb_kwargs.get("date"),
            details=str(failure.value),
        )

    def closed(self, reason):
        """Write article parsing audit records at spider shutdown."""
        with open(self.audit_log_path, "w") as f:
            json.dump(self.article_audit_log, f, indent=4)

    def _record_article_status(
        self, status, reason, article_link=None, article_date=None, details=None
    ):
        record = {
            "status": status,
            "reason": reason,
            "article_link": article_link,
            "article_date": article_date,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if details is not None:
            record["details"] = details

        self.article_audit_log.append(record)

    @staticmethod
    def _is_valid_url(url):
        if not url:
            return False
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def calc_significant_intercepts(self, df, price_dict):
        """
        Return significant drill intercepts based on Au Eq using
        robust column identification instead of manual junk lists.
        """

        df = df.copy()
        all_cols = df.columns

        # --- helpers -------------------------------------------------

        def looks_like_grade(series):
            s = pd.to_numeric(series, errors="coerce")
            if s.notna().mean() < 0.6:
                return False
            if s.abs().max() > 1e6:
                return False
            if s.abs().mean() == 0:
                return False
            return True

        # --- regex rules --------------------------------------------

        GRADE_RE = re.compile(
            r"\b(Au|Ag|Pt|Pd|Cu|Zn|Pb|Ni|Co|Mo|U308)\b.*\b(g/t|gpt|ppm|ppb|%)\b",
            re.IGNORECASE,
        )

        ELEMENT_ONLY_RE = re.compile(
            r"\b(Au|Ag|Pt|Pd|Cu|Zn|Pb|Ni|Co|Mo|U308)\b",
            re.IGNORECASE,
        )

        DOMAIN_EXCLUDE_RE = re.compile(r"\beq\b", re.IGNORECASE)

        # --- normalize names ----------------------------------------

        norm = {c: self._normalize(c) for c in all_cols}

        # --- identify from / to -------------------------------------

        ifrom = [c for c, n in norm.items() if n.startswith("from")]
        ito = [c for c, n in norm.items() if n.startswith("to")]

        for c in ifrom + ito:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        try:
            df["length"] = df[ito[0]] - df[ifrom[0]]
        except Exception:
            df["length"] = np.nan

        # --- primary grade selection (element + unit) ----------------

        grade_cols = []
        for c, n in norm.items():
            if DOMAIN_EXCLUDE_RE.search(n):
                continue
            if GRADE_RE.search(n):
                grade_cols.append(c)

        # --- fallback: element only + numeric behavior ----------------

        for c, n in norm.items():
            if c in grade_cols:
                continue
            if DOMAIN_EXCLUDE_RE.search(n):
                continue
            if ELEMENT_ONLY_RE.search(n):
                if looks_like_grade(df[c]):
                    grade_cols.append(c)

        # --- coerce numerics ----------------------------------------

        for c in grade_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # --- split precious / base ----------------------------------

        precious_map = {"au", "ag", "pt", "pd"}
        base_map = {"cu", "zn", "pb", "ni", "co", "mo", "u308"}

        prec_cols = []
        base_cols = []

        for c in grade_cols:
            n = norm[c]
            for k in precious_map:
                if k in n:
                    prec_cols.append(c)
            for k in base_map:
                if k in n:
                    base_cols.append(c)

        # remove duplicates
        prec_cols = list(dict.fromkeys(prec_cols))
        base_cols = list(dict.fromkeys(base_cols))

        if len(prec_cols + base_cols) == 0:
            return None, None

        # --- compute equivalent grades -------------------------------

        equivalent, commod = self.equivalent_grade(df, prec_cols, base_cols, price_dict)

        sig_ints = equivalent.loc[
            equivalent["AuEQ*length"] >= 75,
            [c for c in ["hole_id", ifrom[0], ito[0], "length", "AuEQ"] if c in equivalent.columns],
        ].copy()

        rename_map = {}
        if len(ifrom) > 0 and ifrom[0] in sig_ints.columns:
            rename_map[ifrom[0]] = "from"
        if len(ito) > 0 and ito[0] in sig_ints.columns:
            rename_map[ito[0]] = "to"
        sig_ints = sig_ints.rename(columns=rename_map)

        sig_ints = sig_ints.reset_index(drop=True)

        if len(sig_ints) > 0:
            return sig_ints, commod
        else:
            return None, None

    def equivalent_grade(self, df, prec_cols, base_cols, price_dict):
        """
        Calculate Au-equivalent grades in a unit-aware, element-aware way.

        All grades are converted to AuEq g/t using:
        - element parsing from column names
        - unit parsing from column names
        - consistent internal element keys (Au, Ag, Cu, ...)

        Returns df with AuEQ and AuEQ*length plus list of contributing commodities.
        """

        grams_per_pound = 453.59237
        grams_per_toz = 31.1035

        # --- helpers -------------------------------------------------

        def parse_element_unit(col):
            """
            Extract canonical element and unit from column name.
            """
            n = self._normalize(col)

            elem_re = re.compile(r"\b(au|ag|pt|pd|cu|zn|pb|ni|co|mo|u308)\b", re.I)
            unit_re = re.compile(r"\b(g/t|gpt|ppm|ppb|%)\b", re.I)

            em = elem_re.search(n)
            um = unit_re.search(n)

            if not em:
                return None, None

            elem = em.group(1).title()
            unit = um.group(1).lower() if um else None

            # normalize synonyms
            if unit == "gpt":
                unit = "g/t"

            return elem, unit

        # --- build conversion table --------------------------------

        grade_map = []

        for col in prec_cols + base_cols:
            elem, unit = parse_element_unit(col)
            if elem is None:
                continue
            if elem not in price_dict:
                continue
            grade_map.append(
                {
                    "col": col,
                    "element": elem,
                    "unit": unit,
                }
            )

        if len(grade_map) == 0:
            df["AuEQ"] = 0.0
            df["AuEQ*length"] = 0.0
            return df, []

        # --- convert each column -----------------------------------

        contrib_cols = []

        for g in grade_map:
            col = g["col"]
            elem = g["element"]
            unit = g["unit"]

            s = pd.to_numeric(df[col], errors="coerce")

            # --- unit to g/t ----------------------------------------

            if unit in ("g/t", None):
                gpt = s

            elif unit == "ppm":
                gpt = s / 1000.0

            elif unit == "ppb":
                gpt = s / 1e6

            elif unit == "%":
                # % -> g/t
                gpt = s * 10000.0

            else:
                continue

            # --- price ratio to Au ----------------------------------

            if elem in ("Au", "Ag", "Pt", "Pd"):
                ratio = price_dict[elem] / price_dict["Au"]
            else:
                ratio = (price_dict[elem] / grams_per_pound) / (
                    price_dict["Au"] / grams_per_toz
                )

            df[f"_AuEQ_{col}"] = gpt * ratio
            contrib_cols.append(f"_AuEQ_{col}")

        # --- sum contributions -------------------------------------

        df["AuEQ"] = df[contrib_cols].sum(axis=1)
        df["AuEQ*length"] = df["AuEQ"] * df["length"]

        commodities = sorted({g["element"] for g in grade_map})

        return df, commodities

    @staticmethod
    def plot_scatter(df):
        fig, ax = plt.subplots(figsize=(6, 6))

        l1 = np.array([[1, 100], [100, 1]])
        l2 = np.array([[1, 500], [500, 1]])
        l3 = np.array([[1, 1000], [1000, 1]])

        # cmap = iter(plt.cm.tab10(np.linspace(0, 0.9, len(df["ticker"].unique()))))
        cmap = iter(sns.cubehelix_palette(n_colors=len(df["ticker"].unique())))

        for key, group in df.groupby(df["ticker"]):
            _ = group.plot.scatter(
                "length",
                "AuEQ",
                ax=ax,
                label=key,
                c=next(cmap),  # .reshape(1, -1),
                ec="k",
                alpha=1.0,
                s=50,
            )
        ax.plot(l1[:, 0], l1[:, 1], marker="None", label=None, c="0.7")
        ax.plot(l2[:, 0], l2[:, 1], marker="None", label=None, c="0.5")
        ax.plot(l3[:, 0], l3[:, 1], marker="None", label=None, c="0.3")
        ax.set_xlim(0.75, 1250)
        ax.set_ylim(0.75, 1250)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylabel("Au Equivalent (g/t)")
        ax.set_xlabel("Intercept Length (m)")
        ax.grid(ls=":")

        plt.savefig("significant_intercepts.png", bbox_inches="tight", dpi=300)

        return fig, ax

    def _normalize(self, col):
        "normalize column names"
        col = col.lower()
        col = col.replace(",", ".")
        col = re.sub(r"[^\w\s%/\.]", " ", col)
        col = re.sub(r"\s+", " ", col).strip()
        return col

    @staticmethod
    def _safe_read_html(html, **kwargs):
        """Read html tables and return an empty list on parser errors."""
        try:
            return pd.read_html(html, **kwargs)
        except ValueError:
            return []
