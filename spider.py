import scrapy
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import fnmatch
import json
import logging
import re
import seaborn as sns

from datetime import datetime, date

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
            date = datetime.strptime(
                article["formatted_date"].strip(), "%B %d, %Y"
            ).date()
            delta = (self.today - date).days
            if delta <= self.n_previous_days:
                url = response.urljoin(article["link"])
                yield scrapy.Request(
                    url,
                    callback=self.parse_article,
                    cb_kwargs={
                        "date": article["formatted_date"],
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
        ticker = info[-2]
        links = item_page.css("div.js-tag-info a::attr(href)").getall()
        web = [x for x in links if "html" not in x][0]
        return ticker, web

    def parse_stock_quote(self, response):
        """parse article financial information"""
        quotes = response.css("table.stock-quote-module").get()
        quotes = pd.concat(pd.read_html(quotes), ignore_index=True)
        quotes = quotes.set_index(0)
        last_trade = quotes.loc["Last Trade:", 1]
        market_cap = quotes.loc["Market Cap:", 1]
        return last_trade, market_cap

    def parse_tabular_intervals(self, response):
        """parse tabular interval data from the article"""

        # get the part of the page with the tabular data
        tables = response.css("table")
        if len(tables) == 0:
            return None

        # check the tables for ddh data and process them
        dfs = []
        for tab in tables:
            table = pd.concat(
                pd.read_html(tab.get().replace(",", "."), header=0),
                ignore_index=True,
            )
            cols = table.columns
            ddh_related = fnmatch.filter(cols, "from*")
            if len(ddh_related) == 0:
                continue
            dfs.append(table)

        if len(dfs) == 0:  # no ddh related tables
            return None
        else:
            # TODO check if tables are the same shape before concat
            df = pd.concat(dfs, ignore_index=True)
            df = df.dropna(how="all")
            return df

    def parse_tabular_drillholes(self, response):
        """parse tabular drillhole data (survey and collars) from the article"""
        # TODO
        pass

    def parse_article(self, response, date, link):
        """parse article information and yield requested tables"""

        # get the part of the page with the tabular data
        full_page = response.css("div.row.row-flex.row-large")
        item_page = full_page.css("div.item-page")

        # meta is item_page, quotes are full_page, tables are item_page
        ticker, web = self.parse_meta(item_page)
        last_trade, market_cap = self.parse_stock_quote(full_page)
        intervals = self.parse_tabular_intervals(item_page)

        if intervals is not None:
            sig_ints, commod = self.calc_significant_intercepts(
                intervals, self.price_dict
            )
        else:
            sig_ints = None

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
        else:
            yield None

    def calc_significant_intercepts(self, df, price_dict):
        """
        Return significant drill intercepts based on Au Eq using
        robust column identification instead of manual junk lists.
        """

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
            ["length", "AuEQ"],
        ].copy()

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
