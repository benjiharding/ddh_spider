import os
import pandas as pd
import json

from datetime import datetime
from pathlib import Path
from scrapy.crawler import CrawlerProcess
from spider import DrillholeDataSpider
from gmailer import GMailer

# CIBC Analyst consensus for 2021
price_dict = {
    "Au": 1949,
    "Ag": 25.15,
    "Cu": 3.12,
    "Co": 16.45,
    "Pb": 0.90,
    "Zn": 1.09,
    "Mo": 9.42,
    "Ni": 6.87,
    "Pd": 2200,
    "Pt": 927,
    "U308": 36,
}
# number of previous days to consider
n_previous_days = 0  # today

# is today a weekday?
today = datetime.now()
is_weekday = today.isoweekday() in range(1, 6)

# initalize spider
process = CrawlerProcess(
    settings={
        "FEEDS": {
            "significant_intercepts.json": {
                "format": "json",
                "overwrite": True,
                "indent": 4,
            },
            # f"s3://{os.environ['AWS_ACCESS_KEY_ID']}:{os.environ['AWS_SECRET_ACCESS_KEY']}@bharding-bucket/ddh_web_scrape/%(time)s.json": {
            #     "format": "json",
            #     "overwrite": True,
            #     "indent": 4,
            # },
        },
        "RANDOMIZE_DOWNLOAD_DELAY ": True,
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    }
)
# deploy spider
if is_weekday:

    # remove any existing exports
    file_paths = [
        Path("significant_intercepts.json"),
        Path("significant_intercepts.png"),
    ]
    for fp in file_paths:
        fp.unlink(missing_ok=True)

    # strat the crawler process
    process.crawl(
        DrillholeDataSpider, price_dict=price_dict, n_previous_days=n_previous_days
    )
    process.start()

    # load json output and plot DataFrame
    with open("significant_intercepts.json", "r") as f:
        intercepts = json.load(f)
    dfs = []
    for ints in intercepts:
        df = pd.DataFrame.from_dict(ints)
        dfs = dfs + [df]
    df = pd.concat(dfs, ignore_index=True)
    fig, ax = DrillholeDataSpider.plot_scatter(df)

    # send the email
    mail = GMailer(
        "Significant Drillhole Intercepts for " + datetime.now().strftime("%Y/%m/%d"),
        [
            "b.e.harding@gmail.com",
            # "battymatthew@gmail.com",
            # "steven.mancell@gmail.com",
            # "jackie.rhind@gmail.com",
        ],
        os.environ["GMAIL_SENDER"],
        os.environ["GMAIL_SECRET_KEY"],
    )
    mail.htmladd(
        "Here are the significant diamond drillhole intercepts for "
        + datetime.now().strftime("%Y/%m/%d")
    )
    mail.htmladd('<img src="cid:significant_intercepts.png" width="500" height="500"/>')
    mail.htmladd("An intercept is considered significant if AuEQ*Length >= 75")
    mail.htmladd("Beep boop beep I am a bot")
    mail.addattach(["significant_intercepts.png", "significant_intercepts.json"])
    mail.send()
else:
    pass
