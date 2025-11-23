#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json
import time
import csv
import random
from typing import List, Dict, Any

# ======================== 配置区域 ========================

URL = "https://www.hifleet.com/particulars/getShipDatav3"

# 1. 把浏览器 F12 里最新的 Cookie 粘到这里
COOKIE_STR = (
    "JSESSIONID=6FF8AFE993DCFC96F22E522142F083A9; HFJSESSIONID=69A72EC8975F56219EDEBB2E4F4C77C0; TGC=TGT-171780-RCFHjv9wm6WYWeDtILoo0soliTCoonEpbF6EWYP1MpNMQBVQfRv-Ut5-315EP9CZ3DciZ1y6w208fk1crZ"
)

# 2. 是否使用本机代理（如果你用 127.0.0.1:7890 就开 True）
USE_PROXY = False
PROXIES = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

# 3. 分页设置：每页 100 条，大约 14640 条 ≈ 147 页
LIMIT = 100
TOTAL_PAGES = 7  # 如果只是测试，可以先改小一点，比如 5、10

# 4. 访问间隔（随机 5~10 秒）
SLEEP_MIN = 5.0
SLEEP_MAX = 10.0

# 5. 请求头（模仿浏览器）
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Origin": "https://www.hifleet.com",
    "Referer": "https://www.hifleet.com/vessels/",
    "X-Requested-With": "XMLHttpRequest",
    "Cookie": COOKIE_STR,
}

# 固定查询条件：只要散货船
BASE_PARAMS = {
    "shipname": "",
    "callsign": "",
    "shiptype": "其他类型液货船",  # ⭐ 只拉散货船
    "shipflag": "",
    "keyword": "",
    "mmsi": -1,
    "imo": -1,
    "shipagemin": -1,
    "shipagemax": -1,
    "loamin": -1,
    "loamax": -1,
    "dwtmin": -1,
    "dwtmax": -1,
    "sortcolumn": "shipname",
    "sorttype": "asc",
    "isFleetShip": 0,
}


# ======================== 工具函数 ========================

def fetch_page(session: requests.Session, page_index: int, limit: int = 100) -> Dict[str, Any]:
    """
    抓一页数据：
    - page_index: 页码，从 1 开始
    - limit: 每页条数
    """
    payload = {
        "offset": page_index,  # ⭐ offset 是页码：1,2,3,...
        "limit": limit,
        "_v": "5.3.588",
        "params": BASE_PARAMS,
    }

    kwargs = {
        "url": URL,
        "headers": HEADERS,
        "json": payload,
        "timeout": 20,
    }
    if USE_PROXY:
        kwargs["proxies"] = PROXIES

    resp = session.post(**kwargs)

    print(f"  -> HTTP {resp.status_code}")

    if resp.status_code != 200:
        print("  !! 非 200 状态码，响应内容前 300 字符：")
        print(resp.text[:300])
        return {}

    try:
        data = resp.json()
    except Exception:
        print("  !! 无法解析为 JSON，响应内容前 300 字符：")
        print(resp.text[:300])
        return {}

    return data


def extract_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从返回 JSON 中提取船舶列表。
    一般结构类似：{"status":"1","rows":[...],"total":14640,...}
    """
    if not isinstance(data, dict):
        return []

    status = data.get("status")
    if status not in ("1", 1, "success", True, None):
        print("  !! status 非正常：", status, "msg:", data.get("msg"))
        # 比如 402 就会走这里，然后返回空，主循环会提前停止
        return []

    rows = data.get("rows")
    if isinstance(rows, list):
        return rows

    # 兜底：有些接口可能包一层 data
    inner = data.get("data")
    if isinstance(inner, dict) and isinstance(inner.get("rows"), list):
        return inner["rows"]
    if isinstance(inner, list):
        return inner

    return []


def save_to_csv(rows: List[Dict[str, Any]], filename: str) -> None:
    if not rows:
        print("没有数据可写入 CSV。")
        return

    # ✅ 用所有行的 key 做表头，保证包含 flagnameCN 等所有字段
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)  # 排个序，顺眼一点

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"已保存 {len(rows)} 条记录到 {filename}")


# ======================== 主流程 ========================

def main() -> None:
    session = requests.Session()
    all_rows: List[Dict[str, Any]] = []

    for page in range(1, TOTAL_PAGES + 1):
        print(f"===> 正在抓取第 {page}/{TOTAL_PAGES} 页 (每页 {LIMIT} 条) ...")

        data = fetch_page(session, page_index=page, limit=LIMIT)
        if not data:
            print("  !! 本页返回为空/解析失败，提前停止。")
            break

        rows = extract_rows(data)
        if not rows:
            print("  !! 本页未提取到任何记录（可能是 402 等限制），提前停止。")
            break

        print(f"  -> 本页获取到 {len(rows)} 条记录。")
        all_rows.extend(rows)

        # 随机休眠 5~10 秒
        sleep_seconds = random.uniform(SLEEP_MIN, SLEEP_MAX)
        print(f"  -> 休眠 {sleep_seconds:.1f} 秒后再请求下一页 ...")
        time.sleep(sleep_seconds)

    print(f"总计获取到 {len(all_rows)} 条记录。")

    if all_rows:
        save_to_csv(all_rows, "yhc.csv")


if __name__ == "__main__":
    main()
