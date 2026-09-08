# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Browser smoke test for HIP EP Studio.

The API smoke test in `smoke.py` proves the server is correct; this one proves the
console actually renders against it. It drives a real browser, visits every view,
asserts the content that should appear, and fails on any JavaScript error or failed
network request, because a console that logs exceptions is broken even when it looks
fine in a screenshot.

    python tests/console.py --base http://127.0.0.1:11434
    python tests/console.py --headed --screenshots out/

Requires Playwright and a Chromium-family browser:

    pip install playwright
    playwright install chromium     # or pass --channel msedge to use installed Edge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - the message is the whole point
    sys.exit("playwright is not installed. Run: pip install playwright")

# Ignore noise the console cannot control: favicon-style requests and the aborted
# fetches that streaming views produce on navigation are not defects.
IGNORED_ERRORS = ("net::ERR_ABORTED", "favicon")


class Checks:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def ok(self, label: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"  ok    {label}")
        else:
            self.failed.append(label)
            print(f"  FAIL  {label}{f'  ({detail})' if detail else ''}")

    def report(self) -> int:
        total = self.passed + len(self.failed)
        print(f"\n{self.passed}/{total} checks passed")
        for label in self.failed:
            print(f"  failed: {label}")
        return 1 if self.failed else 0


def run(base: str, *, headed: bool, channel: str | None, screenshots: Path | None) -> int:
    checks = Checks()
    problems: list[str] = []

    with sync_playwright() as playwright:
        launch: dict = {"headless": not headed}
        if channel:
            launch["channel"] = channel
        try:
            browser = playwright.chromium.launch(**launch)
        except PlaywrightError as error:
            sys.exit(
                f"could not launch a browser: {error}\n"
                "Run 'playwright install chromium', or pass --channel msedge."
            )

        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on(
            "pageerror",
            lambda error: problems.append(f"page error: {error}"),
        )
        page.on(
            "console",
            lambda message: (
                message.type == "error"
                and not any(noise in message.text for noise in IGNORED_ERRORS)
                and problems.append(f"console error: {message.text}")
            ),
        )
        page.on(
            "requestfailed",
            lambda request: (
                not any(noise in (request.failure or "") for noise in IGNORED_ERRORS)
                and not any(noise in request.url for noise in IGNORED_ERRORS)
                and problems.append(f"request failed: {request.url} ({request.failure})")
            ),
        )

        def shot(name: str) -> None:
            if screenshots:
                screenshots.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshots / f"{name}.png"), full_page=False)

        def goto(route: str) -> None:
            page.goto(f"{base}/#/{route}", wait_until="load")
            page.wait_for_timeout(700)

        # `Element.append(null)` renders the word "null". It is an easy mistake to make
        # in a conditional child and an easy one to miss by eye, so assert against it.
        def stray_nulls() -> int:
            return page.evaluate(
                """() => {
                    const walker = document.createTreeWalker(
                        document.body, NodeFilter.SHOW_TEXT);
                    let count = 0;
                    while (walker.nextNode()) {
                        const text = walker.currentNode.nodeValue.trim();
                        if (text === "null" || text === "undefined") count += 1;
                    }
                    return count;
                }"""
            )

        # --- shell -------------------------------------------------------------
        print("\nshell")
        goto("chat")
        checks.ok("title", "HIP EP" in page.title(), page.title())
        checks.ok("sidebar rendered", page.locator("#nav a").count() == 7)
        checks.ok(
            "engine badge resolved",
            page.locator("#engine-badge").inner_text().strip().lower()
            in {"gpu engine", "replay engine"},
            page.locator("#engine-badge").inner_text(),
        )
        checks.ok(
            "device line populated",
            "detecting" not in page.locator("#device-line").inner_text().lower(),
            page.locator("#device-line").inner_text(),
        )
        checks.ok("load pill populated", "session" in page.locator("#load-pill").inner_text())

        # Some later checks depend on whether real weights are being executed: the
        # replay engine answers for any model, a GPU one needs an artifact on disk.
        engine_is_real = bool(
            page.request.get(f"{base}/v1/status").json().get("engine", {}).get("real")
        )
        print(f"  ..    engine is {'real' if engine_is_real else 'replay'}")

        # --- chat --------------------------------------------------------------
        print("\nchat")
        checks.ok(
            "model selector populated", page.locator(".chat-side select option").count() > 3
        )
        checks.ok("empty state shown", page.locator(".chat-log .empty").count() == 1)
        checks.ok("composer present", page.locator(".composer textarea").count() == 1)

        # Asking for a short greeting and then asserting a long answer made a correct
        # model fail the check: "Hello!" is six characters and exactly what was asked
        # for. The check is that a streamed reply renders, so the prompt has to be one
        # whose correct answer is a paragraph.
        page.fill(
            ".composer textarea",
            "In two or three sentences, explain what a GPU execution provider does.",
        )
        page.click(".composer-actions .btn-primary")
        page.wait_for_selector(".msg.assistant .md p", timeout=60_000)
        page.wait_for_function(
            "() => document.querySelector('.composer-actions .btn-primary')?.hidden === false",
            timeout=120_000,
        )
        checks.ok("user message rendered", page.locator(".msg.user").count() == 1)
        assistant = page.locator(".msg.assistant .md").inner_text()
        checks.ok("assistant replied", len(assistant) > 40, f"{len(assistant)} chars")
        checks.ok(
            "markdown rendered, not escaped",
            "<p>" not in assistant and page.locator(".msg.assistant .md p").count() >= 1,
        )
        metrics = page.locator(".live-metrics").inner_text()
        checks.ok(
            "ttft reported", "TTFT" in metrics and metrics.split("\n")[1] != "-", metrics
        )
        checks.ok(
            "per-turn metrics on message",
            "tok/s" in page.locator(".msg.assistant .meta").inner_text(),
        )
        shot("chat")

        # --- vision ------------------------------------------------------------
        print("\nvision")
        goto("vision")
        options = page.locator(".chat-side select option").all_inner_texts()
        checks.ok("vision models only", len(options) > 0, f"{len(options)} options")
        checks.ok("attach button visible", page.locator("text=Attach image").count() == 1)
        shot("vision")

        # --- models ------------------------------------------------------------
        print("\nmodels")
        goto("models")
        cards = page.locator(".model-card")
        checks.ok("model cards rendered", cards.count() >= 4, f"{cards.count()} cards")
        checks.ok("tier tabs rendered", page.locator(".tabs button").count() == 4)
        page.click(".tabs button:nth-child(2)")
        page.wait_for_timeout(300)
        checks.ok("vlm tab switches", page.locator(".model-card .tag-vlm").count() >= 1)
        page.click(".tabs button:nth-child(1)")
        page.wait_for_timeout(300)
        page.locator(".model-card").first.locator("text=Details").click()
        page.wait_for_selector(".modal", timeout=5_000)
        modal = page.locator(".modal").inner_text()
        checks.ok("model card modal opens", "Quantization" in modal and "Run it" in modal)
        checks.ok("no stray null in the model card", stray_nulls() == 0)
        shot("model-card")
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        checks.ok("modal closes", page.locator(".modal").count() == 0)
        shot("models")

        # --- benchmarks --------------------------------------------------------
        print("\nbenchmarks")
        goto("benchmark")
        checks.ok("certified table rendered", page.locator("table.data tbody tr").count() >= 4)
        checks.ok(
            "release header shown", "Release" in page.locator(".notice-card").inner_text()
        )
        shot("benchmarks")
        page.click(".tabs button:nth-child(2)")
        page.wait_for_timeout(400)
        checks.ok("run form rendered", page.locator("text=Run benchmark").count() == 1)

        # --- port --------------------------------------------------------------
        print("\nport")
        goto("port")
        checks.ok("port form rendered", page.locator("text=Start porting").count() == 1)
        checks.ok(
            "quark defaults preselected",
            page.locator("input[type=number]").first.input_value() == "512",
            page.locator("input[type=number]").first.input_value(),
        )
        recipes = page.locator(".card select option").all_inner_texts()
        checks.ok("recipe collection offered", len(recipes) >= 8, f"{len(recipes)} recipes")
        checks.ok(
            "generic recipe is the default",
            "round-to-nearest" in page.locator(".card select").input_value()
            or page.locator(".card select").input_value() == "int4-rtn-g128",
            page.locator(".card select").input_value(),
        )
        shot("port")

        ported = "console-test-port"
        page.fill("input[placeholder^='org/model']", f"hf-internal-testing/{ported}")
        page.fill("input[placeholder='Optional display name']", ported)
        page.click("text=Start porting")
        page.wait_for_selector(".job-panel", timeout=10_000)
        checks.ok("port job starts", page.locator(".stage-row").count() == 8)
        checks.ok(
            "no stray null in the job panel", stray_nulls() == 0, f"{stray_nulls()} found"
        )
        shot("port-running")
        # Scope to the live panel: the history table also carries state badges.
        page.wait_for_selector(
            ".job-head .badge-succeeded, .job-head .badge-failed", timeout=180_000
        )
        state = page.locator(".job-head .badge").inner_text()
        checks.ok("port job succeeds", "done" in state, state)
        result = page.locator(".job-result").inner_text()
        checks.ok("share command offered", "hipep pull" in result, result[:120])
        shot("port-done")

        goto("models")
        # The zoo repaints when the catalog reload triggered by the finished job lands,
        # which can be just after this navigation.
        try:
            page.wait_for_selector("text=your import", timeout=15_000)
            tagged = True
        except PlaywrightTimeout:
            tagged = False
        checks.ok("ported model appears in the zoo", tagged)

        # An imported model must be usable, not merely listed.
        goto("chat")
        options = page.locator(".chat-side select option").all_inner_texts()
        checks.ok(
            "ported model is selectable", any(ported in o for o in options), str(options[:3])
        )

        # A GPU host with no export toolchain simulates the port, which produces a
        # catalog entry and no weights. Asking it to generate is then not a fair check:
        # what has to be true is that the server explains itself instead of failing as
        # though the model were merely missing.
        local = next(
            (
                m
                for m in page.request.get(f"{base}/v1/local").json()["models"]
                if m["id"] == ported
            ),
            {},
        )
        weightless = local.get("extra", {}).get("simulated") and engine_is_real
        if weightless:
            reply = page.request.post(
                f"{base}/v1/chat/completions",
                data={"model": ported, "messages": [{"role": "user", "content": "Hi"}]},
            )
            detail = reply.text()
            checks.ok(
                "a weightless port says why it cannot serve",
                not reply.ok and "simulated port" in detail,
                f"{reply.status}: {detail[:160]}",
            )
        else:
            page.select_option(".chat-side select", ported)
            page.fill(".composer textarea", "Does my imported model answer?")
            page.click(".composer-actions .btn-primary")
            page.wait_for_selector(".msg.assistant .md p", timeout=60_000)
            page.wait_for_function(
                "() => document.querySelector('.composer-actions .btn-primary')?.hidden"
                " === false",
                timeout=120_000,
            )
            answer = page.locator(".msg.assistant .md").inner_text()
            checks.ok("ported model answers", len(answer) > 40, f"{len(answer)} chars")

        # Leave no trace: the imported model would otherwise persist in the data
        # directory and change what the next run sees.
        removed = page.request.delete(f"{base}/v1/local/{ported}")
        checks.ok("imported model can be removed", removed.ok, str(removed.status))

        # --- server ------------------------------------------------------------
        print("\nserver")
        goto("server")
        page.wait_for_timeout(2_500)  # one status tick
        stats = page.locator(".stat-card")
        checks.ok("status cards rendered", stats.count() >= 6, f"{stats.count()} cards")
        body = page.locator("#view").inner_text()
        checks.ok("client sessions listed", "Clients" in body)
        checks.ok("hardware reported", "operating system" in body and "python" in body)
        checks.ok("recent completions listed", "Recent completions" in body)
        checks.ok("no stray null on the server page", stray_nulls() == 0)
        shot("server")

        # --- develop -----------------------------------------------------------
        print("\ndevelop")
        goto("develop")
        checks.ok("sample tabs rendered", page.locator(".tabs button").count() >= 5)
        code = page.locator(".code-block pre").inner_text()
        checks.ok("python sample generated", "OpenAI(base_url=" in code)
        checks.ok("sample points at this server", base.split("//")[1] in code, code[:80])
        page.click(".tabs button:nth-child(3)")
        page.wait_for_timeout(300)
        cpp = page.locator(".code-block pre").inner_text()
        checks.ok("c++ sample renders", "http_stream.hpp" in cpp and "post_lines" in cpp)
        shot("develop")

        # --- theme -------------------------------------------------------------
        print("\ntheme")
        page.click("#theme-toggle")
        page.wait_for_timeout(200)
        checks.ok(
            "light theme applies",
            page.evaluate("() => document.documentElement.dataset.theme") == "light",
        )
        shot("develop-light")
        page.click("#theme-toggle")

        browser.close()

    print("\njavascript and network")
    checks.ok("no page or console errors", not problems, "; ".join(problems[:4]))

    return checks.report()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:11434")
    parser.add_argument("--headed", action="store_true", help="watch the run in a window")
    parser.add_argument(
        "--channel",
        help="browser channel to use instead of a downloaded Chromium, e.g. msedge",
    )
    parser.add_argument(
        "--screenshots",
        type=Path,
        help="directory to write one screenshot per view",
    )
    args = parser.parse_args()

    print(f"HIP EP Studio console test against {args.base}")
    return run(
        args.base.rstrip("/"),
        headed=args.headed,
        channel=args.channel,
        screenshots=args.screenshots,
    )


if __name__ == "__main__":
    raise SystemExit(main())
