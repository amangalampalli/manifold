import { mkdir } from "node:fs/promises";
import { chromium } from "playwright";

const outputDir = new URL("../test-results/", import.meta.url);
await mkdir(outputDir, { recursive: true });

const browser = await chromium.launch();
const viewports = [
  ["desktop", { width: 1440, height: 900 }],
  ["mobile", { width: 390, height: 844 }]
];

for (const [name, viewport] of viewports) {
  const page = await browser.newPage({ viewport });
  await page.goto("http://127.0.0.1:5173", { waitUntil: "domcontentloaded" });
  await page.waitForSelector("svg", { timeout: 15000 });
  await page.waitForTimeout(2500);
  const result = await page.evaluate(() => {
    const svgs = [...document.querySelectorAll("svg")];
    const maps = svgs.map((svg) => ({
      nodes: svg.querySelectorAll("circle[fill]").length,
      edges: svg.querySelectorAll("line").length,
      labels: svg.querySelectorAll("text").length
    }));
    const best = maps.sort((left, right) => right.nodes - left.nodes)[0];
    if (!best) {
      return { ok: false, reason: "missing svg", nodes: 0, edges: 0, labels: 0 };
    }
    return {
      ok: best.nodes >= 40 && best.edges > 20 && best.edges < 320 && best.labels >= 10,
      reason: "svg threshold",
      ...best
    };
  });
  await page.screenshot({ path: new URL(`${name}.png`, outputDir).pathname, fullPage: true });
  if (!result.ok) {
    throw new Error(`${name} map check failed: ${JSON.stringify(result)}`);
  }
  const bodyText = await page.locator("body").innerText();
  if (bodyText.toLowerCase().includes(`open${String.fromCharCode(45)}loop`)) {
    throw new Error(`${name} page contains forbidden mode copy`);
  }
  console.log(`${name}: nodes=${result.nodes}, edges=${result.edges}, labels=${result.labels}`);
  await page.close();
}

await browser.close();
