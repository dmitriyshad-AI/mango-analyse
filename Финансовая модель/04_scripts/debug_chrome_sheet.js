const { chromium } = require("playwright");
const fs = require("node:fs/promises");

(async () => {
  const browser = await chromium.connectOverCDP("http://127.0.0.1:9222");
  const context = browser.contexts()[0];
  const page = context.pages()[0] || await context.newPage();
  await page.waitForTimeout(10000);
  console.log("url", page.url());
  console.log("title", await page.title());
  await page.screenshot({ path: "/tmp/och_lsh_test_sheet.png", fullPage: false });
  await browser.close();
})();
