const { chromium } = require("playwright");
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");

const [spreadsheetId, gid, range, tsvPath] = process.argv.slice(2);

if (!spreadsheetId || !gid || !range || !tsvPath) {
  console.error("Usage: node paste_google_sheet_tsv.js <spreadsheetId> <gid> <range> <tsvPath>");
  process.exit(2);
}

const data = fs.readFileSync(tsvPath, "utf8");
const userDataDir = `${process.env.HOME}/Library/Application Support/Google/Chrome`;
const url = `https://docs.google.com/spreadsheets/d/${spreadsheetId}/edit#gid=${gid}&range=${encodeURIComponent(range)}`;

(async () => {
  execFileSync("pbcopy", { input: data });
  const context = await chromium.launchPersistentContext(userDataDir, {
    channel: "chrome",
    headless: false,
    viewport: { width: 1440, height: 900 },
  });
  const page = context.pages()[0] || await context.newPage();
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForTimeout(7000);
  await page.keyboard.press("Meta+V");
  await page.waitForTimeout(12000);
  await context.close();
})();
