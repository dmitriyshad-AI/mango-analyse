const { chromium } = require("playwright");
const { execFileSync } = require("node:child_process");

const spreadsheetId = "19l-3YPaqH75AS0kbr1wW4cy4W5K91Ck4EvJGhjUNicU";
const userDataDir = `${process.env.HOME}/Library/Application Support/Google/Chrome`;
const url = `https://docs.google.com/spreadsheets/d/${spreadsheetId}/edit#gid=1011&range=A1`;

(async () => {
  const context = await chromium.launchPersistentContext(userDataDir, {
    channel: "chrome",
    headless: false,
    viewport: { width: 1440, height: 900 },
  });
  const page = context.pages()[0] || await context.newPage();
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForTimeout(8000);
  execFileSync("pbcopy", { input: "PING_TEST_OCH_LSH" });
  await page.keyboard.press("Meta+V");
  await page.waitForTimeout(5000);
  await context.close();
})();
