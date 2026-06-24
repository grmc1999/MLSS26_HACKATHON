const puppeteer = require("puppeteer");
const path = require("path");

(async () => {
  const browser = await puppeteer.launch({ headless: "new", args: ["--no-sandbox"] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080 });
  await page.goto("http://localhost:3000", { waitUntil: "networkidle0", timeout: 30000 });
  await page.waitForSelector("h1,h2", { timeout: 10000 }).catch(() => {});
  await new Promise(r => setTimeout(r, 2000));
  await page.screenshot({ path: path.resolve(__dirname, "../flu_dashboard.png"), fullPage: true });
  await browser.close();
  console.log("Captured flu_dashboard.png");
})();
