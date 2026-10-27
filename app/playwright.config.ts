// End-to-end tests of the desktop app: see e2e/README.md.
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e/specs",
  globalSetup: "./e2e/global-setup.ts",
  globalTeardown: "./e2e/global-teardown.ts",
  // One app window: the specs run one after another, in file order.
  workers: 1,
  fullyParallel: false,
  timeout: 90_000,
  expect: { timeout: 10_000 },
  outputDir: "./e2e/.results",
  reporter: [["list"], ["html", { outputFolder: "./e2e/.report", open: "never" }]],
});
